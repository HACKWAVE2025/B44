import file_scanning
import threading
from werkzeug.utils import secure_filename
import queue
from flask import Flask, request, jsonify, send_from_directory, Response, stream_with_context, g, session
from flask_session import Session
from flask_caching import Cache
import os
import re
import time
import json
import random
import logging
import sqlite3
import traceback
import uuid
from pathlib import Path
from google import genai
import pypandoc
from dotenv import load_dotenv
import webscrapper
from tavily import TavilyClient
import datetime

naw = datetime.datetime.now()
script_dir = Path(__file__).resolve().parent
keys_env_path = script_dir / 'keys.env'
if keys_env_path.is_file():
    load_dotenv(dotenv_path=keys_env_path)

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'txt', 'pdf','docx', 'png', 'jpg', 'jpeg', 'gif', 'csv', 'md', 'py', 'js', 'html', 'css', 'json', 'xml', 'log', 'c', 'cpp', 'java', 'rb', 'php', 'go', 'rs', 'swift', 'kt','mp4','mp3'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

file_scan_keys_str = os.getenv("FILE_SCANNING_GEMINI_KEYS") # Use a default or raise error if needed
FILE_SCANNING_API_KEYS = [key.strip() for key in file_scan_keys_str.split(',') if key.strip()]
if not FILE_SCANNING_API_KEYS:
    logging.warning("FILE_SCANNING_GEMINI_KEYS not found in .env or empty. File analysis may fail.")
print(FILE_SCANNING_API_KEYS)

analysis_results_store = {}
analysis_results_lock = threading.Lock()

analysis_progress_queues = {}
analysis_progress_lock = threading.Lock()


app.secret_key = os.getenv("SECRET_KEY", "dev-secret-key-replace-in-prod")
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_FILE_DIR'] = './.flask_session/'
app.config['SESSION_COOKIE_NAME'] = 'stellar_session_test'



IS_PRODUCTION = os.getenv('FLASK_ENV') == 'production'
app.config['SESSION_COOKIE_SECURE'] = True


app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'

app.config['SESSION_PERMANENT'] = True
app.config['PERMANENT_SESSION_LIFETIME'] = datetime.timedelta(days=7)

Session(app)
os.makedirs(app.config['SESSION_FILE_DIR'], exist_ok=True)

cache = Cache(app, config={'CACHE_TYPE': 'simple'})

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_NAMES = {
    "gemini-2.0-flash-lite": "Emerald",
    "gemini-2.0-flash": "Lunarity",
    "gemini-2.0-flash-thinking-exp-01-21": "Crimson",
    "gemini-2.5-pro-exp-03-25": "Obsidian",
}
ERROR_CODE = "ERROR_CODE_ABC123XYZ456"

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
NEBULA_API_KEYS = {
    'step1': os.getenv("NEBULA_KEY_STEP1"),
    'step2': os.getenv("NEBULA_KEY_STEP2"),
    'step3': os.getenv("NEBULA_KEY_STEP3"),
    'step4': os.getenv("NEBULA_KEY_STEP4")
}
REFINE_API_KEY = os.getenv("REFINE_API_KEY")
SEARCH_API_KEY = os.getenv("SEARCH_API_KEY")
RTP_API_KEY = os.getenv("RTP_API_KEY")
NEBULA_COMPATIBLE_MODELS=["gemini-2.0-flash-thinking-exp-01-21", "gemini-2.5-pro-exp-03-25"]
#REFINE_API_KEY=RTP_API_KEY


DATABASE_FILE = 'conversation_history.db'


def get_current_session_id():
    if 'initialized' not in session:
        session['initialized'] = True
        logger.info(f"Flask-Session initialized new session: {session.sid}")
    return session.sid


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def run_file_analysis(session_id, filepath, filename):
    logger.info(f"Background Thread: Starting analysis for {filename} (Session: {session_id})")
    analyzer = None
    progress_q = None

    try:
        with analysis_progress_lock:
            if session_id not in analysis_progress_queues:
                analysis_progress_queues[session_id] = queue.Queue()
            progress_q = analysis_progress_queues[session_id]

        if not FILE_SCANNING_API_KEYS:
             raise ValueError("File scanning API keys are not configured.")

        analyzer = file_scanning.FileAnalyzer(session_id, FILE_SCANNING_API_KEYS, temp_base_folder=app.config['UPLOAD_FOLDER'])
        
        analysis_message_queue = analyzer.get_message_queue()


        analyzer.analyze_file(filepath)


        final_analysis_data = None
        while True:
            message = analysis_message_queue.get()
            if message is None:
                logger.info(f"Background Thread: Received end signal for {filename} (Session: {session_id})")
                break


            if progress_q:
                 try:
                     progress_q.put(message, block=False)
                     logger.debug(f"Background Thread: Put message type '{message.get('type')}' for {filename} onto SSE queue (Session: {session_id})")
                 except queue.Full:
                     logger.warning(f"Background Thread: SSE queue full for session {session_id}. Discarding message: {message.get('type')}")


            if message.get("type") == "file_complete":
                final_analysis_data = message
                analysis_text = message.get("combined_analysis", "[Analysis Error or No Content Retrieved]")
                status = message.get("status", "UNKNOWN")
                logger.info(f"Background Thread: Analysis complete for {filename}. Status: {status}. (Session: {session_id})")


                with analysis_results_lock:
                    if session_id not in analysis_results_store:
                        analysis_results_store[session_id] = {}
                    analysis_results_store[session_id][filename] = analysis_text
                    logger.info(f"Background Thread: Stored final analysis ({len(analysis_text)} chars) for {filename} (Session: {session_id})")

    except Exception as e:
        logger.error(f"Background Thread: Unhandled error during analysis of {filename} (Session: {session_id}): {e}\n{traceback.format_exc()}", exc_info=True)
        error_message_payload = {
            "type": "file_error",
            "session_id": session_id,
            "filename": filename,
            "error": f"Analysis process encountered a critical error: {str(e)}"
        }

        if progress_q:
             try:
                 progress_q.put(error_message_payload, block=False)
             except queue.Full:
                  logger.warning(f"Background Thread: SSE queue full for session {session_id} when reporting critical error for {filename}.")


        with analysis_results_lock:
            if session_id not in analysis_results_store:
                analysis_results_store[session_id] = {}
            analysis_results_store[session_id][filename] = f"[Analysis Failed Critically: {str(e)}]"

    finally:
        logger.info(f"Background Thread: Analysis thread finished for {filename} (Session: {session_id})")

        if progress_q:

             final_sse_msg = final_analysis_data if final_analysis_data else {"type": "analysis_thread_end", "filename": filename, "status": "Ended"}
             try:
                 progress_q.put(final_sse_msg, block=False)
             except queue.Full:
                 logger.warning(f"Background Thread: SSE queue full for session {session_id} when sending final status for {filename}.")


@app.route('/upload_files', methods=['POST'])
def upload_files():

    session_id = get_current_session_id()
    if not session_id:
        logger.error("Upload attempt failed - could not establish session.")
        return jsonify({'error': 'Session initialization failed. Please refresh.'}), 500

    uploaded_files = request.files.getlist("file")

    if not uploaded_files or all(f.filename == '' for f in uploaded_files):
        logger.warning(f"Upload request with no files selected for session {session_id}.")
        return jsonify({'error': 'No files selected'}), 400

    successful_uploads = []
    failed_uploads = []
    disallowed_file_types = []

    session_upload_folder = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
    os.makedirs(session_upload_folder, exist_ok=True)


    with analysis_progress_lock:
        if session_id not in analysis_progress_queues:
            analysis_progress_queues[session_id] = queue.Queue()

    for file in uploaded_files:
        if file and file.filename != '':
            if allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(session_upload_folder, filename)
                try:
                    file.save(filepath)
                    logger.info(f"File '{filename}' saved to '{filepath}' for session {session_id}")
                    successful_uploads.append(filename)

                except Exception as e:
                    logger.error(f"Error saving file {filename} (Session: {session_id}): {e}", exc_info=True)
                    failed_uploads.append(filename)

                    if os.path.exists(filepath):
                        try: os.remove(filepath)
                        except OSError: pass
            else:
                logger.warning(f"Upload attempt with disallowed file type: '{file.filename}' (Session: {session_id})")
                disallowed_file_types.append(file.filename)
        else:
             logger.debug(f"Skipping empty file part in upload request for session {session_id}")

    response_message = f"Processed upload request. Saved {len(successful_uploads)} allowed file(s)."
    if disallowed_file_types:
        response_message += f" Skipped {len(disallowed_file_types)} disallowed file type(s): {', '.join(disallowed_file_types)}."
    if failed_uploads:
        response_message += f" Failed to process {len(failed_uploads)} file(s): {', '.join(failed_uploads)}."

    status_code = 200 if successful_uploads else 400


    return jsonify({
        'status': response_message,
        'uploaded_files': successful_uploads,
        'files_disallowed': disallowed_file_types,
        'files_failed': failed_uploads
    }), status_code

@app.route('/analysis_progress')
def analysis_progress():

    session_id = get_current_session_id()
    if not session_id:
        logger.warning("SSE connection attempt failed - could not establish session.")

        return Response("data: {\"type\":\"error\", \"error\":\"Session initialization failed. Please refresh.\"}\n\n",
                        mimetype='text/event-stream', status=500)


    def generate_progress_stream():
        q = None
        with analysis_progress_lock:

            if session_id not in analysis_progress_queues:
                analysis_progress_queues[session_id] = queue.Queue()
                logger.info(f"SSE Stream: Created new progress queue for session {session_id}")
            q = analysis_progress_queues[session_id]

        logger.info(f"SSE Stream: Connection established for session {session_id}")

        yield f"data: {json.dumps({'type': 'sse_connected', 'session_id': session_id})}\n\n"

        keep_alive_counter = 0
        max_keep_alive_without_message = 4

        try:
            while True:
                try:

                    message = q.get(timeout=30)

                    if message is None:
                        logger.info(f"SSE Stream: Received None, ignoring. (Session: {session_id})")
                        continue

                    keep_alive_counter = 0


                    logger.debug(f"SSE Stream: Sending message to session {session_id}: Type '{message.get('type')}' for file '{message.get('filename', 'N/A')}'")
                    yield f"data: {json.dumps(message)}\n\n"


                    if message.get("type") == "file_complete" or message.get("type") == "analysis_thread_end":
                        logger.info(f"SSE Stream: Noted completion/end event for file '{message.get('filename')}' (Session: {session_id}). Stream remains open for other files/messages.")

                except queue.Empty:

                    keep_alive_counter += 1
                    if keep_alive_counter >= max_keep_alive_without_message:
                         logger.debug(f"SSE Stream: Sending keepalive for session {session_id}")
                         yield ": keepalive\n\n"
                         keep_alive_counter = 0
                    else:

                         pass
                    continue

                except Exception as e:

                     logger.error(f"SSE Stream: Error during message processing for session {session_id}: {e}", exc_info=True)
                     try:

                         yield f"data: {json.dumps({'type': 'sse_error', 'session_id': session_id, 'error': f'Stream error: {str(e)}'})}\n\n"
                     except Exception as send_err:
                         logger.error(f"SSE Stream: FAILED TO SEND error message to client {session_id}: {send_err}")
                     time.sleep(5)

        except GeneratorExit:

            logger.info(f"SSE Stream: Client disconnected for session {session_id}. Closing stream.")

        finally:

            logger.info(f"SSE Stream: Ending generate_progress_stream for session {session_id}.")


    return Response(stream_with_context(generate_progress_stream()), mimetype='text/event-stream')


def get_db():

    if 'db' not in g:
        g.db = sqlite3.connect(DATABASE_FILE)
        g.db.row_factory = sqlite3.Row
    return g.db

@app.teardown_appcontext
def close_db(error):

    db = g.pop('db', None)
    if db is not None:
        db.close()

def initialize_database():

    db = None
    try:
        db = sqlite3.connect(DATABASE_FILE)
        db.row_factory = sqlite3.Row
        cursor = db.cursor()


        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='messages'")
        table_exists = cursor.fetchone()


        required_cols = {
            'id': 'INTEGER PRIMARY KEY AUTOINCREMENT',
            'session_id': 'TEXT NOT NULL',
            'message_type': 'TEXT NOT NULL',
            'message_content': 'TEXT NOT NULL',
            'is_research_output': 'BOOLEAN DEFAULT 0',
            'html_file': 'TEXT',
            'nebula_step1': 'TEXT',
            'nebula_step2_frontend': 'TEXT',
            'nebula_step3_backend': 'TEXT',
            'nebula_step4_verification': 'TEXT',
            'file_analysis_context': 'TEXT',
            'timestamp': 'DATETIME DEFAULT CURRENT_TIMESTAMP'
        }
        schema_changed = False

        if not table_exists:

            cols_sql = ", ".join([f'"{name}" {definition}' for name, definition in required_cols.items()])
            cursor.execute(f'''CREATE TABLE messages ({cols_sql})''')
            logger.info("Created 'messages' table.")
            schema_changed = True
        else:

            cursor.execute("PRAGMA table_info(messages)")
            columns_info = cursor.fetchall()
            existing_columns = {row['name'].lower() for row in columns_info}

            for col_name, col_def in required_cols.items():
                if col_name.lower() not in existing_columns:
                    logger.info(f"Adding missing column: {col_name}")
                    try:

                        add_col_def = col_def.replace('NOT NULL', '')

                        if 'PRIMARY KEY' in col_def.upper() and col_name.lower() == 'id':
                             logger.warning(f"Column 'id' with PRIMARY KEY definition found missing in existing table. Schema evolution might be complex.")
                             continue


                        cursor.execute(f'ALTER TABLE messages ADD COLUMN "{col_name}" {add_col_def}')
                        schema_changed = True
                        logger.info(f"Added column {col_name}.")
                    except sqlite3.OperationalError as e:

                        logger.error(f"Could not add column {col_name}: {e}. Might already exist or DB locked.")


        cursor.execute("SELECT name FROM sqlite_master WHERE type='index' AND name='idx_session_id'")
        if not cursor.fetchone():
            cursor.execute('CREATE INDEX idx_session_id ON messages (session_id)')
            logger.info("Created index 'idx_session_id'.")
            schema_changed = True

        if schema_changed:
            db.commit()
            logger.info("Database schema verified/updated and committed.")
        else:
            logger.info("Database schema is up to date.")

    except sqlite3.Error as e:
        logger.error(f"Database initialization error: {e}")
        raise
    finally:
        if db:
            db.close()

initialize_database()

def insert_message(session_id, message_type, message_content,
                   is_research_output=False, html_file=None,
                   nebula_steps=None, file_analysis_context=None):
    if not session_id:
        logger.error("Attempted to insert message with no session_id.")
        return None
    try:
        db = get_db()
        nebula_data = nebula_steps or {}


        nebula_step1 = nebula_data.get('step1')
        nebula_step2 = nebula_data.get('step2')
        nebula_step3 = nebula_data.get('step3')
        nebula_step4 = nebula_data.get('step4')

        cursor = db.execute(
            '''INSERT INTO messages (session_id, message_type, message_content,
                                   is_research_output, html_file,
                                   nebula_step1, nebula_step2_frontend, nebula_step3_backend, nebula_step4_verification,
                                   file_analysis_context)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
            (session_id, message_type, message_content,
             is_research_output, html_file,
             nebula_step1, nebula_step2, nebula_step3, nebula_step4,
             file_analysis_context)
        )
        db.commit()
        last_id = cursor.lastrowid


        cache.delete_memoized(get_conversation_history, session_id)

        logger.info(f"Inserted message ID {last_id} for session {session_id}, type {message_type}")
        return last_id
    except sqlite3.Error as e:
        logger.error(f"Database error in insert_message for session {session_id}: {e}")
        return None


@cache.memoize(timeout=300)
def get_conversation_history(session_id):
    if not session_id:
        logger.warning("get_conversation_history called with no session_id.")
        return []
    try:
        db = get_db()
        cursor = db.execute(
            '''SELECT id, message_type, message_content, is_research_output, html_file,
                      nebula_step1, nebula_step2_frontend, nebula_step3_backend, nebula_step4_verification,
                      file_analysis_context, timestamp
               FROM messages WHERE session_id = ? ORDER BY timestamp ASC''',
            (session_id,)
        )
        rows = cursor.fetchall()

        # Check if history is empty AND no welcome message exists
        if not rows:
            logger.info(f"No history found for session {session_id}, inserting welcome message.")

            welcome_message = "Heyy there! I'm Stellar, and I can help you with research papers using Spectrum Mode, which includes Spectral Search! and building websites/apps with Nebula Mode! (Only exclusive to Crimson and Obsidian models). You can even Preview code blocks to see them live! I've got different models too, like Emerald for quick stuff or Obsidian for super complex things! ✨"
            welcome_id = insert_message(session_id, "stellar", welcome_message)

            if welcome_id:
                # Refetch after inserting welcome message
                cursor = db.execute(
                    '''SELECT id, message_type, message_content, is_research_output, html_file,
                              nebula_step1, nebula_step2_frontend, nebula_step3_backend, nebula_step4_verification,
                              file_analysis_context, timestamp
                       FROM messages WHERE session_id = ? ORDER BY timestamp ASC''',
                    (session_id,)
                )
                rows = cursor.fetchall()
            else:
                 logger.error(f"Failed to insert welcome message for session {session_id}")
                 return []


        history = []
        for row in rows:

            msg = dict(row)


            nebula_output_data = {
                'step1': msg.pop('nebula_step1', None),
                'step2': msg.pop('nebula_step2_frontend', None),
                'step3': msg.pop('nebula_step3_backend', None),
                'step4': msg.pop('nebula_step4_verification', None),
            }

            if any(v is not None for v in nebula_output_data.values()):
                msg['nebula_output'] = {k: v for k, v in nebula_output_data.items() if v is not None}


            if msg.get('message_type') == 'nebula_output' and msg.get('html_file'):

                 safe_filename = os.path.basename(msg['html_file'])
                 msg['report_url'] = f'/download/{safe_filename}'


            msg['id'] = str(msg['id'])



            history.append(msg)

        logger.info(f"History fetched for session {session_id}, {len(history)} messages.")
        return history
    except sqlite3.Error as e:
        logger.error(f"Database error in get_conversation_history for session {session_id}: {e}\n{traceback.format_exc()}")
        return []


def update_message(message_id, content):
    try:
        db = get_db()

        session_info = db.execute('SELECT session_id FROM messages WHERE id = ?', (message_id,)).fetchone()
        if not session_info:
            logger.error(f"Attempted to update non-existent message ID: {message_id}")
            return False

        session_id = session_info['session_id']


        db.execute('UPDATE messages SET message_content = ? WHERE id = ?', (content, message_id))
        db.commit()
        logger.info(f"Updated message ID {message_id} for session {session_id}")


        cache.delete_memoized(get_conversation_history, session_id)
        return True
    except sqlite3.Error as e:
        logger.error(f"Database error in update_message for ID {message_id}: {e}")
        return False


def sanitize_filename(filename: str) -> str:

    filename = filename.replace(' ', '_')

    sanitized = re.sub(r'[^\w\-\.]+', '', filename)

    return sanitized[:100] if len(sanitized) > 100 else sanitized


def tavily_search(query, search_depth="advanced", topic="general", time_range=None, max_results=15, include_images=False, include_answer="advanced"):
    try:
        if not TAVILY_API_KEY:
            logger.error("Tavily API Key is not configured.")
            return {"error": "Tavily search failed: API Key missing."}
        client = TavilyClient(TAVILY_API_KEY)
        response = client.search(
            query=query,
            search_depth=search_depth,
            topic=topic,
            max_results=max_results,
            time_range=time_range,
            include_images=include_images,
            include_answer=include_answer
        )

        return response
    except Exception as e:
        logger.error(f"Tavily search error for query '{query}': {e}", exc_info=True)
        return {"error": f"Tavily search failed: {str(e)}"}

def scrape_url(url: str) -> str:

    if not url or not url.startswith(('http://', 'https://')):
        logger.warning(f"Invalid URL provided for scraping: {url}")
        return f"Error scraping {url}: Invalid URL format"
    try:

        return webscrapper.scrape_url(url)
    except Exception as e:
        logger.error(f"Error scraping {url}: {e}", exc_info=True)
        return f"Error scraping {url}: {str(e)}"

stop_sequence="8919018818"


def classify_real_time_needed(query: str, key: str = None) -> str:

    query_lower = query.lower()

    check_segment = query_lower[:min(len(query_lower), 250)]
    logger.info(f"Checking real-time keywords in: '{check_segment[:60]}...'")


    real_time_keywords = [

        "latest", "current", "recent", "today", "now", "live", "ongoing", "update", "new", "breaking",
        "up-to-the-minute", "presently", "happening", "unfolding", "developments", "changes",
        "emerging", "novel", "trends", "upto date", "current edition",

        "verify", "fact check", "accurate", "true", "false", "confirm", "evidence", "sources",
        "reliable", "validate", "authenticate", "debunk",

        "look up", "find out", "define", "what is", "who is", "statistics", "data", "details",
        "specifics", "information on", "tell me about", "explain", "research", "report on",
        "compare", "vs", "versus", "stats",

        "financial", "stock", "market", "economic", "rates", "prices", "investment", "business",
        "weather", "news", "politics", "election", "sports score", "game result",

        "courses", "books", "material", "syllabus", "curriculum", "learning", "study guide",
        "tutorial", "documentation", "api reference",

        "which", "who", "when", "where", "how much", "cost of", "price of", "status of",

        "search for", "get me", "summarize article", "find paper"
    ]


    for keyword in real_time_keywords:

        if re.search(r'\b' + re.escape(keyword) + r'\b', check_segment, re.IGNORECASE):
            logger.info(f"Real-time keyword '{keyword}' found. Classifying as 'yes'.")
            return "yes"


    api_key = key or RTP_API_KEY
    if not api_key:
        logger.error("API key missing for real-time classification LLM call. Defaulting to 'no'.")
        return "no"


    model_name = 'gemini-2.0-flash-lite'
    client = None
    try:

        client = genai.Client(api_key=api_key, http_options={'api_version': 'v1alpha'})

        chat = client.chats.create(model=model_name, config={'tools': []})
    except Exception as e:
        logger.error(f"Could not create client/chat for model {model_name} (Real-time classification): {e}. Defaulting to 'no'.")


        return "no"


    prompt = crtp(query)
    logger.info(f"Sending classification prompt to {model_name}...")

    try:

        r = chat.send_message(prompt)


        if r.candidates and r.candidates[0].content and r.candidates[0].content.parts:
            response_text = r.candidates[0].content.parts[0].text.strip().lower()

            if "yes" in response_text:
                logger.info("LLM classified as needing real-time info ('yes').")
                return "yes"
            elif "no" in response_text:
                logger.info("LLM classified as *not* needing real-time info ('no').")
                return "no"
            else:

                logger.warning(f"Unexpected response from real-time classifier LLM ({model_name}): '{response_text}'. Defaulting to 'no'.")
                return "no"
        else:

            logger.error(f"Empty or invalid API response from real-time classifier ({model_name}). Finish Reason: {getattr(r, 'prompt_feedback', {}).get('finish_reason', 'N/A')}. Defaulting to 'no'. Response: {r}")
            return "no"
    except Exception as e:
        logger.error(f"Error during real-time classification LLM call ({model_name}): {e}. Defaulting to 'no'.", exc_info=True)
        return "no"


def gemini_generate(prompt: str, model_id: str, key: str, attempts: int = 3, backoff_factor: float = 1.5, model_display_name=None):

    display_name = model_display_name or MODEL_NAMES.get(model_id, model_id)
    last_exception = None
    current_prompt = prompt


    if model_id in ["gemini-2.0-flash-thinking-exp-01-21"]:
        logger.info(f"Checking if real-time data is needed for model {display_name}...")

        real_time_needed = classify_real_time_needed(current_prompt, RTP_API_KEY)
        logger.info(f"Real-time information needed for {display_name}: {real_time_needed}")

        if real_time_needed == "yes":
            yield {'status': f'Fetching real-time data...'}
            real_time_info_prompt = rtp(current_prompt)
            rtp_output = ""


            rtp_model_id = 'gemini-2.0-flash'
            rtp_key = RTP_API_KEY
            rtp_display_name = MODEL_NAMES.get(rtp_model_id, rtp_model_id)

            if not rtp_key:
                 logger.error("RTP_API_KEY is not set. Cannot fetch real-time data.")
                 yield {'status': f'Warning: Cannot fetch real-time data (API key missing).'}
            else:
                 logger.info(f"Fetching real-time info using {rtp_display_name}...")

                 rtp_generator = gemini_generate(
                     real_time_info_prompt,
                     rtp_model_id,
                     rtp_key,
                     attempts=2,
                     backoff_factor=1.75,
                     model_display_name=f"{rtp_display_name} (RTP Fetch)"
                 )

                 for item in rtp_generator:
                     if 'result' in item:
                         result_text = item['result']

                         if isinstance(result_text, str) and not result_text.startswith(ERROR_CODE):
                             rtp_output = result_text
                             logger.info(f"Successfully fetched real-time info ({len(rtp_output)} chars).")
                         else:
                             logger.error(f"RTP Fetch using {rtp_display_name} failed: {result_text}")
                         break
                     elif 'status' in item:

                         logger.info(f"RTP Fetch Status ({rtp_display_name}): {item['status']}")


                 if rtp_output:

                      current_prompt += f"\n\n---\n**ADDITIONAL REAL-TIME CONTEXT (Internal Use Only - Do Not Mention This Section Header in Final Output):**\n{rtp_output}\n---\n"
                      logger.info(f"Real-time data integrated into prompt for {display_name}.")
                      yield {'status': f'Real-time data fetched. Proceeding...'}
                 else:
                     logger.warning(f"Warning: Could not fetch real-time data for {display_name}. Proceeding without it.")
                     yield {'status': f'Warning: Proceeding without real-time data.'}
        else:
            logger.info(f"Real-time information not needed for {display_name}, proceeding directly.")



    for attempt in range(1, attempts + 1):
        try:
            yield {'status': f"{display_name} is thinking..."}
            if not key:

                raise ValueError(f"API key is missing for Gemini generation ({display_name}). Cannot proceed.")


            client = genai.Client(api_key=key, http_options={'api_version': 'v1alpha'})


            tools_config = []

            models_without_search = ["gemini-2.0-flash-lite", "gemini-2.0-flash-thinking-exp-01-21", "gemini-1.0-pro"]
            if model_id not in models_without_search:

                 search_tool = {'google_search': {}}
                 tools_config = [search_tool]
                 logger.info(f"Google Search tool potentially enabled for {display_name}")
            else:
                 logger.info(f"NOT using Google Search tool for {display_name}")


            chat = client.chats.create(model=model_id, config={'tools': tools_config})


            logger.debug(f"Sending prompt to {display_name} (Attempt {attempt}):\n{current_prompt[:300]}...")
            r = chat.send_message(current_prompt)
            logger.debug(f"Received response object from {display_name} (Attempt {attempt}).")



            output = ""


            if not r.candidates:
                 finish_reason_obj = getattr(r, 'prompt_feedback', {}).get('finish_reason', 'UNKNOWN')
                 finish_reason = finish_reason_obj.name if hasattr(finish_reason_obj, 'name') else str(finish_reason_obj)
                 safety_ratings = getattr(r, 'prompt_feedback', {}).get('safety_ratings', [])
                 safety_details = ", ".join([f"{sr.category.name}: {sr.probability.name}" for sr in safety_ratings if hasattr(sr, 'category') and hasattr(sr.category, 'name')]) if safety_ratings else "N/A"
                 error_msg = f"API Error ({display_name}): No candidates received. Finish Reason: {finish_reason}, Safety: {safety_details}"
                 logger.error(f"{error_msg}. Full Response: {r}")

                 if finish_reason == 'SAFETY':
                     last_exception = ValueError(f"Prompt blocked by API due to safety ({safety_details}).")

                     yield {'status': f'Prompt blocked due to safety. Retrying...'}
                     continue
                 elif finish_reason == 'RECITATION':
                     last_exception = ValueError("Prompt blocked by API due to recitation.")
                     yield {'status': f'Prompt blocked due to recitation. Retrying...'}
                     continue
                 else:

                     raise ValueError(error_msg)



            candidate = r.candidates[0]
            candidate_finish_reason_obj = getattr(candidate, 'finish_reason', 'UNKNOWN')
            candidate_finish_reason = candidate_finish_reason_obj.name if hasattr(candidate_finish_reason_obj, 'name') else str(candidate_finish_reason_obj)
            logger.info(f"Candidate finish reason for {display_name} (Attempt {attempt}): {candidate_finish_reason}")



            error_finish_reasons = ['SAFETY', 'RECITATION', 'OTHER']
            if candidate_finish_reason in error_finish_reasons:
                 candidate_safety_ratings = getattr(candidate, 'safety_ratings', [])
                 candidate_safety_details = ", ".join([f"{sr.category.name}: {sr.probability.name}" for sr in candidate_safety_ratings if hasattr(sr, 'category') and hasattr(sr.category, 'name')]) if candidate_safety_ratings else "N/A"
                 error_msg = f"Content generation stopped by API ({display_name}). Reason: {candidate_finish_reason}, Safety: {candidate_safety_details}"
                 logger.error(error_msg)
                 last_exception = ValueError(error_msg)

                 yield {'status': f'Content generation blocked ({candidate_finish_reason}). Retrying...'}
                 continue


            parts = getattr(candidate.content, 'parts', None)
            if parts is None:

                logger.warning(f"No parts found in candidate content for {display_name} (Finish Reason: {candidate_finish_reason}). Response: {candidate}")

                yield {'result': ""}
                return


            for part in parts:
                if hasattr(part, 'text') and part.text:
                    output += part.text
                elif hasattr(part, 'executable_code') and part.executable_code:

                    lang = part.executable_code.language.lower() if hasattr(part.executable_code, 'language') else 'python'
                    output += f"\n```python\n{part.executable_code.code}\n```\n"

                elif hasattr(part, 'function_call') and part.function_call:

                    logger.warning(f"Received unexpected function call part from {display_name}: {part.function_call.name}")
                    output += f"\n[Function Call: {part.function_call.name}]\n"

                elif hasattr(part, 'google_search_result') and part.google_search_result:

                     logger.info(f"Received raw Google Search result part for {display_name}.")

                     output += "\n[Google Search Result Data Received]\n"
                else:

                    logger.warning(f"Unsupported part type encountered from {display_name}: {type(part)}")
                    try:

                        dump = json.dumps(part.model_dump(exclude_none=True), indent=2)
                        output += f"\n```json\n# Unsupported Part Type\n{dump}\n```\n"
                    except Exception:
                        output += "\n[Unsupported/Undumpable part type]\n"



            grounding_metadata = getattr(candidate, 'grounding_metadata', None)
            if grounding_metadata:
                 logger.info(f"Grounding metadata found for {display_name}.")

                 search_entry = getattr(grounding_metadata, 'search_entry_point', None)
                 if search_entry and hasattr(search_entry, 'rendered_content') and search_entry.rendered_content:

                      output += f"\n\n---\n*Note: The following information may be based on or synthesized from Google Search results.*\n{search_entry.rendered_content}\n---\n"
                      logger.info("Rendered search content from grounding metadata included.")
                 elif hasattr(grounding_metadata, 'web_search_queries') and grounding_metadata.web_search_queries:
                     logger.info(f"Grounding metadata contains search queries: {grounding_metadata.web_search_queries}, but no rendered_content.")


                 else:
                     logger.info("Grounding metadata found but doesn't contain rendered search content or known query structure.")



            final_output = output.strip()


            if not final_output and candidate_finish_reason not in ['MAX_TOKENS', 'SAFETY', 'RECITATION']:
                 logger.warning(f"Generated output was empty for {display_name} despite finish reason '{candidate_finish_reason}'.")


            logger.info(f"Successfully generated response with {display_name} (Attempt {attempt}). Output length: {len(final_output)}")
            yield {'result': final_output}
            return

        except ValueError as ve:

            last_exception = ve
            logger.error(f"ValueError during gemini_generate attempt {attempt}/{attempts} for {display_name}: {ve}")
            yield {'status': f"Error: {str(ve)}"}

            if "API key is missing" in str(ve):
                 break


            if attempt < attempts:
                 delay = (backoff_factor ** (attempt - 1)) + random.uniform(0, 0.5)
                 logger.info(f"Retrying in {delay:.2f} seconds...")
                 time.sleep(delay)


        except Exception as e:

            last_exception = e
            logger.error(f"Unexpected error during gemini_generate attempt {attempt}/{attempts} for {display_name}: {e}\n{traceback.format_exc()}")
            yield {'status': f"Encountered an error, retrying..."}

            if attempt < attempts:
                delay = (backoff_factor ** (attempt - 1)) + random.uniform(0, 0.5)
                logger.info(f"Retrying in {delay:.2f} seconds...")
                time.sleep(delay)



    error_message = f"{ERROR_CODE}: Failed to generate response for {display_name} after {attempts} attempts. Last Error: {str(last_exception)}"
    logger.error(f"Final failure for {display_name}: {str(last_exception)}")
    yield {'result': error_message}


def create_output_file(query_or_base_name: str, content: str, extension: str = "txt") -> str | None:

    try:
        output_dir = "outputs"
        os.makedirs(output_dir, exist_ok=True)


        base_filename = sanitize_filename(query_or_base_name[:60].strip())
        if not base_filename:
            base_filename = "output"

        safe_filename = f"{base_filename}.{extension}"
        full_path = os.path.join(output_dir, safe_filename)
        counter = 1


        max_attempts_filename = 100
        while os.path.exists(full_path) and counter <= max_attempts_filename:
            safe_filename = f"{base_filename}_{counter}.{extension}"
            full_path = os.path.join(output_dir, safe_filename)
            counter += 1

        if counter > max_attempts_filename:
            logger.error(f"Could not find unique filename for base '{base_filename}.{extension}' after {max_attempts_filename} attempts.")
            return None


        max_write_attempts = 3
        for attempt in range(max_write_attempts):
            try:
                with open(full_path, "w", encoding="utf-8") as file:
                    file.write(content)
                logger.info(f"Successfully saved file: {full_path}")

                return os.path.join(output_dir, safe_filename)
            except IOError as e:
                logger.error(f"Error writing file '{full_path}' (Attempt {attempt+1}/{max_write_attempts}): {e}")
                if attempt < max_write_attempts - 1:
                    time.sleep(0.5 * (2 ** attempt))
                    continue
                else:
                    logger.error(f"Failed to write file '{full_path}' after {max_write_attempts} attempts.")

                    return None
            except Exception as e:
                 logger.error(f"Unexpected error writing file '{full_path}' (Attempt {attempt+1}): {e}", exc_info=True)

            if os.path.exists(full_path):
                try:
                    os.remove(full_path)
                except OSError:
                    pass
            return None

    except Exception as e:

        logger.error(f"Error in create_output_file setup for base '{query_or_base_name}': {e}", exc_info=True)
        return None


    return None


def rtp(alpha: str):

    return (
        f"Please provide the most current and factual real-time information regarding the following query. "
        f"Focus on verifiable data, statistics, recent developments, or official status updates. Cite reliable sources where possible.\n\n"
        f"Query: '{alpha}'"
    )

def crtp(beta: str):

    return (
        f"Analyze the user's query below. Does it require accessing information beyond general knowledge or historical data that doesn't change frequently? "
        f"Consider if the query involves any of the following:\n"
        f"*   **Current Events:** News, politics, ongoing situations, live updates.\n"
        f"*   **Recent Data:** Statistics, prices, market trends, scientific findings published recently.\n"
        f"*   **Fact-Checking:** Verifying specific claims, checking accuracy.\n"
        f"*   **Specific Entities:** Looking up details about specific people, organizations, products, or places where information might change.\n"
        f"*   **Dynamic Information:** Weather, stock prices, game scores.\n"
        f"*   **Resource Updates:** Current versions of software, documentation, course materials.\n"
        f"*   **Comparative/Evaluative:** Asking for the 'best' or 'latest' version/option.\n\n"
        f"Answer exactly 'yes' if the query *benefits significantly* from up-to-date or external information lookup. "
        f"Answer exactly 'no' if the query is purely creative, historical (without needing recent context), philosophical, or based on widely known, static facts.\n\n"
        f"User Query: '{beta}'\n\n"
        f"Classification (yes/no):"
    )

def get_refinement_prompt(user_query: str, conversation_history_list) -> str:
    conv_hist_str = "\n".join(conversation_history_list) if conversation_history_list else "No previous conversation turns."
    internal_guidelines_header = f"<!-- Internal Processing Guidelines (v1.2) -->"
    return (
        f"{internal_guidelines_header}\n"
        f"Role: You are Stellar, an advanced AI assistant specializing in research (Spectrum Mode) and web/app development (Nebula Mode for Crimson/Obsidian models). You offer features like Spectral Search (external info integration), code preview, and model selection (Emerald: Quick, Lunarity: Balanced, Crimson: Complex, Obsidian: VERY DRAMATICALLY POWERFUL).\n"
        "When giving out any code always Give the full code without any comments in the code unless specified other wise.\n"
        "All the modes are buttons which can be toggled on or off.\n"
        f"Interaction Style:\n"
        f"*   **Mirror User:** Adapt your tone, capitalization, slang, and energy to match the user's *current* message. Be conversational and natural.\n"
        f"*   **Direct Answers:** Respond directly to the user's query without unnecessary preface like 'Okay, here is...' or 'Stellar:'.\n"
        f"*   **Concise:** Avoid asking excessive clarifying questions unless absolutely necessary. Answer the query based on the information provided.\n"
        f"*   **Contextual:** If prior messages or research content exist, incorporate them naturally into the conversation flow.\n"
        f"*   **Emoji Use:** Use emojis sparingly, mirroring the user's frequency.\n"
        f"*   **Mode Awareness:** Briefly mention relevant modes (Spectrum, Nebula) if applicable to the user's request but don't over-explain unless asked. These modes are buttons which can be toggled on or off.\n"
        f"<!-- End Internal Guidelines -->\n\n"
        f"**Conversation History:**\n{conv_hist_str}\n\n"
        f"**Current User Query:** {user_query}\n\n"
        f"**Your Response:**"
    )


def get_research_analysis_prompt(query: str, full_context: str) -> str:
                                            
    return (
        "Using the following multi-source context, perform an exhaustive, research-level analysis. Based on the information provided, do your own research and fact-check everything. Return only the raw URLs (no HTML/CSS formatting). "
        "Your output should consist of two parts:\n\n"
        "1. Comprehensive Analysis: Synthesize the given information into a detailed review that serves as the backbone of a research paper. This analysis must include:\n"
        "- A literature review and background discussion.\n"
        "- Detailed technical and methodological explanations.\n"
        "- A critical evaluation of approaches, highlighting strengths and limitations.\n"
        "- Key findings and insights drawn from the data.\n"
        "- Potential future research directions and actionable recommendations.\n\n"
        "2. Prompt: Based on your analysis, generate a specific, refined prompt for another LLM to further expand on the topic. Analyze the topic and determine the appropriate academic structure for the research paper.\n"
        "- Identify the discipline (STEM, humanities, social sciences, business, or policy analysis).\n"
        "- Suggest a suitable formatting style (e.g., IMRaD, essay-style, executive summary).\n"
        "- Ensure your formatting aligns with academic best practices and citation standards. If any links are broken, mention only their titles without URLs.\n"
        "- Proceed with the comprehensive analysis using the recommended structure.\n\n"
        "This prompt should instruct the model to:\n"
        "- Act as a scientist or researcher and conduct further research on the topic.\n"
        "- Suggest 8-10 areas for further exploration.\n"
        "- Update technical details with the latest information.\n"
        "- Elaborate on methodologies and results.\n"
        "- Integrate recent developments and emerging trends, including a section for officially cited works and their descriptions.\n"
        "- Aim for a word count of approximately 5000 words or more.\n"
        "- Format the output as a structured research paper draft with detailed analysis.\n\n"
        "Ensure your response is formal, technically precise, and properly cited. "
        f"Additionally, include a section that evaluates the relevance of your analysis to the user's query: {query}\n"
        "Include a section with a novel solution for breakthrough research on the query, discussing feasibility.\n\n"
        f"Context:\n{full_context}\n"
        "Instruct the other AI to expand on everything to reach a minimum of 30,000 characters."
    )



def get_final_expansion_prompt(query: str, research_analysis_result: str, full_context: str) -> str:
                                          
    return (
        f"Include everything from the comprehensive analysis:\n{research_analysis_result}\n"
        "You are the LLM mentioned in the previous prompt. Follow its instructions but feel free to modify the format as needed. Respond directly without prefacing with phrases like 'Okay, here's the comprehensive research paper draft, as requested.' "
        "Expand on every aspect, ensuring that each paragraph introduces fresh, non-repetitive information. "
        "Include inline citations and a final list of references for all sourced information.\n\n"
        "Deliver the entire research paper in one output, ensuring thorough coverage of all sections. The paper should be academically rigorous, logically organized, and highly detailed.\n"
        "Incorporate additional research, including relevant case studies and empirical data.\n"
        "Adhere to academic writing standards and citation styles consistently.\n"
        "Include URLs where necessary but do not include any 'Hypothetical URL'; either show a URL or omit it.\n"
        "Integrate both qualitative and quantitative analyses where applicable.\n\n"
        f"Additionally, evaluate the relevance of your analysis to the user's query: {query}\n"
        "Include a section with a novel solution for breakthrough research on the query, discussing feasibility.\n\n"
        "Clearly demonstrate how the findings and methodologies address the user's needs.\n\n"
        f"Context:\n{full_context}\n\n"
        "Produce an original solution that is novel, relevant, accurate, and feasible, including:\n"
        "1. A comprehensive literature review summarizing the current state-of-the-art.\n"
        "2. A clear problem statement identifying an unresolved challenge.\n"
        "3. A novel theoretical framework with rigorous conceptual support.\n"
        "4. A detailed proposed methodology, including evaluation metrics.\n"
        "5. A feasibility analysis outlining technical challenges and mitigation strategies.\n"
        "6. An exploration of the broader impact and future directions.\n"
        "Search and include a section on market and industry insights such as market size, growth trends, key companies, and investment trends, supported by examples and data, please fact check this data again and again and make sure not to overestimate or underestimate anything.\n"
        "Finally, fact-check every piece of information before providing the output, and if any links are broken, mention only their titles without URLs.\n"
        "Do not include any 'Note:' stuff at the end of the paper, and DO NOT INLCUDE 'Okay, here is the comprehensive research paper draft, as requested'. no need to mention that you followed instructions and all."
    )



def get_nebula_step1_plan_prompt(user_plan: str, regeneration_feedback: str | None = None, web_context: str | None = None) -> str:
    feedback_section = ""
    wc=""
    if web_context:
        wc=(f"""**Pre-fetched Web Context (Use this for planning):**
---
{web_context}
---
**Instruction:** Base your plan on the user's vision AND the provided web context. Incorporate relevant facts, data, or ideas from the context.""")
    if regeneration_feedback:
        feedback_section = (
            f"\n\n**Regeneration Feedback:**\n"
            f"{regeneration_feedback}\n"
            f"**Instructions:** Please incorporate the above feedback when generating the plan.\n"
        )
    return (
       f"You are Stellar, You are the best Ai chat bot out there and you have acess to real time information so fecth the live information when needed based on todays date {naw},ONLY Google new llm is gemini 2.0 flash always use it whenever needed an llm for the project do not use it for every project just use it wherever you think a chat bot would be helpful. if the code requires realtime information like stocks and all please provide it directly and you can also use yfinace and all but like for current news and on going events please provide the information in the planning itself."
        f"**Nebula Code Generation - Step 1: Comprehensive Website Plan**\n\n"
        f"**User's Vision:**\n\n{user_plan}\n\n"
        f"{feedback_section}"
        f"{wc}"
        "**Guideline:** The user wants a **full, visually stunning website**, with premium significant animations, unique themes, and detailed content. Plan thoroughly to meet this vision. If the request implies complex features (frontend or backend), incorporate them into the plan.\n\n"
        f"**Your Task:** Outline a **detailed plan** to create a single HTML file (with embedded CSS/JS) and a supportive Python Flask backend (`app.py`) that fully realizes the user's request. **Avoid planning for placeholders; plan the actual content and features.**\n\n"
        "2.  **Serve Frontend:** **Crucially, include the route `@app.route('/')` to serve the `index.html` file** from the application's root directory always use send_from_directory to do this dont use anything else.\n"
        f"**Output Requirements:**\n"
        f"1.  **HTML File Outline (`index.html`):**\n"
        f"    *   **Structure:** Define the key sections and HTML elements needed (e.g., `<header>`, `<nav>`, specific `<section>` IDs for content, `<footer>`).\n"
        f"    *   **Content Ideas:** Briefly describe the *actual content* for each section (not just 'placeholder for intro').\n"
        f"    *   **Visuals/Animations:** Note where animations, specific visual styles, or interactive elements should be implemented. This needs to be done a lot whereever needed like educational content or the other stuff please make a lot animated demos.\n"
        f"    *   **Core JS Functions:** Outline necessary JS functions (e.g., `fetchData`, `handleAnimation`, `updateUI`) and any interactivity.\n"
         "    * VERY IMPORTANT: Whenever you want to put images use unplash the api key is defined in a .env file in the same directory so dont worry about it, if your using unsplash in the front end make sure to call the backend for the access token and set up the backend accordingly, For using any sort of llm's in the project we've already set up a gemini api key in the same directory in the .env file ALWAYS USE gemini 2.0 flash model EVEN IF ITS NOT THE STANDARD JUST ALWAYS USE IT, we've also set a yotube api in the same file as well. Use these whenever needed"
        "\nAlso set up the llm with necessary prompts based on the project, ONLY ONLY ONLY USE gemini 2.0 flash AND NOTHING ELSE NO MATTER WHAT JUST USE gemini 2.0 flash WHEN AN LLM FEATURE IS NEEDED\n"
        "\nIf your going to use unsplash images only use 5 images per project.\n"
        f"    *   **Backend Interaction:** Specify the *exact* Flask endpoint(s) and method(s) the frontend will use (e.g., `/api/get_content` GET, `/api/submit_form` POST).\n\n"
        f"2.  **Flask Backend Outline (`app.py`):**\n"
        f"    *   **Routes:** Define all necessary Flask route(s) to support the frontend features (matching the HTML plan).\n"
        f"    *   **Function Descriptions:** Describe what each backend function will do (e.g., 'Fetch detailed Python feature descriptions', 'Process contact form data'). Outline logic needed, including any data processing or interaction if complexity was requested.\n"
        f"    *   **Data Examples:** Specify the structure and *type* of real data the backend should return (e.g., JSON with list of features, success/error messages).\n\n"
        f"3.  **Aim for Completeness and Detail:** The plan should be a blueprint for a *finished* website section, not a skeleton. Think about the final look, feel, and content. Also make it fully functional and ready to test. we've checked and verified that gemini-2.0-flash is the best useable gemini model"
        "At last also provide another section which describes the plan in such a way that even non technical people would understand easily."
    )

def get_nebula_step2_frontend_prompt(user_plan: str, step1_output: str, regeneration_feedback: str | None = None) -> str:
    feedback_section = ""
    if regeneration_feedback:
        feedback_section = (
            f"\n\n**Regeneration Feedback:**\n"
            f"{regeneration_feedback}\n"
            f"**Instructions:** Please incorporate the above feedback when generating the frontend code, considering the original plan and user request.\n"
        )
    return (
        f"**Nebula Code Generation - Step 2: Frontend Development**\n\n"
        f"**User's Initial Request:**\n```\n{user_plan}\n```\n\n"
        f"**Step 1 - Planning Output:**\n```markdown\n{step1_output}\n```\n\n"
        f"{feedback_section}"
        f"**Your Task:** Based on the **Comprehensive Plan** from Step 1, create **ONE single, complete HTML file** (`index.html`) with embedded CSS (in `<style>`) and JS (in `<script>`). Bring the planned vision to life!\n\n"
        f"**Output Requirements:**\n"
        f"1.  **Single File:** Output only one HTML code block containing everything.\n"
        f"2.  **Full Implementation:** Implement all planned HTML structure, detailed CSS styling (including themes, layouts mentioned), and JavaScript logic (animations, interactivity, data fetching).\n"
        f"3.  **Real Content:** Populate the HTML with the *actual content* outlined in the plan. **NO PLACEHOLDERS.** If the plan specified 'Python is versatile...', write that section out fully like do your research and fill out everything properly and accurately.\n"
        f"4.  **Backend Call:** Ensure JavaScript `fetch` calls precisely match the endpoints and methods specified in the Step 1 plan.\n"
        f"5.  **Completeness & Polish:** The file should represent a fully realized frontend for the planned sections. Strive for visual appeal and functional completeness according to the plan. Implement animations and specific styling requested.\n\n"
        f"**IMPORTANT:** Always give the full code without any comments in the code. Build the complete frontend as planned in Step 1. Fill all content sections. Implement visuals and interactions. Ensure backend calls are correct. **No placeholders!**"
    )


def get_nebula_step3_backend_prompt(user_plan: str, step1_output: str, step2_output: str, regeneration_feedback: str | None = None) -> str:
    feedback_section = ""
    if regeneration_feedback:
        feedback_section = (
            f"\n\n**Regeneration Feedback:**\n"
            f"{regeneration_feedback}\n"
            f"**Instructions:** Please incorporate the above feedback when generating the backend code, considering the plan and frontend code.\n"
        )
    return (
        f"**Nebula Code Generation - Step 3: Backend Flask App**\n\n"
        f"**User's Initial Vision:**\n{user_plan}\n\n"
        f"**Step 1 - Comprehensive Plan:**\n{step1_output}\n\n"
        f"**Step 2 - Front end:**\n{step2_output}\n\n"
        f"{feedback_section}"
        f"**Your Task:** Based on the **Comprehensive Plan** from Step 1, create the Python Flask application (`app.py`) needed to fully support the frontend.\n\n"
        f"**Output Requirements:**\n"
        "For using any sort of llm's in the project we've already set up a gemini api key in the same directory in the .env file ALWAYS USE gemini 2.0 flash model EVEN IF ITS NOT THE STANDARD JUST ALWAYS USE IT. Also set up the llm with necessary prompts based on the project"
        f"1.  **Full Setup:** Include necessary imports (`Flask`, `request`, `jsonify`, `send_from_directory`, etc.), and app initialization.\n"
        f"2.  **Serve Frontend:** **Crucially, include the route `@app.route('/')` to serve the `index.html` file** from the application's root directory always use send_from_directory to do this dont use anything else.\n"
        f"3.  **API Routes:** Create all Flask API route(s) *exactly* as defined in the Step 1 plan (endpoints and methods).\n"
        f"4.  **Functional Logic:** Implement Proper databases and full the backend logic described in the plan. If the plan requires processing data, fetching information, or handling complex requests, implement that logic. Return realistic data structures (JSON) as planned – **avoid simple placeholder strings if detailed data was intended.**\n"
        f"5.  **Run Block:** Include the standard `if __name__ == '__main__': app.run(debug=True)` block.\n"
        f"6.  **Dependencies:** List any non-standard Python libraries needed in a comment (e.g., `# requirements: Flask, requests`).\n\n"
        f"**IMPORTANT:**  Always give the full code without any comments in the code. Build a functional backend that fully supports the features and data requirements outlined in Step 1. Ensure the `index.html` serving route is present. Implement the planned logic and return meaningful data."
    )

def get_nebula_step4_verify_prompt(user_plan: str, step1_output: str, step2_frontend_code: str, step3_backend_code: str) -> str:
    return (
        f"**Nebula Code Generation - Step 4: Final Verification**\n\n"
        f"**User's Initial Request:**\n```\n{user_plan}\n```\n\n"
        f"**Step 1 - Planning Output:**\n```markdown\n{step1_output}\n```\n\n"
        f"**Step 2 - Generated Frontend Code (HTML/CSS/JS):**\n```html\n{step2_frontend_code}\n```\n\n"
        f"**Step 3 - Generated Backend Code (Python/Flask):**\n```python\n{step3_backend_code}\n```\n\n"
        f"**Your Task:** Act as QA. Verify Frontend (Step 2) and Backend (Step 3) against the Plan (Step 1) and user request.\n\n"
        f"**Verification Checklist:**\n"
        f"1.  **Plan Adherence:** Does frontend/backend implement components, calls, endpoints, logic from Step 1 plan? Tech stack followed?\n"
        f"2.  **Integration:** Do frontend API calls *exactly* match backend routes (URL, method, format)? Consistent data flow?\n"
        f"3.  **Code Quality:** Structured? Commented? Obvious errors? Dependencies listed? Complete enough for core request?\n"
        f"4.  **Fulfillment:** Does combined code address user request (as per Step 1)?\n"
        f"5.  **Issues/Suggestions:** Inconsistencies, bugs, missing edge cases, improvements?\n\n"
        "Now Your main job:\n If any issue is too major and can impact the quality and the functioning of the code please return back the full code front end or back end with all the issues fixed"
        f"**Output Requirements:** Concise markdown report addressing checklist points. Be specific. Conclude with confidence summary.\n\n"
        f"**Example:**\n"
        f"**Verification Report:**\n\n"
        f"1.  **Plan Adherence:** Frontend/Backend implement planned items. Tech stack OK. [PASS]\n"
        f"2.  **Integration:** API call `/api/data` (POST) matches route. Formats align. [PASS]\n"
        f"3.  **Code Quality:** Generally good. Backend could use more comments. Deps listed. [PASS/PARTIAL]\n"
        f"4.  **Fulfillment:** Core feature covered. [PASS]\n"
        f"5.  **Issues/Suggestions:** Add input validation on backend. Add frontend error handling for API failures.\n\n"
        
        f"**Summary:** High confidence/Low confidence. Minor improvements suggested.\n"

        f"**IMPORTANT:** Be objective. Ensure consistency and function according to the plan."
    )



                  

@app.route('/get_history', methods=['GET'])
def get_history_route():
    try:
        # Access session directly - Flask-Session loads/creates it
        if 'initialized' not in session:
             session['initialized'] = True
             logger.info(f"get_history initialized new session: {session.sid}")
             # No history yet for a brand new session, handle accordingly
             # Maybe insert welcome message here ONLY if session was just created?
             history = get_conversation_history(session.sid) # Fetch (will insert welcome if needed)
             return jsonify({'history': history})
        else:
             logger.info(f"get_history found existing session: {session.sid}")
             session_id = session.sid # Get the ID from the loaded session
             history = get_conversation_history(session_id)
             return jsonify({'history': history})

    except Exception as e:
        logger.error(f"Error in get_history_route: {e}\n{traceback.format_exc()}", exc_info=True)
        return jsonify({'status': 'Failed: Server error fetching history', 'history': []}), 500


@app.route('/update_message', methods=['POST'])
def update_message_route():

    try:
        session_id = get_current_session_id()
        if not session_id:
            return jsonify({'status': 'Failed: No active session'}), 401

        data = request.get_json()
        if not data:
            return jsonify({'status': 'Failed: No JSON data received'}), 400

        message_id = data.get('id')
        content = data.get('content')

        if not message_id:
            return jsonify({'status': 'Failed: Missing message ID parameter'}), 400

        try:
            message_id_int = int(message_id)
        except (ValueError, TypeError):
             return jsonify({'status': 'Failed: Invalid message ID format'}), 400


        db = get_db()

        check = db.execute('SELECT 1 FROM messages WHERE id = ? AND session_id = ?', (message_id_int, session_id)).fetchone()
        if not check:
            logger.warning(f"Update attempt failed: Message {message_id_int} not found or not owned by session {session_id}")
            return jsonify({'status': 'Failed: Message not found or permission denied'}), 403


        success = update_message(message_id_int, content if content is not None else "")
        if success:
            return jsonify({'status': 'Success'})
        else:

             return jsonify({'status': 'Failed: Database update error'}), 500
    except Exception as e:
        logger.error(f"Error in update_message_route: {e}\n{traceback.format_exc()}", exc_info=True)
        return jsonify({'status': 'Failed: Server error during update'}), 500


@app.route('/register_query', methods=['POST'])
def register_query():
    try:
        session_id = get_current_session_id()
        if not session_id:
            return jsonify({'error': 'No active session found. Please refresh.'}), 401

        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data received'}), 400

        query = data.get('query')
        model_id = data.get('model_id')
        mode = data.get('mode')
        pending_files = data.get('pending_files', [])

        if not query or not model_id or not mode:
            return jsonify({'error': 'Missing required data: query, model_id, mode'}), 400

        if not isinstance(pending_files, list):
             logger.warning(f"Received non-list type for pending_files for session {session_id}: {type(pending_files)}. Resetting to empty list.")
             pending_files = []


        query_id = str(uuid.uuid4())


        if 'pending_queries' not in session:
            session['pending_queries'] = {}

        session['pending_queries'][query_id] = {
            'query': query,
            'model_id': model_id,
            'mode': mode,
            'pending_files': pending_files,
            'timestamp': time.time()
        }

        session.modified = True

        logger.info(f"Registered query_id {query_id} for session {session_id} (Mode: {mode}, Files: {len(pending_files)})")
        return jsonify({'query_id': query_id}), 200

    except Exception as e:
        logger.error(f"Error registering query for session {session.get('sid', 'N/A')}: {e}\n{traceback.format_exc()}", exc_info=True)
        return jsonify({'error': 'Internal server error during query registration'}), 500


def run_analysis_for_files(session_id, filenames):

    if not filenames:
        return "", {}

    if not isinstance(filenames, list):
         logger.error(f"run_analysis_for_files expected list, got {type(filenames)}. Aborting.")
         return "[Internal Error: Invalid file list]", {}


    session_upload_folder = os.path.join(app.config['UPLOAD_FOLDER'], session_id)
    threads = []
    local_results = {}
    analysis_start_time = time.time()
    logger.info(f"Starting analysis run for {len(filenames)} file(s) in session {session_id}: {filenames}")


    progress_q = None
    with analysis_progress_lock:
        if session_id not in analysis_progress_queues:
            analysis_progress_queues[session_id] = queue.Queue()
        progress_q = analysis_progress_queues[session_id]


    files_to_analyze = []
    for filename in filenames:
        if not isinstance(filename, str) or not filename:
             logger.warning(f"Skipping invalid filename entry in list for session {session_id}: {filename}")
             continue


        safe_filename = secure_filename(filename)
        filepath = os.path.join(session_upload_folder, safe_filename)

        if os.path.exists(filepath) and os.path.isfile(filepath):

            with analysis_results_lock:
                if session_id in analysis_results_store and safe_filename in analysis_results_store[session_id]:
                    del analysis_results_store[session_id][safe_filename]
                    logger.debug(f"Cleared stale analysis result for {safe_filename} (Session: {session_id}) before starting new analysis.")


            analysis_thread = threading.Thread(target=run_file_analysis, args=(session_id, filepath, safe_filename), daemon=True)
            threads.append({'thread': analysis_thread, 'filename': safe_filename})
            files_to_analyze.append(safe_filename)
            analysis_thread.start()
            logger.info(f"Started analysis thread for '{safe_filename}', session {session_id}")


            start_payload = { "type": "file_start", "session_id": session_id, "filename": safe_filename }
            if progress_q:
                try:
                    progress_q.put(start_payload, block=False)
                except queue.Full:
                    logger.warning(f"SSE queue full for session {session_id} when reporting start for {safe_filename}.")
            else:
                 logger.warning(f"Progress queue not found for session {session_id} when trying to send start message for {safe_filename}")

        else:
            logger.warning(f"File '{safe_filename}' (path: {filepath}) not found or not a file in session folder for analysis. Skipping.")

            local_results[safe_filename] = "[File Not Found During Analysis Trigger]"


    files_to_wait_for = set(files_to_analyze)
    completed_files = set(local_results.keys())
    max_wait_time = 300
    start_wait_time = time.time()

    logger.info(f"Waiting for analysis completion. Files pending: {files_to_wait_for} (Session: {session_id})")

    while files_to_wait_for and (time.time() - start_wait_time) < max_wait_time:
        files_just_completed = set()
        with analysis_results_lock:

            if session_id in analysis_results_store:
                session_results = analysis_results_store[session_id]

                for filename in list(files_to_wait_for):
                    if filename in session_results:

                        result_text = session_results.get(filename, "[Analysis Result Missing Error]")
                        local_results[filename] = result_text
                        files_just_completed.add(filename)
                        logger.info(f"Analysis result received for {filename} (Session: {session_id}). Result length: {len(result_text)} chars.")


        if files_just_completed:
             files_to_wait_for -= files_just_completed
             logger.info(f"Files remaining to analyze: {len(files_to_wait_for)} (Session: {session_id})")

        if not files_to_wait_for:
            break

        time.sleep(0.5)


    if files_to_wait_for:
        logger.warning(f"Analysis timed out for files: {files_to_wait_for} (Session: {session_id}) after {max_wait_time}s.")
        timeout_message = f"[Analysis Timed Out after {max_wait_time}s]"
        for filename in files_to_wait_for:
            if filename not in local_results:
                 local_results[filename] = timeout_message

                 timeout_payload = { "type": "file_error", "session_id": session_id, "filename": filename, "error": "Analysis timed out" }
                 if progress_q:
                     try:
                         progress_q.put(timeout_payload, block=False)
                     except queue.Full:
                         logger.warning(f"SSE queue full for session {session_id} when reporting timeout for {filename}.")
                 else:
                     logger.warning(f"Progress queue not found for session {session_id} when trying to send timeout message for {filename}")


    total_time = time.time() - analysis_start_time
    logger.info(f"Analysis run finished for session {session_id}. Took {total_time:.2f}s. Results gathered for {len(local_results)}/{len(filenames)} files.")


    file_context_to_inject = ""
    if local_results:

        file_context_to_inject += "**Analysis Results from Uploaded Files:**\n"
        for filename, analysis_text in local_results.items():

            file_context_to_inject += (
                f"\n<details>\n"
                f"  <summary>📄 Analysis Summary: {filename}</summary>\n\n"
                f"  **File:** `{filename}`\n\n"
                f"  **Analysis:**\n"
                f"  ```text\n"
                f"{analysis_text}\n"
                f"  ```\n\n"
                f"</details>\n"
            )
        file_context_to_inject += "\n---\n"


    with analysis_results_lock:
        if session_id in analysis_results_store:
            session_store = analysis_results_store[session_id]
            cleared_count = 0
            for filename in local_results.keys():
                 if filename in session_store:
                     session_store.pop(filename, None)
                     cleared_count += 1
            logger.info(f"Cleared {cleared_count} used analysis results from shared store for session {session_id}")

            if not session_store:
                 del analysis_results_store[session_id]
                 logger.info(f"Removed empty session entry from analysis results store: {session_id}")


    return file_context_to_inject, local_results


@app.route('/refine_stream', methods=['GET'])
def refine_stream():
    start_time = time.time()
    query_id = request.args.get('query_id')

    session_id = get_current_session_id()
    if not session_id:
        logger.error("Refine stream failed: No session ID.")
        def error_stream(): yield f"data: {json.dumps({'status': 'Session error. Please refresh.', 'error': True})}\n\n"
        return Response(stream_with_context(error_stream()), mimetype='text/event-stream', status=500)

    if not query_id:
        logger.error("Refine stream failed: Missing query_id parameter.")
        def error_stream(): yield f"data: {json.dumps({'status': 'Error: Missing query identifier.', 'error': True})}\n\n"
        return Response(stream_with_context(error_stream()), mimetype='text/event-stream', status=400)


    query_data = None
    if 'pending_queries' in session and query_id in session['pending_queries']:
        pending_queries = session['pending_queries']
        query_data = pending_queries.pop(query_id)

        session.modified = True


        if not pending_queries:
            session.pop('pending_queries', None)
            logger.info(f"Removed empty 'pending_queries' from session {session_id}")

    if not query_data:
        logger.error(f"Refine stream failed: query_id {query_id} not found in session {session_id} or already processed.")
        def error_stream(): yield f"data: {json.dumps({'status': 'Error: Query session expired or invalid.', 'error': True})}\n\n"

        return Response(stream_with_context(error_stream()), mimetype='text/event-stream', status=404)



    user_query_from_frontend = query_data.get('query', '')
    model_id = query_data.get('model_id')
    pending_files = query_data.get('pending_files', [])
    mode = query_data.get('mode')


    if not user_query_from_frontend or not model_id:
        logger.error(f"Refine stream error: Missing query or model_id in query data for query_id {query_id}, session {session_id}.")
        def error_stream(): yield f"data: {json.dumps({'status': 'Error: Invalid query data retrieved.', 'error': True})}\n\n"
        return Response(stream_with_context(error_stream()), mimetype='text/event-stream', status=500)



    fallback_model = "gemini-2.0-flash-thinking-exp-01-21"
    max_model_attempts = 2


    user_message_id = insert_message(session_id, "user", user_query_from_frontend)
    if not user_message_id:
         logger.error(f"Failed to insert user message for refine query {query_id} in session {session_id}. Proceeding anyway.")


    logger.info(f"Refine stream started for query_id {query_id}, session {session_id}. Mode: {mode}. Files to analyze: {len(pending_files)}")


    def generate_refinement_stream_with_analysis():
        file_analysis_context = ""
        analysis_results_dict = {}
        final_stellar_message_id = None
        llm_error_occurred = False

        try:

            if pending_files:
                 yield f"data: {json.dumps({'status': f'Analyzing {len(pending_files)} file(s)...', 'phase': 'analysis'})}\n\n"
                 logger.info(f"Refine Stream (QueryID: {query_id}): Triggering analysis for {len(pending_files)} files: {pending_files}")


                 file_analysis_context, analysis_results_dict = run_analysis_for_files(session_id, pending_files)


                 if analysis_results_dict:

                     yield f"data: {json.dumps({'status': 'File analysis complete. Refining query...', 'phase': 'refining', 'analysis_results': analysis_results_dict })}\n\n"
                     logger.info(f"Refine Stream (QueryID: {query_id}): Analysis complete. Proceeding to refine query.")
                 else:

                     yield f"data: {json.dumps({'status': 'File analysis finished (no results?). Refining query...', 'phase': 'refining'})}\n\n"
                     logger.warning(f"Refine Stream (QueryID: {query_id}): Analysis run yielded empty results dict for session {session_id}. Proceeding without file context.")
            else:

                 yield f"data: {json.dumps({'status': 'No files to analyze. Refining query...', 'phase': 'refining'})}\n\n"
                 logger.info(f"Refine Stream (QueryID: {query_id}): No pending files. Proceeding directly to refine query.")



            user_query_for_llm = file_analysis_context + user_query_from_frontend

            user_query_for_llm += f"\n\n(Responding using Stellar model: {MODEL_NAMES.get(model_id, model_id)})"


            conversation_history = get_conversation_history(session_id)
            conv_hist_list = []
            if conversation_history:
                for msg in conversation_history:

                    if str(msg.get('id')) == str(user_message_id):
                        continue

                    role = 'User' if msg.get('message_type') == 'user' else 'Stellar'
                    content = msg.get('message_content', '')

                    conv_hist_list.append(f"{role}: {content}")


            refined_query_result = None
            selected_model = model_id

            for model_attempt in range(max_model_attempts):
                current_model = selected_model
                display_name = MODEL_NAMES.get(current_model, current_model)
                current_api_key = REFINE_API_KEY


                if not current_api_key:
                    logger.error(f"Refine API key is missing! Cannot proceed with model {display_name}.")
                    yield f"data: {json.dumps({'status': 'Error: API Key Configuration Missing.', 'error': True})}\n\n"
                    llm_error_occurred = True
                    return


                if model_attempt > 0:
                    yield f"data: {json.dumps({'status': f'Initial model failed. Falling back to {display_name}...', 'phase': 'refining'})}\n\n"
                    time.sleep(1)

                logger.info(f"Refine Stream (QueryID: {query_id}): Calling {display_name} (Attempt {model_attempt+1}/{max_model_attempts})...")
                yield f"data: {json.dumps({'status': f'Thinking with {display_name}...', 'phase': 'refining'})}\n\n"


                prompt = get_refinement_prompt(user_query_for_llm, conv_hist_list)


                generator_output = gemini_generate(
                    prompt=prompt,
                    model_id=current_model,
                    key=current_api_key,
                    attempts=1,
                    model_display_name=f"{display_name}"
                )


                temp_result = None
                for item in generator_output:
                    if 'status' in item:

                        yield f"data: {json.dumps({'status': item['status'], 'phase': 'refining'})}\n\n"
                    elif 'result' in item:
                        temp_result = item['result']

                        if isinstance(temp_result, str) and temp_result.startswith(ERROR_CODE):
                            logger.error(f"Error from {display_name} during refinement (Attempt {model_attempt+1}): {temp_result}")
                            temp_result = None
                        else:

                            refined_query_result = temp_result
                        break


                if refined_query_result is not None:
                    logger.info(f"Refine Stream (QueryID: {query_id}): Successfully refined query using {display_name}.")
                    break
                else:

                    if model_attempt == 0 and fallback_model and fallback_model != model_id:
                        selected_model = fallback_model
                        logger.warning(f"Refine Stream (QueryID: {query_id}): Model {display_name} failed. Will attempt fallback with {fallback_model}.")
                    else:

                         logger.error(f"Refine Stream (QueryID: {query_id}): Refinement failed after {model_attempt+1} attempts (Model: {display_name}).")



            if refined_query_result is not None:


                stellar_message_id = insert_message(
                    session_id,
                    "stellar",
                    refined_query_result,
                    file_analysis_context=file_analysis_context
                )

                if stellar_message_id:
                     final_stellar_message_id = stellar_message_id

                     final_data = {
                         'status': 'refined_ready',
                         'session_id': session_id,
                         'message_id': str(final_stellar_message_id),
                         'user_message_id': str(user_message_id) if user_message_id else None,
                         'refined_query': refined_query_result,
                         'analysis_context_used': file_analysis_context,
                         'analysis_results': analysis_results_dict
                     }
                     yield f"data: {json.dumps(final_data)}\n\n"
                     logger.info(f"Refinement complete for query_id {query_id}. Sent final response. Took {time.time() - start_time:.2f}s")
                else:

                      error_msg = "Refinement generated but failed to save AI response to database."
                      logger.error(error_msg + f" (Session: {session_id}, QueryID: {query_id})")
                      yield f"data: {json.dumps({'status': error_msg, 'error': True})}\n\n"
                      llm_error_occurred = True

            else:

                 error_msg = "Encountered an error: Unable to refine query after all attempts."
                 logger.error(error_msg + f" (Session: {session_id}, QueryID: {query_id})")
                 yield f"data: {json.dumps({'status': error_msg, 'error': True})}\n\n"
                 llm_error_occurred = True

        except Exception as e:

            logger.error(f"Unexpected error IN generate_refinement_stream for session {session_id}, query {query_id}: {e}\n{traceback.format_exc()}", exc_info=True)
            yield f"data: {json.dumps({'status': 'Severe error during refinement stream processing.', 'error': True})}\n\n"
            llm_error_occurred = True
        finally:

             logger.info(f"Refine stream generator finished for query_id {query_id}, session {session_id}.")


    return Response(stream_with_context(generate_refinement_stream_with_analysis()), mimetype='text/event-stream')


@app.route('/search_stream', methods=['GET'])
def search_stream():
    start_time = time.time()
    query_id = request.args.get('query_id')


    session_id = get_current_session_id()
    if not session_id:
        logger.error("Search stream failed: No session ID.")
        def error_stream(): yield f"data: {json.dumps({'status': 'Session error. Please refresh.', 'error': True})}\n\n"
        return Response(stream_with_context(error_stream()), mimetype='text/event-stream', status=500)

    if not query_id:
        logger.error("Search stream failed: Missing query_id parameter.")
        def error_stream(): yield f"data: {json.dumps({'status': 'Error: Missing query identifier.', 'error': True})}\n\n"
        return Response(stream_with_context(error_stream()), mimetype='text/event-stream', status=400)


    query_data = None
    if 'pending_queries' in session and query_id in session['pending_queries']:
        pending_queries = session['pending_queries']
        query_data = pending_queries.pop(query_id)
        session.modified = True
        if not pending_queries:
            session.pop('pending_queries', None)
            logger.info(f"Removed empty 'pending_queries' from session {session_id}")
    else:
        logger.error(f"Search stream failed: query_id {query_id} not found in session {session_id} or already processed.")
        def error_stream(): yield f"data: {json.dumps({'status': 'Error: Query session expired or invalid.', 'error': True})}\n\n"
        return Response(stream_with_context(error_stream()), mimetype='text/event-stream', status=404)


    user_query = query_data.get('query', '')
    model_id = query_data.get('model_id')
    mode = query_data.get('mode')
    pending_files = query_data.get('pending_files', [])


    use_tavily = (mode == 'search_tavily')

    if not user_query or not model_id:
        logger.error(f"Search stream error: Missing query or model_id in query data for query_id {query_id}, session {session_id}.")
        def error_stream(): yield f"data: {json.dumps({'status': 'Error: Invalid query data retrieved.', 'error': True})}\n\n"
        return Response(stream_with_context(error_stream()), mimetype='text/event-stream', status=500)


    fallback_model = "gemini-2.0-flash-thinking-exp-01-21"
    max_model_attempts = 2


    user_message_id = insert_message(session_id, "user", user_query)
    if not user_message_id:
        logger.error(f"Failed to insert user message for search query {query_id} in session {session_id}. Proceeding anyway.")

    logger.info(f"Search stream started. QueryID: {query_id}, Session: {session_id}, Query: '{user_query[:50]}...', Tavily: {use_tavily}, Model: {model_id}, Files: {len(pending_files)}")


    def generate_research_stream_with_id():
        full_context = ""
        web_search_context = ""
        file_analysis_context = ""
        analysis_results_dict = {}
        research_analysis_result = None
        final_result = None
        html_filepath_rel = None
        research_message_id = None
        error_occurred = False

        try:

            if pending_files:
                 yield f"data: {json.dumps({'status': f'Analyzing {len(pending_files)} file(s)...', 'phase': 'analysis'})}\n\n"
                 logger.info(f"Search Stream (QueryID: {query_id}): Triggering analysis for {len(pending_files)} files: {pending_files}")
                 file_analysis_context, analysis_results_dict = run_analysis_for_files(session_id, pending_files)
                 yield f"data: {json.dumps({'status': 'File analysis complete.', 'phase': 'context_gathering', 'analysis_results': analysis_results_dict })}\n\n"
                 logger.info(f"Search Stream (QueryID: {query_id}): File analysis complete.")
            else:
                logger.info(f"Search Stream (QueryID: {query_id}): No pending files for analysis.")



            conversation_history = get_conversation_history(session_id)
            conv_hist_list = []
            if conversation_history:
                for msg in conversation_history:
                    if str(msg.get('id')) == str(user_message_id): continue
                    role = 'User' if msg.get('message_type') == 'user' else 'Stellar'
                    content = msg.get('message_content', '')
                    conv_hist_list.append(f"{role}: {content}")
            conv_hist_str = "\n".join(conv_hist_list) if conv_hist_list else "No previous conversation."
            history_context = f"**Conversation History:**\n{conv_hist_str}\n\n---\n"



            if use_tavily:
                yield f"data: {json.dumps({'status': 'Performing Spectral Search...', 'phase': 'context_gathering'})}\n\n"
                logger.info(f"Search Stream (QueryID: {query_id}): Starting Spectral Search (Tavily)...")
                tavily_success = False
                for attempt in range(2):
                    try:
                        status_msg = 'Performing Spectral Search...' if attempt == 0 else f'Retrying Spectral Search... (Attempt {attempt + 1})'
                        yield f"data: {json.dumps({'status': status_msg, 'phase': 'context_gathering'})}\n\n"

                        tavily_response = tavily_search(user_query)
                        if isinstance(tavily_response, dict) and "error" in tavily_response:
                            raise ValueError(f"Tavily API Error: {tavily_response['error']}")
                        if not isinstance(tavily_response, dict) or "results" not in tavily_response:
                             raise TypeError(f"Tavily returned unexpected/invalid response format: {type(tavily_response)}")

                        tavily_answer = tavily_response.get("answer", "")
                        results = tavily_response.get("results", [])
                        logger.info(f"Search Stream (QueryID: {query_id}): Tavily search yielded {len(results)} results.")


                        current_web_context = f"**Spectral Search Summary:**\n{tavily_answer if tavily_answer else 'No summary provided.'}\n\n**Scraped Content Details:**\n"

                        scraped_contents = []
                        urls_to_scrape = [r.get("url") for r in results if r.get("url")]
                        urls_scraped_count = 0

                        for url in urls_to_scrape:
                            if not url or not isinstance(url, str) or not (url.startswith('http://') or url.startswith('https://')):
                                logger.warning(f"Skipping invalid URL from Tavily: {url}")
                                continue

                            yield f"data: {json.dumps({'status': f'Scraping {url}...', 'phase': 'context_gathering'})}\n\n"
                            logger.info(f"Search Stream (QueryID: {query_id}): Scraping {url}...")
                            content = scrape_url(url)
                            if content and isinstance(content, str) and not content.startswith("Error scraping"):


                                scraped_contents.append(f"<details><summary>Content from: {url}</summary>\n\n```text\n{content}\n```\n\n</details>\n")
                                urls_scraped_count += 1
                            elif content and content.startswith("Error scraping"):
                                logger.warning(f"Scraping failed for {url}: {content}")
                                scraped_contents.append(f"*   Content from {url}: [Scraping Error: {content}]*\n")
                            else:
                                logger.warning(f"Empty content returned from scraping {url}")
                                scraped_contents.append(f"*   Content from {url}: [No Content Scraped]*\n")

                        current_web_context += "\n".join(scraped_contents) if scraped_contents else "No content could be scraped from search results.\n"
                        current_web_context += "\n---\n"
                        web_search_context = current_web_context
                        tavily_success = True
                        yield f"data: {json.dumps({'status': f'Spectral Search completed ({urls_scraped_count} sources scraped).', 'phase': 'context_gathering'})}\n\n"
                        logger.info(f"Search Stream (QueryID: {query_id}): Spectral Search and scraping completed.")
                        break

                    except Exception as e:
                        logger.error(f"Search Stream (QueryID: {query_id}): Tavily/Scraping error (Attempt {attempt+1}): {e}", exc_info=True)
                        if attempt < 1:
                             yield f"data: {json.dumps({'status': f'Spectral Search failed (Attempt {attempt+1}). Retrying...', 'error': True, 'phase': 'context_gathering'})}\n\n"
                             time.sleep(1.5)
                        else:
                             yield f"data: {json.dumps({'status': 'Spectral Search failed after retries. Proceeding without web context.', 'error': True, 'phase': 'context_gathering'})}\n\n"
                             logger.warning(f"Search Stream (QueryID: {query_id}): Tavily search failed. Proceeding without web context.")
                             web_search_context = "**Spectral Search Attempted:** Failed after retries.\n\n---\n"
                             break
            else:

                 yield f"data: {json.dumps({'status': 'Proceeding without Spectral Search (disabled)...', 'phase': 'context_gathering'})}\n\n"
                 logger.info(f"Search Stream (QueryID: {query_id}): Proceeding without Spectral Search (disabled by mode).")
                 web_search_context = "**Spectral Search Attempted:** Skipped by user/mode.\n\n---\n"



            full_context =file_analysis_context + web_search_context



            yield f"data: {json.dumps({'status': 'Starting research analysis...', 'phase': 'analysis_llm'})}\n\n"
            logger.info(f"Search Stream (QueryID: {query_id}): Starting research analysis LLM call.")
            selected_analysis_model = model_id

            for model_attempt in range(max_model_attempts):
                current_model = selected_analysis_model
                display_name = MODEL_NAMES.get(current_model, current_model)
                current_api_key = SEARCH_API_KEY

                if not current_api_key:
                    logger.error("API Key SEARCH_API_KEY is not set.")
                    yield f"data: {json.dumps({'status': 'Error: API Key for Search Analysis is missing.', 'error': True, 'phase': 'analysis_llm'})}\n\n"
                    error_occurred = True
                    return

                if model_attempt > 0:
                     fallback_status = f'Analysis model failed. Falling back to {display_name}...'
                     logger.warning(fallback_status)
                     yield f"data: {json.dumps({'status': fallback_status, 'phase': 'analysis_llm'})}\n\n"
                     time.sleep(1)

                logger.info(f"Search Stream (QueryID: {query_id}): Research Analysis Attempt {model_attempt+1}/{max_model_attempts} using {display_name}")
                yield f"data: {json.dumps({'status': f'Analyzing context with {display_name}...', 'phase': 'analysis_llm'})}\n\n"

                research_prompt = get_research_analysis_prompt(user_query, full_context)

                generator_output_analysis = gemini_generate(
                    prompt=research_prompt, model_id=current_model, key=current_api_key,
                    attempts=1,
                    model_display_name=f"{display_name} (Analysis)"
                )
                temp_result_analysis = None
                for item in generator_output_analysis:
                    if 'status' in item:
                        yield f"data: {json.dumps({'status': item['status'], 'phase': 'analysis_llm'})}\n\n"
                    elif 'result' in item:
                        temp_result_analysis = item['result']
                        if isinstance(temp_result_analysis, str) and temp_result_analysis.startswith(ERROR_CODE):
                            logger.error(f"Error from {display_name} during analysis: {temp_result_analysis}")
                            temp_result_analysis = None
                        else:
                            research_analysis_result = temp_result_analysis
                        break

                if research_analysis_result is not None:
                     logger.info(f"Search Stream (QueryID: {query_id}): Research analysis successful with {display_name}.")
                     break
                else:
                     if model_attempt == 0 and fallback_model and fallback_model != model_id:
                         selected_analysis_model = fallback_model
                         logger.warning(f"Search Stream (QueryID: {query_id}): Analysis model {display_name} failed. Will attempt fallback with {fallback_model}.")
                     else:
                         logger.error(f"Search Stream (QueryID: {query_id}): Analysis failed after {model_attempt+1} attempts (Model: {display_name}).")


            if not research_analysis_result:
                error_msg = f"Research analysis failed after all attempts for query_id {query_id}."
                logger.error(error_msg + f" Session: {session_id}")
                yield f"data: {json.dumps({'status': error_msg, 'error': True, 'phase': 'analysis_llm'})}\n\n"
                error_occurred = True
                return



            yield f"data: {json.dumps({'status': 'Expanding analysis into full research paper...', 'phase': 'expansion_llm'})}\n\n"
            logger.info(f"Search Stream (QueryID: {query_id}): Expanding analysis into full paper...")
            selected_expansion_model = model_id

            for model_attempt in range(max_model_attempts):
                current_model = selected_expansion_model
                display_name = MODEL_NAMES.get(current_model, current_model)

                current_api_key = SEARCH_API_KEY

                if not current_api_key:
                    logger.error("API Key SEARCH_API_KEY is not set (for expansion).")
                    yield f"data: {json.dumps({'status': 'Error: API Key for Search Expansion is missing.', 'error': True, 'phase': 'expansion_llm'})}\n\n"
                    error_occurred = True
                    return

                if model_attempt > 0:
                    fallback_status = f'Expansion model failed. Falling back to {display_name}...'
                    logger.warning(fallback_status)
                    yield f"data: {json.dumps({'status': fallback_status, 'phase': 'expansion_llm'})}\n\n"
                    time.sleep(1)

                logger.info(f"Search Stream (QueryID: {query_id}): Expansion Attempt {model_attempt+1}/{max_model_attempts} using {display_name}")
                yield f"data: {json.dumps({'status': f'{display_name} is finalizing the paper...', 'phase': 'expansion_llm'})}\n\n"

                final_prompt = get_final_expansion_prompt(user_query, research_analysis_result, full_context)

                generator_output_expansion = gemini_generate(
                    prompt=final_prompt, model_id=current_model, key=current_api_key,
                    attempts=1,
                    model_display_name=f"{display_name} (Expansion)"
                )
                temp_result_expansion = None
                for item in generator_output_expansion:
                    if 'status' in item:
                         yield f"data: {json.dumps({'status': item['status'], 'phase': 'expansion_llm'})}\n\n"
                    elif 'result' in item:
                        temp_result_expansion = item['result']
                        if isinstance(temp_result_expansion, str) and temp_result_expansion.startswith(ERROR_CODE):
                            logger.error(f"Error from {display_name} during expansion: {temp_result_expansion}")
                            temp_result_expansion = None
                        else:
                            final_result = temp_result_expansion
                        break

                if final_result is not None:
                    logger.info(f"Search Stream (QueryID: {query_id}): Paper expansion successful with {display_name}.")
                    break
                else:
                    if model_attempt == 0 and fallback_model and fallback_model != model_id:
                        selected_expansion_model = fallback_model
                        logger.warning(f"Search Stream (QueryID: {query_id}): Expansion model {display_name} failed. Will attempt fallback with {fallback_model}.")
                    else:
                        logger.error(f"Search Stream (QueryID: {query_id}): Expansion failed after {model_attempt+1} attempts (Model: {display_name}).")


            if not final_result:
                error_msg = f"Failed to generate the final research paper after all attempts for query_id {query_id}."
                logger.error(error_msg + f" Session: {session_id}")
                yield f"data: {json.dumps({'status': error_msg, 'error': True, 'phase': 'expansion_llm'})}\n\n"
                error_occurred = True
                return


            yield f"data: {json.dumps({'status': 'Formatting paper (HTML)...', 'phase': 'formatting'})}\n\n"
            logger.info(f"Search Stream (QueryID: {query_id}): Formatting and saving the paper...")

            html_content_for_db = None
            try:

                html_filepath_rel = create_output_file(user_query, final_result, extension="md")

                if html_filepath_rel:
                     logger.info(f"Saved raw Markdown to: {html_filepath_rel}")
                     html_output_path = html_filepath_rel.replace(".md", ".html")
                     try:

                         pypandoc.convert_file(
                             source_file=html_filepath_rel,
                             to='html5',
                             format='markdown_strict+pipe_tables+implicit_figures+footnotes-native_divs-native_spans',
                             outputfile=html_output_path,
                             extra_args=['--standalone', '--toc', '--mathjax', '--css=default.min.css', '--highlight-style=pygments', '--wrap=none', '--columns=1000'],
                             encoding='utf-8'
                         )

                         html_filepath_rel = html_output_path
                         logger.info(f"HTML conversion successful: {html_filepath_rel}")


                     except Exception as pandoc_e:
                         logger.error(f"Pandoc HTML conversion error from {html_filepath_rel}: {pandoc_e}", exc_info=True)
                         yield f"data: {json.dumps({'status': 'Warning: Failed to convert paper to HTML. Providing Markdown link.', 'error': False, 'phase': 'formatting'})}\n\n"

                         html_filepath_rel = html_filepath_rel.replace(".html", ".md")
                else:
                     logger.error(f"Search Stream (QueryID: {query_id}): Failed to save raw Markdown output file.")
                     yield f"data: {json.dumps({'status': 'Error: Failed to save output file.', 'error': True, 'phase': 'formatting'})}\n\n"


            except Exception as e:
                logger.error(f"Search Stream (QueryID: {query_id}): Error during output file saving/formatting: {e}", exc_info=True)
                yield f"data: {json.dumps({'status': 'Error during file saving/formatting.', 'error': True, 'phase': 'formatting'})}\n\n"
                html_filepath_rel = None



            research_message_id = insert_message(
                session_id=session_id,
                message_type="stellar",
                message_content=final_result,
                is_research_output=True,
                html_file=html_filepath_rel,
                file_analysis_context=file_analysis_context + web_search_context
            )

            if not research_message_id:
                logger.error(f"Search Stream (QueryID: {query_id}): Failed to save research paper result to database!")
                yield f"data: {json.dumps({'status': 'Error: Failed to save result to database.', 'error': True, 'phase': 'saving'})}\n\n"


                error_occurred = True

            else:
                 logger.info(f"Search Stream (QueryID: {query_id}): Research paper saved to DB. Message ID: {research_message_id}")


                 final_data = {
                     'status': 'display_result',
                     'session_id': session_id,
                     'message_id': str(research_message_id),
                     'user_message_id': str(user_message_id) if user_message_id else None,
                     'result': final_result,

                     'file_url': f'/view/{os.path.basename(html_filepath_rel)}' if html_filepath_rel else None,
                     'download_url': f'/download/{os.path.basename(html_filepath_rel)}' if html_filepath_rel else None,
                     'file_type': os.path.splitext(html_filepath_rel)[1].lower() if html_filepath_rel else None
                 }
                 yield f"data: {json.dumps(final_data)}\n\n"
                 logger.info(f"Search Stream (QueryID: {query_id}): Research paper generation complete. Took {time.time() - start_time:.2f}s")

        except Exception as e:

            logger.error(f"Unexpected error IN generate_research_stream_with_id (QueryID: {query_id}): {e}\n{traceback.format_exc()}", exc_info=True)
            try:
                yield f"data: {json.dumps({'status': 'Severe error during research generation.', 'error': True})}\n\n"
            except Exception as yield_err:
                 logger.error(f"Failed to yield final error message for QueryID {query_id}: {yield_err}")
            error_occurred = True
        finally:
            logger.info(f"Search stream generator finished for query_id {query_id}, session {session_id}.")



    return Response(stream_with_context(generate_research_stream_with_id()), mimetype='text/event-stream')


@app.route('/nebula/step', methods=['POST'])
def nebula_step():
    start_time = time.time()
    try:
        session_id = get_current_session_id()
        if not session_id:
             return jsonify({'error': 'No active session found'}), 401

        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data received'}), 400


        process_id = data.get('processId')
        step = data.get('step')
        model_id = data.get('model_id')
        context = data.get('context', {})
        regenerate = data.get('regenerate', False)
        regeneration_feedback = context.get('regeneration_feedback') if regenerate else None

        if not process_id or step is None or not model_id:
            return jsonify({'error': 'Missing required parameters: processId, step, model_id'}), 400

        try:
            step = int(step)
            if step < 1 or step > 4: raise ValueError("Step must be between 1 and 4")
        except (ValueError, TypeError):
             return jsonify({'error': 'Invalid step number provided'}), 400

        if model_id not in NEBULA_COMPATIBLE_MODELS:

             logger.warning(f"Nebula process {process_id} requested with model {model_id}, which might not be optimal. Recommended: {NEBULA_COMPATIBLE_MODELS}.")



        if 'nebula_processes' not in session:
            session['nebula_processes'] = {}
            session.modified = True

        process_id_str = str(process_id)
        process_state = session['nebula_processes'].get(process_id_str)


        if process_state is None and step == 1 and not regenerate:
             logger.info(f"Initializing Nebula process {process_id_str} for session {session_id}, step 1.")
             user_query = context.get('query', '').strip()
             if not user_query:
                 return jsonify({'error': 'Missing user query in context for step 1'}), 400
             process_state = {
                 'query': user_query,
                 'model_id': model_id,
                 'outputs': {},
                 'current_step': 1
             }
             session['nebula_processes'][process_id_str] = process_state
             session.modified = True
        elif process_state is None:

             logger.error(f"Received request for step {step} (Regen: {regenerate}) for non-existent Nebula process {process_id_str}")
             return jsonify({'error': f'Nebula process {process_id_str} not found or expired. Please start a new Nebula process.'}), 404



        step_key = f'step{step}'


        if regenerate:
            if step_key in process_state.get('outputs', {}):
                 logger.info(f"Regenerating Nebula step {step} for process {process_id_str}. Clearing previous output for '{step_key}'.")
                 process_state.get('outputs', {}).pop(step_key, None)

                 if step < 4 and 'step'+str(step+1) in process_state.get('outputs',{}): process_state.get('outputs', {}).pop('step'+str(step+1), None)
                 if step < 3 and 'step'+str(step+2) in process_state.get('outputs',{}): process_state.get('outputs', {}).pop('step'+str(step+2), None)
                 if step < 2 and 'step'+str(step+3) in process_state.get('outputs',{}): process_state.get('outputs', {}).pop('step'+str(step+3), None)
                 logger.info(f"Cleared subsequent step outputs after regenerating step {step}.")

            process_state['current_step'] = step
            session.modified = True


        prompt = ""
        prompt_func = None
        prompt_args = []
        api_key_name = f'step{step}'
        required_context_keys = []
        outputs_dict = process_state.get('outputs', {})


        web_context_for_step1 = None
        if step == 1:
            user_query = process_state['query']
            regeneration_feedback_for_step1 = context.get('regeneration_feedback')


            if not regenerate:
                 logger.info(f"Nebula Step 1 (Process: {process_id_str}): Classifying query for real-time needs: '{user_query[:50]}...'")
                 real_time_needed = classify_real_time_needed(user_query, RTP_API_KEY)
                 logger.info(f"Nebula Step 1 (Process: {process_id_str}): Real-time classification result: {real_time_needed}")

                 if real_time_needed == "yes":
                     logger.info(f"Nebula Step 1 (Process: {process_id_str}): Real-time context needed. Starting Tavily search and scrape.")


                     try:

                         instruction_prompt = user_query + """\nAnalyze the user request to identify core entities, desired actions/info, and constraints. Generate concise instructions for another AI on how to formulate an effective Tavily search query."""
                         instruction_gen = gemini_generate(prompt=instruction_prompt, model_id="gemini-2.0-flash-lite", key=RTP_API_KEY, attempts=1)
                         instruction = next((item['result'] for item in instruction_gen if 'result' in item), None)

                         search_query_for_tavily = user_query
                         if instruction and not instruction.startswith(ERROR_CODE):
                             query_gen_prompt = instruction + f"\nBased on the instruction, create a specific Tavily search query for:\nOriginal Query: {user_query}\nReturn only the search query string."
                             query_gen = gemini_generate(prompt=query_gen_prompt, model_id="gemini-2.0-flash-lite", key=RTP_API_KEY, attempts=1)
                             generated_query = next((item['result'] for item in query_gen if 'result' in item), None)
                             if generated_query and not generated_query.startswith(ERROR_CODE):
                                 search_query_for_tavily = generated_query.strip().strip('"')
                                 logger.info(f"Generated specialized Tavily query: {search_query_for_tavily}")


                         tavily_response = tavily_search(search_query_for_tavily, max_results=5)
                         if isinstance(tavily_response, dict) and "error" in tavily_response:
                             raise ValueError(f"Tavily Error: {tavily_response['error']}")
                         if not isinstance(tavily_response, dict):
                             raise TypeError(f"Tavily returned unexpected type: {type(tavily_response)}")

                         tavily_answer = tavily_response.get("answer", "")
                         results = tavily_response.get("results", [])
                         logger.info(f"Nebula Step 1 (Process: {process_id_str}): Tavily yielded {len(results)} results.")


                         scraped_contents = []
                         urls_to_scrape = [r.get("url") for r in results if r.get("url")]
                         logger.info(f"Nebula Step 1 (Process: {process_id_str}): Attempting to scrape up to {len(urls_to_scrape)} URLs.")
                         max_urls_scrape_step1 = 3
                         for url in urls_to_scrape[:max_urls_scrape_step1]:

                             if not url or not url.startswith(('http://', 'https://')): continue
                             logger.info(f"Scraping {url} for Nebula plan...")
                             content = scrape_url(url)
                             if content and isinstance(content, str) and not content.startswith("Error scraping"):
                                  scraped_contents.append(f"Content from {url}:\n{content}\n")
                             elif content: logger.warning(f"Scraping failed for {url}: {content}")
                             else: logger.warning(f"Empty content from scraping {url}")

                         combined_scraped_context = "\n---\n".join(scraped_contents) if scraped_contents else "No additional content could be scraped."
                         web_context_for_step1 = f"**Web Search Summary:**\n{tavily_answer if tavily_answer else '(Not provided)'}\n\n**Scraped Context:**\n{combined_scraped_context}"
                         logger.info(f"Nebula Step 1 (Process: {process_id_str}): Web context gathering complete.")

                     except Exception as e:
                         logger.error(f"Nebula Step 1 (Process: {process_id_str}): Web context gathering failed: {e}. Proceeding without web context.", exc_info=True)
                         web_context_for_step1 = "[Web Context Fetching Failed]"
                 else:
                     logger.info(f"Nebula Step 1 (Process: {process_id_str}): Real-time context not needed, skipping web search.")
                     web_context_for_step1 = None


            prompt_func = get_nebula_step1_plan_prompt

            prompt_args = [user_query, regeneration_feedback_for_step1, web_context_for_step1]

        elif step == 2:
            required_context_keys = ['step1']
            prompt_func = get_nebula_step2_frontend_prompt
            prompt_args = [process_state['query'], outputs_dict.get('step1'), regeneration_feedback]
        elif step == 3:
            required_context_keys = ['step1', 'step2']
            prompt_func = get_nebula_step3_backend_prompt
            prompt_args = [process_state['query'], outputs_dict.get('step1'), outputs_dict.get('step2'), regeneration_feedback]
        elif step == 4:
            required_context_keys = ['step1', 'step2', 'step3']
            prompt_func = get_nebula_step4_verify_prompt
            prompt_args = [ process_state['query'], outputs_dict.get('step1'), outputs_dict.get('step2'), outputs_dict.get('step3') ]


        for idx, req_step_key in enumerate(required_context_keys):

             if req_step_key not in outputs_dict or not outputs_dict[req_step_key]:
                 error_msg = f"Cannot proceed to Nebula step {step}: Required output from '{req_step_key}' is missing for process {process_id_str}. Please complete or regenerate step {req_step_key.replace('step','')}"
                 logger.error(error_msg)
                 return jsonify({'error': error_msg}), 400


             context_arg_index = idx + 1
             if len(prompt_args) > context_arg_index:
                 prompt_args[context_arg_index] = outputs_dict[req_step_key]
             else:

                 internal_error = f"Internal error: Mismatch between required_context_keys and prompt_args structure for step {step}."
                 logger.critical(internal_error)
                 return jsonify({'error': internal_error}), 500

        if not prompt_func:
            logger.error(f"No prompt function defined for Nebula step {step}")
            return jsonify({'error': 'Internal server error: Invalid step configuration'}), 500


        try:
            prompt = prompt_func(*prompt_args)
        except TypeError as e:
            logger.error(f"Error calling prompt function for step {step}: {e}. Check arguments.", exc_info=True)
            return jsonify({'error': f'Internal server error creating prompt for step {step}'}), 500


        api_key = NEBULA_API_KEYS.get(api_key_name)
        if not api_key:
             logger.error(f"API key not found for Nebula step {step} (Key Name: '{api_key_name}')")
             return jsonify({'error': 'Internal server error: API key configuration missing for this step'}), 500


        logger.info(f"Calling Gemini for Nebula step {step}, process {process_id_str}, model {model_id}")

        generator_output = gemini_generate(
             prompt=prompt, model_id=model_id, key=api_key, attempts=3, backoff_factor=1.8,
             model_display_name=f"{MODEL_NAMES.get(model_id, model_id)} (Nebula Step {step})"
        )

        step_result = None
        generation_successful = False
        for item in generator_output:

            if 'status' in item:
                logger.info(f"Nebula Step {step} (Process: {process_id_str}) Status: {item['status']}")
            elif 'result' in item:
                step_result = item['result']
                if isinstance(step_result, str) and step_result.startswith(ERROR_CODE):
                    logger.error(f"Nebula step {step} failed for process {process_id_str}: {step_result}")
                    step_result = None
                elif step_result is not None:
                     generation_successful = True
                break


        if not generation_successful or step_result is None:
            error_detail = step_result or "Generation failed after retries."
            logger.error(f"Nebula step {step} ultimately failed for process {process_id_str}. Last Error: {error_detail}")

            error_msg = f"Failed to generate output for step {step}." + (f" Details: {error_detail}" if error_detail != "Generation failed after retries." else "")
            return jsonify({'error': error_msg}), 500
        else:

             logger.info(f"Nebula step {step} successful for process {process_id_str}.")
             process_state['outputs'][step_key] = step_result


             is_code_fix = False
             if step == 4:

                 trimmed_result = step_result.strip()
                 if trimmed_result.startswith(('```html', '<!DOCTYPE', '```python', '# requirements:', 'import ', 'from ')):
                      is_code_fix = True
                      logger.info(f"Nebula Step 4 (Process: {process_id_str}): Output detected as code fix.")

                      if trimmed_result.startswith(('```html', '<!DOCTYPE')):
                           logger.info("Step 4 returned corrected Frontend code. Updating step 2 output.")
                           process_state['outputs']['step2'] = step_result
                           process_state['outputs'].pop('step3', None)
                           process_state['outputs']['step4'] = "[Verification Result: Frontend Code Corrected. Regenerate Step 3]"
                           process_state['current_step'] = 3
                      elif trimmed_result.startswith(('```python', '# requirements:', 'import ', 'from ')):
                           logger.info("Step 4 returned corrected Backend code. Updating step 3 output.")
                           process_state['outputs']['step3'] = step_result
                           process_state['outputs']['step4'] = "[Verification Result: Backend Code Corrected. Review Code.]"
                           process_state['current_step'] = 5
                      else:
                            logger.warning("Step 4 returned code, but couldn't determine if frontend/backend.")

                            process_state['outputs']['step4'] = step_result
                            process_state['current_step'] = 5
                 else:
                      logger.info(f"Nebula Step 4 (Process: {process_id_str}): Output detected as verification report.")

                      process_state['outputs']['step4'] = step_result
                      process_state['current_step'] = 5



             if step < 4 and not (step == 4 and is_code_fix and process_state['current_step'] < 5):
                 process_state['current_step'] = step + 1
             elif step == 4 and not is_code_fix:
                 process_state['current_step'] = 5


             session['nebula_processes'][process_id_str] = process_state
             session.modified = True


             if process_state['current_step'] == 5:
                 logger.info(f"Nebula process {process_id_str} reached completion state (Step 4 processed). Generating report and saving final state.")
                 nebula_outputs = process_state.get('outputs', {})


                 final_report_content = f"# Nebula Code Generation Report (Process ID: {process_id_str})\n\n## User Request\n```\n{process_state.get('query', 'N/A')}\n```\n\n"
                 final_report_content += f"## Step 1: Planning\n```markdown\n{nebula_outputs.get('step1', '[Not Available]')}\n```\n\n"
                 final_report_content += f"## Step 2: Frontend Code\n```html\n{nebula_outputs.get('step2', '[Not Available]')}\n```\n\n"
                 final_report_content += f"## Step 3: Backend Code\n```python\n{nebula_outputs.get('step3', '[Not Available]')}\n```\n\n"
                 final_report_content += f"## Step 4: Verification/Final Output\n"

                 step4_output = nebula_outputs.get('step4', '[Not Available]')
                 if is_code_fix:
                     final_report_content += f"**Note:** Step 4 resulted in code corrections. See updated code in Step 2 or Step 3 above.\n```text\n{step4_output}\n```\n"
                 else:
                     final_report_content += f"```markdown\n{step4_output}\n```\n"



                 report_filename_base = f"nebula_report_{sanitize_filename(process_state.get('query', 'process_'+process_id_str))}"

                 report_filepath_rel = create_output_file(report_filename_base, final_report_content, extension="md")

                 report_url = None
                 if report_filepath_rel:
                     report_url = f'/download/{os.path.basename(report_filepath_rel)}'
                 else:
                      logger.error(f"Failed to create report file for Nebula process {process_id_str}")


                 message_content_json = json.dumps({
                     'steps': nebula_outputs,
                     'user_query': process_state.get('query', 'N/A'),
                     'currentStep': 4
                 })

                 nebula_message_id = insert_message(
                     session_id=session_id,
                     message_type="nebula_output",
                     message_content=message_content_json,
                     is_research_output=False,
                     nebula_steps=nebula_outputs,
                     html_file=report_filepath_rel
                 )

                 if nebula_message_id:
                     logger.info(f"Nebula process {process_id_str} final state saved to DB. Message ID: {nebula_message_id}")
                 else:
                      logger.error(f"Failed to save completed Nebula process {process_id_str} to database!")



                 session['nebula_processes'].pop(process_id_str, None)
                 session.modified = True
                 logger.info(f"Cleaned up session state for completed Nebula process {process_id_str}")


                 response_data = {
                     'processId': process_id,
                     'step': step,
                     'output': step_result,
                     'status': 'nebula_complete',
                     'result': 'Nebula process completed successfully.',
                     'report_url': report_url,
                     'message_id': str(nebula_message_id) if nebula_message_id else None,
                     'is_code_fix': is_code_fix,
                     'final_outputs': nebula_outputs
                 }
                 return jsonify(response_data)
             else:

                 response_data = {
                     'processId': process_id,
                     'step': step,
                     'output': step_result,
                     'next_step': process_state['current_step']
                 }
                 return jsonify(response_data)

    except Exception as e:

        logger.error(f"Error in /nebula/step route (Process: {data.get('processId', 'N/A')}, Step: {data.get('step', 'N/A')}): {e}\n{traceback.format_exc()}", exc_info=True)
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500


@app.route('/clear_history', methods=['POST'])
def clear_history():
    try:
        session_id = get_current_session_id()
        if not session_id:

            logger.warning("Clear history called without an active session.")
            return jsonify({'status': 'Success', 'message': 'No active session to clear'}), 200


        cleared_pending = session.pop('pending_queries', None)
        if cleared_pending is not None:
            logger.info(f"Cleared {len(cleared_pending)} pending queries for session {session_id} during history clear.")
            session.modified = True

        cleared_nebula = session.pop('nebula_processes', None)
        if cleared_nebula is not None:
            logger.info(f"Cleared {len(cleared_nebula)} active Nebula processes for session {session_id} during history clear.")
            session.modified = True


        db = get_db()
        cursor = db.execute('DELETE FROM messages WHERE session_id = ?', (session_id,))
        deleted_count = cursor.rowcount
        db.commit()


        cache.delete_memoized(get_conversation_history, session_id)


        welcome_message = "Heyy there! I'm Stellar, and I can help you with research papers using Spectrum Mode, which includes Spectral Search! and building websites/apps with Nebula Mode! (Only exclusive to Crimson and Obsidian models). You can even Preview code blocks to see them live! I've got different models too, like Emerald for quick stuff or Obsidian for super complex things! ✨"
        insert_message(session_id, "stellar", welcome_message)


        logger.info(f"Cleared {deleted_count} history messages and reset state for session {session_id}")
        return jsonify({'status': 'Success', 'message': 'Conversation history cleared'})

    except sqlite3.Error as db_e:
         logger.error(f"Database error clearing history for session {session.get('sid', 'N/A')}: {db_e}\n{traceback.format_exc()}", exc_info=True)
         return jsonify({'status': 'Failed', 'message': f"Database error clearing history: {str(db_e)}"}), 500
    except Exception as e:
        logger.error(f"Error clearing history for session {session.get('sid', 'N/A')}: {e}\n{traceback.format_exc()}", exc_info=True)
        return jsonify({'status': 'Failed', 'message': f"Server error clearing history: {str(e)}"}), 500


@app.route('/download/<path:filename>')
def download_file(filename):
    logger.info(f"Download requested for file: {filename}")


    if '..' in filename or filename.startswith('/'):
        logger.warning(f"Directory traversal attempt blocked for download: {filename}")
        return "Invalid path", 400


    directory = os.path.abspath(os.path.join(os.path.dirname(__file__), "outputs"))


    safe_filename = os.path.basename(filename)
    file_path = os.path.join(directory, safe_filename)


    if not os.path.abspath(file_path).startswith(directory):
         logger.warning(f"Attempt to download file outside 'outputs' directory blocked: {filename} (Resolved: {file_path})")
         return "Access denied", 403


    if not os.path.exists(file_path) or not os.path.isfile(file_path):
        logger.error(f"Download failed: File not found at {file_path}")
        return jsonify({'status': 'Failed: File not found'}), 404

    logger.info(f"Sending file '{safe_filename}' from directory '{directory}' for download")

    return send_from_directory(directory, safe_filename, as_attachment=True)

@app.route('/view/<path:filename>')
def view_file(filename):
    logger.info(f"View requested for file: {filename}")
    if '..' in filename or filename.startswith('/'):
        logger.warning(f"Directory traversal attempt blocked for view: {filename}")
        return "Invalid path", 400

    directory = os.path.abspath(os.path.join(os.path.dirname(__file__), "outputs"))
    safe_filename = os.path.basename(filename)
    file_path = os.path.join(directory, safe_filename)


    if not os.path.abspath(file_path).startswith(directory):
         logger.warning(f"Attempt to view file outside 'outputs' directory blocked: {filename} (Resolved: {file_path})")
         return "Access denied", 403

    if not os.path.exists(file_path) or not os.path.isfile(file_path):
         logger.error(f"View failed: File not found at {file_path}")
         return "File not found", 404


    mimetype = 'text/plain'
    if safe_filename.lower().endswith(('.html', '.htm')):
        mimetype = 'text/html'
    elif safe_filename.lower().endswith('.md'):
         mimetype = 'text/markdown'
    elif safe_filename.lower().endswith('.css'):
        mimetype = 'text/css'
    elif safe_filename.lower().endswith('.js'):
        mimetype = 'application/javascript'


    logger.info(f"Serving file '{safe_filename}' for viewing with mimetype {mimetype}")

    return send_from_directory(directory, safe_filename, mimetype=mimetype)


@app.route('/default.min.css')
def serve_highlight_css():
    return send_from_directory('.', 'default.min.css')

@app.route('/highlight.min.js')
def serve_highlight_js():
    return send_from_directory('.', 'highlight.min.js')

@app.route('/marked.min.js')
def serve_marked():
    return send_from_directory('.', 'marked.min.js')

@app.route('/turndown.js')
def serve_turndown():
    return send_from_directory('.', 'turndown.js')


@app.route('/')
def index():
    # Let Flask-Session load or create the session when 'session' is accessed
    if 'initialized' not in session:
        session['initialized'] = True
        session.permanent = True 
        logger.info(f"Root access, initializing session: {session.sid}")
    else:
         logger.info(f"Root access, existing session found: {session.sid}")
    return send_from_directory('.', 'index.html')


if __name__ == '__main__':

    port = int(os.environ.get('PORT', 500))


    is_debug_mode = os.getenv('FLASK_ENV') != 'production'
    logger.info(f"Starting Flask server. Debug Mode: {is_debug_mode}, Port: {port}, Production Env: {IS_PRODUCTION}")
    app.run(host='0.0.0.0', port=port, debug=is_debug_mode, threaded=True)
