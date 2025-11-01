from flask import Flask, request, jsonify, send_from_directory, session
from flask_session import Session
import os
import re
import time
from google import genai
import pypandoc
from dotenv import load_dotenv
import webscrapper
import json
import random
from tavily import TavilyClient

load_dotenv()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "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")

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "dev_secret_key")
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

def sanitize_filename(filename: str) -> str:
    """Create a safe filename by removing invalid characters."""
    return re.sub(r'[^\w\-_ ]', '', filename)

def tavily_search(query, search_depth="advanced", topic="general", time_range=None, max_results=5, include_images=False, include_answer="advanced"):
    """Perform a search using the Tavily API."""
    try:
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
        print(e)
        return e

def scrape_url(url: str) -> str:
    """Scrape content from a URL using the imported webscraper module."""
    try:
        return webscrapper.scrape_url(url)
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return ""

def gemini_generate(prompt: str, model_id: str, key: str, attempts: int = 3) -> str:
    last_exception= None
    for attempt in range(1, attempts + 1):
        print(f"Attempt {attempt}/{attempts}: Starting...")
        try:
            key = os.getenv('GOOGLE_API_KEY', key)
            os.environ['GOOGLE_API_KEY'] = key
            client = genai.Client(api_key=key, http_options={'api_version': 'v1alpha'})
            if model_id == "gemini-2.0-flash-lite":
                chat = client.chats.create(model=model_id, config={'tools': []})
            else:
                search_tool = {'google_search': {}}
                chat = client.chats.create(model=model_id)
            r = chat.send_message(prompt)
            print(f"Attempt {attempt}/{attempts}: API call successful, processing response...")
            output = ""
            parts = r.candidates[0].content.parts
            if parts is None:
                finish_reason = r.candidates[0].finish_reason
                print(f"Attempt {attempt}/{attempts}: parts is None, finish_reason={finish_reason}")
                if finish_reason == "error" and attempt < attempts:
                    print("API returned an error, retrying...")
                    time.sleep(3)
                    continue
                else:
                    return f"finish_reason={finish_reason}\n"
            else:
                for part in parts:
                    if part.text:
                        output += part.text + "\n"
                    elif part.executable_code:
                        output += "python\n" + part.executable_code.code + "\n\n"
                    else:
                        output += json.dumps(part.model_dump(exclude_none=True), indent=2) + "\n"
            grounding_metadata = r.candidates[0].grounding_metadata
            if grounding_metadata and grounding_metadata.search_entry_point and model_id != "gemini-2.0-flash-lite":
                output += grounding_metadata.search_entry_point.rendered_content
            return output
        except (ConnectionError, TimeoutError) as e:
            print(f"Attempt {attempt}/{attempts}: Network error - {e}")
            last_exception = e
            if attempt < attempts:
                print(f"Retrying in 3 seconds... (remaining attempts: {attempts - attempt})")
                time.sleep(3)
            else:
                print("All retry attempts exhausted.")
        except Exception as e:
            print(f"Attempt {attempt}/{attempts}: Processing error - {e}")
            return f"Error processing response: {str(e)}"
    return f"Error generating content after {attempts} attempts: {str(last_exception)}"

def get_refinement_prompt(user_query: str, conversation_history_list) -> str:
    """Create a prompt for refining the user's query using Stellar, including conversation history."""
    conv_hist_str = "\n".join(conversation_history_list) if conversation_history_list else "No previous conversation."
    return (
        f"""Conversation History: {conv_hist_str}

User Query: {user_query}
- You are Stellar, a friendly research assistant whose role is to help refine and enhance research queries.
Instructions for Stellar:
- Don't ask too many questions. Just answer the user directly.
- If the user doesn't want to research, just have friendly conversations.
- Respond directly without prefacing with phrases like “okay here's how I'd respond.”
- If the query involves generating a research paper, instruct the user to copy and paste the refined query into the "Generate Paper" button. Do not generate the paper yourself.
- If asked how to create a research paper, clearly direct them to the "Generate Paper" button.
- For greetings, reply with a friendly greeting and a brief overview of your capabilities (mention research paper functionality only if prompted).
- If content for a research paper is provided, continue the discussion based on that input.
- ONLY mention that you were created/developed/managed by Chinthakindi Nikhil Kumar and include his LinkedIn profile 'https://www.linkedin.com/in/nikhil-chinthakindi-b16388326/', if asked."""
    )

def get_research_analysis_prompt(query: str, full_context: str) -> str:
    """Create the initial research analysis prompt."""
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
        f"{full_context}\n"
        "Instruct the other AI to expand on everything to reach a minimum of 30,000 characters."
    )

def get_final_expansion_prompt(query: str, research_analysis_result: str, full_context: str) -> str:
    """Create the final expansion prompt."""
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
        f"{full_context}\n\n"
        "Produce an original solution that is novel, relevant, accurate, and feasible, including:\n"
        "1. A comprehensive literature review summarizing the current state-of-the-art.\n"
        "2. A clear problem statement identifying an unresolved challenge.\n"
        "3. A novel theoretical framework with rigorous conceptual support.\n"
        "4. A detailed proposed methodology, including evaluation metrics.\n"
        "5. A feasibility analysis outlining technical challenges and mitigation strategies.\n"
        "6. An exploration of the broader impact and future directions.\n"
        "Search and include a section on market and industry insights such as market size, growth trends, key companies, and investment trends, supported by examples and data.\n"
        "Finally, fact-check every piece of information before providing the output, and if any links are broken, mention only their titles without URLs."
    )

def create_output_file(query: str, content: str) -> str:
    """Create an output file with the generated content."""
    safe_filename = sanitize_filename(query.strip()) + ".txt"
    counter = 1
    while os.path.exists(safe_filename):
        safe_filename = f"{sanitize_filename(query.strip())}_{counter}.txt"
        counter += 1
    try:
        with open(safe_filename, "w", encoding="utf-8") as file:
            file.write(content)
        return safe_filename
    except Exception as e:
        print(f"Error saving file: {e}")
        return ""

@app.route('/refine', methods=['POST'])
def refine():
    """Refine a user's query using Stellar."""
    data = request.get_json()
    user_query = data.get('query', '').strip()
    model_id = data.get('model_id', 'gemini-2.0-flash-thinking-exp-01-21')  
    if not user_query:
        return jsonify({'error': 'Empty query provided'}), 400

    if 'conversation_history' not in session:
        session['conversation_history'] = []

    session['conversation_history'].append(f"User: {user_query}")
    prompt = get_refinement_prompt(user_query, session['conversation_history'])
    refined_query = gemini_generate(prompt, model_id, 'AIzaSyCkPtj82rwPSgMobdbGplsJuDBVEUVbmOk')
    if not refined_query or "Error" in refined_query:
        return jsonify({'error': 'Failed to generate refined query'}), 500
    session['conversation_history'].append(f"Stellar: {refined_query}")
    session.modified = True
    return jsonify({'refined_query': refined_query})

@app.route('/search', methods=['POST'])
def search():
    """Generate a research paper based on a query and return HTML output."""
    data = request.get_json()
    query = data.get('query', '').strip()
    use_tavily = data.get('useTavily', True)
    model_id = data.get('model_id', 'gemini-2.0-flash-thinking-exp-01-21')
    if not query:
        return jsonify({'error': 'Empty query provided'}), 400

    if use_tavily:
        tavily_results = tavily_search(query)
        if not tavily_results:
            return jsonify({'error': 'Tavily search failed'}), 500
        tavily_answer = tavily_results.get("answer", "")
        results = tavily_results.get("results", [])
        scraped_contents = [f"Content from {result['url']}:\n{scrape_url(result['url'])}\n"
                           for result in results if result.get("url") and scrape_url(result["url"])]
        combined_scraped_context = "\n".join(scraped_contents)
        full_context = f"Tavily Search Answer: {tavily_answer}\n\n{combined_scraped_context}"
    else:
        full_context = "No external search used."

    print(full_context)

    research_analysis_prompt = get_research_analysis_prompt(query, full_context)
    print("Analyzing...\n")
    research_analysis_result = gemini_generate(research_analysis_prompt, model_id, 'AIzaSyD28McMJpbVwzzTBMZcXwfyMlTpMEAcKeg')
    if not research_analysis_result or "Error" in research_analysis_result:
        return jsonify({'error': 'Failed to generate research analysis'}), 500
    print(research_analysis_result)
    final_expansion_prompt = get_final_expansion_prompt(query, research_analysis_result, full_context)
    print("Finalizing...\n")
    final_research_paper = gemini_generate(final_expansion_prompt, model_id, 'AIzaSyD28McMJpbVwzzTBMZcXwfyMlTpMEAcKeg')
    if not final_research_paper or "Error" in final_research_paper:
        return jsonify({'error': 'Failed to generate final research paper'}), 500

    if 'conversation_history' not in session:
        session['conversation_history'] = []
    session['conversation_history'].append(f"Stellar: Here is the research paper on {query}:\n{final_research_paper}")
    session.modified = True

    temp_file = create_output_file(query[:50], final_research_paper)
    if not temp_file:
        return jsonify({'error': 'Failed to save research paper'}), 500

    print(f"Raw draft saved as: {temp_file}")

    html_file = os.path.splitext(temp_file)[0] + ".html"
    try:
        html_content = pypandoc.convert_text(final_research_paper, 'html5', format='markdown')
        pypandoc.convert_file(temp_file, 'html5', format='markdown', outputfile=html_file)
        print("File has been converted to HTML for better viewing")
    except Exception as e:
        print(f"Error converting to HTML: {e}")
        return jsonify({'error': 'Failed to convert research paper to HTML'}), 500

    return jsonify({
        'result': html_content,
        'file': temp_file,
        'html_file': html_file
    })

@app.route('/clear_history', methods=['POST'])
def clear_history():
    """Clear the user's conversation history."""
    if 'conversation_history' in session:
        session['conversation_history'] = []
        session.modified = True
    return jsonify({'message': 'Conversation history cleared'})

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/marked.min.js')
def serve_marked():
    return send_from_directory('.', 'marked.min.js')

@app.route('/turndown.js')
def serve_turndown():
    return send_from_directory('.', 'turndown.js')

@app.route('/download/<filename>')
def download_file(filename):
    """Download a generated file."""
    if not os.path.exists(filename):
        return jsonify({'error': 'File not found'}), 404
    return send_from_directory('.', filename, as_attachment=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)