import requests
from bs4 import BeautifulSoup

def scrape_url(url):
    headers = {'User-Agent': 'Mozilla/5.0 (compatible; MyScraper/1.0)'}
    try:
        response = requests.get(url, headers=headers, timeout=15) # added timeout here

        if response.status_code != 200:
            raise Exception(f"Failed to retrieve {url}: Status code {response.status_code}")

        html = response.text
        soup = BeautifulSoup(html, 'html.parser')

        title = soup.title.string.strip() if soup.title and soup.title.string else "No title found"
        paragraphs = [p.get_text(strip=True) for p in soup.find_all('p')]

        a= {
            'url': url,
            'title': title,
            'paragraphs': paragraphs,
            'raw_html': html
        }
        b=""
        print("Title:", a['title'])
        print("\nParagraphs:")
        for p in a['paragraphs']:
            b+=p
        return b

    except requests.exceptions.Timeout: # catching the timeout exception
        return "Took too long or some shi" # returning the message if timeout occurs

    except Exception as e: # catching other potential exceptions
        return f"Error occurred: {e}" # returning a generic error message
