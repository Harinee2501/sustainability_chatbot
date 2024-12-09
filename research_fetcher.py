import wikipedia
import feedparser
import urllib.parse

def fetch_wikipedia_summary(query):
    """
    Fetches a summary from Wikipedia based on the query.
    """
    try:
        return wikipedia.summary(query, sentences=3)
    except wikipedia.exceptions.DisambiguationError as e:
        return f"Disambiguation error: {e.options}"
    except wikipedia.exceptions.PageError:
        return "No Wikipedia page found for this topic."

def fetch_research(query):
    """
    Fetches research papers from arXiv based on the query.
    """
    # Encode the query to handle spaces and special characters
    encoded_query = urllib.parse.quote(query)
    url = f"http://export.arxiv.org/api/query?search_query=all:{encoded_query}&start=0&max_results=5"
    
    feed = feedparser.parse(url)
    research_papers = []

    for entry in feed.entries:
        title = entry.title
        authors = ", ".join([author.name for author in entry.authors])
        link = entry.link
        research_papers.append({"title": title, "authors": authors, "link": link})
    
    return research_papers

# Example Usage
if __name__ == "__main__":
    query = "machine learning"
    
    print("Fetching Wikipedia Summary:")
    wikipedia_summary = fetch_wikipedia_summary(query)
    print(wikipedia_summary)
    
    print("\nFetching Research Papers from arXiv:")
    arxiv_research = fetch_research(query)
    for idx, paper in enumerate(arxiv_research, start=1):
        print(f"{idx}. {paper['title']} by {paper['authors']}")
        print(f"   Link: {paper['link']}")


