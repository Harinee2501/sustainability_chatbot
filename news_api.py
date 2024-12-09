import requests

def fetch_news():
    """
    Fetches the latest sustainability news using the News API.
    """
    api_key = "7afc18db8383496aa26fe0778d6127a5"  # Replace with your News API key
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": "sustainability OR climate change OR renewable energy OR sustainable agriculture OR SDGs OR environmental impact",
        "sources": "bbc-news, environment, the-hindu, reuters, national-geographic, bloomberg",
        "language": "en",
        "sortBy": "publishedAt",  # Sort by most recent
        "apiKey": api_key
    }

    try:
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            response_data = response.json()
            articles = response_data.get("articles", [])
            if not articles:
                return [{"title": "No articles found", "description": "Try a different query or check API limits.", "url": ""}]
            
            # Return top 5 articles
            return articles[:3]
        else:
            return [{"title": "Error fetching news", "description": response.text, "url": ""}]
    except Exception as e:
        return [{"title": "Exception occurred while fetching news", "description": str(e), "url": ""}]

if __name__ == "__main__":
    sustainability_news = fetch_news()
    print("Sustainability Hub 🌱\nLatest News on SDGs\n")
    for idx, article in enumerate(sustainability_news, start=1):
        print(f"{idx}. {article['title']}")
        print(f"   {article['description'] or 'No description available.'}")
        print(f"   [Read more]({article['url']})\n")





