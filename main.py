import requests
import re
import json
from datetime import datetime
from bs4 import BeautifulSoup
import asyncio
import aiohttp

# --- Configuration ---
BASE_URL = "https://www.reddit.com/r/news"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/110.0.0.0 Safari/537.36"
}

SENTIMENT_API_URL = "https://api.groq.com/openai/v1/chat/completions"
SENTIMENT_API_KEY = "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"  # Use your secure key!


# --- Synchronous helper functions ---

def fetch_social_preview(url):
    """
    Given an external URL, fetch the page and try to extract a social preview image
    from meta tags (og:image or twitter:image).
    """
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        if r.status_code == 200:
            soup = BeautifulSoup(r.text, "html.parser")
            meta_tag = soup.find("meta", property="og:image")
            if meta_tag and meta_tag.get("content"):
                return meta_tag["content"]
            meta_tag = soup.find("meta", property="twitter:image")
            if meta_tag and meta_tag.get("content"):
                return meta_tag["content"]
    except Exception as e:
        print(f"Error fetching social preview for {url}: {e}")
    return "No social preview available"


def fetch_posts(sort="hot", limit=10):
    """
    Fetch posts for a given sort type ("hot", "new", or "rising") with a limit.
    Extracts title, external URL, upvotes, timestamp, and thumbnail.
    """
    url = f"{BASE_URL}/{sort}.json?limit={limit}"
    response = requests.get(url, headers=HEADERS)
    if response.status_code != 200:
        print(f"Failed to fetch {sort} posts: {response.status_code}")
        return []

    data = response.json()
    posts = []
    for post in data["data"]["children"]:
        post_data = post["data"]
        external_url = post_data.get("url", "No external URL")
        thumbnail = None
        if "preview" in post_data:
            try:
                thumbnail = post_data["preview"]["images"][0]["source"]["url"]
            except Exception:
                thumbnail = None
        if not thumbnail or not thumbnail.startswith("http"):
            thumbnail = fetch_social_preview(external_url)

        posts.append({
            "title": post_data.get("title", "No Title"),
            "id": post_data.get("id", ""),
            "external_url": external_url,
            "upvotes": post_data.get("score", 0),
            "timestamp": post_data.get("created_utc", 0),
            "thumbnail": thumbnail
        })
    return posts


def fetch_comments(post_id):
    """
    Fetch up to 10 top-level comments for a given post.
    """
    url = f"{BASE_URL}/comments/{post_id}/.json"
    response = requests.get(url, headers=HEADERS)
    if response.status_code != 200:
        print(f"Failed to fetch comments for post {post_id}: {response.status_code}")
        return []

    data = response.json()
    comments = []
    try:
        for comment in data[1]["data"]["children"]:
            if "body" in comment["data"]:
                comments.append(comment["data"]["body"])
    except Exception as e:
        print(f"Error parsing comments for post {post_id}: {e}")
    return comments[:10]


def convert_timestamp(timestamp):
    """Convert a Unix timestamp to MM/DD/YYYY format."""
    return datetime.utcfromtimestamp(timestamp).strftime('%m/%d/%Y')


# --- Asynchronous functions for sentiment analysis ---

async def analyze_sentiment_async(text, session):
    """
    Analyze the sentiment of a given text by calling the external AI sentiment API.
    Returns a numerical score (0 to 100) or None.
    """
    headers = {
        "Authorization": f"Bearer {SENTIMENT_API_KEY}",
        "Content-Type": "application/json"
    }
    prompt = (
        f"Analyze the sentiment of the following text on a scale from 0 to 100, "
        f"where 0 is extremely negative, 50 is neutral, and 100 is extremely positive.\n"
        f"Text: \"{text}\"\nRespond only with the numerical score."
    )
    payload = {
        "model": "llama-3.3-70b-versatile",
        "messages": [
            {
                "role": "system",
                "content": "You are a sentiment analysis tool. Provide a sentiment score from 0 to 100."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_tokens": 100,
        "temperature": 0.75,
        "top_p": 1,
        "frequency_penalty": 0,
        "presence_penalty": 0.3,
        "n": 1,
        "stop": ["\n", "User:"],
        "logit_bias": {}
    }
    try:
        async with session.post(SENTIMENT_API_URL, headers=headers, json=payload, timeout=30) as response:
            if response.status == 200:
                result = await response.json()
                content = result["choices"][0]["message"]["content"].strip()
                try:
                    sentiment_score = float(content)
                except ValueError:
                    match = re.search(r'\d+(\.\d+)?', content)
                    sentiment_score = float(match.group()) if match else None
                return sentiment_score
            else:
                print("Sentiment API call failed with status:", response.status)
    except Exception as e:
        print("Error during sentiment analysis:", e)
    return None


async def process_posts_json(all_posts, sentiment_limit=3):
    """
    Process all posts: for each post, fetch comments and analyze sentiment
    for only the first `sentiment_limit` comments concurrently.
    Build and return a JSON-serializable result.
    """
    results = []
    async with aiohttp.ClientSession() as session:
        for sort, post in all_posts:
            post_date = convert_timestamp(post["timestamp"])
            comments = fetch_comments(post["id"])
            # Only analyze up to 'sentiment_limit' comments (e.g., 3)
            sentiment_comments = comments[:sentiment_limit] if comments else []
            analyzed_comments = []
            if sentiment_comments:
                sentiments = await asyncio.gather(
                    *[analyze_sentiment_async(comment, session) for comment in sentiment_comments]
                )
                for comment, sentiment in zip(sentiment_comments, sentiments):
                    analyzed_comments.append({
                        "text": comment,
                        "sentiment": sentiment
                    })
                valid_sentiments = [s for s in sentiments if s is not None]
                aggregate_sentiment = sum(valid_sentiments) / len(valid_sentiments) if valid_sentiments else None
            else:
                aggregate_sentiment = None

            result_post = {
                "category": sort.upper(),
                "title": post["title"],
                "external_url": post["external_url"],
                "upvotes": post["upvotes"],
                "date": post_date,
                "thumbnail": post["thumbnail"],
                "analyzed_comments": analyzed_comments,
                "aggregate_sentiment": aggregate_sentiment
            }
            results.append(result_post)
    return results


def main():
    all_posts = []
    for sort in ["hot", "new", "rising"]:
        print(f"Fetching {sort.upper()} posts...")
        posts = fetch_posts(sort, limit=10)
        all_posts.extend([(sort, post) for post in posts])

    # Process posts asynchronously to analyze sentiment on only a few comments per post.
    results = asyncio.run(process_posts_json(all_posts, sentiment_limit=3))

    # Output the final JSON result (for example, printing to stdout)
    output_json = json.dumps({"posts": results}, indent=4)
    print(output_json)


if __name__ == "__main__":
    main()
