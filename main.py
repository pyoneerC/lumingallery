import requests
import re
import json
import random
from datetime import datetime
from datetime import timezone
from bs4 import BeautifulSoup
import asyncio
import aiohttp

# --- Configuration ---
BASE_URL = "https://www.reddit.com/r/news"
HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/110.0.0.0 Safari/537.36")
}

# Sentiment API settings
SENTIMENT_API_URL = "https://api.groq.com/openai/v1/chat/completions"
SENTIMENT_API_KEY = "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"  # Use your secure key!


# --- Synchronous helper functions ---

def fetch_social_preview(url):
    """
    Fetches the external URL and extracts a social preview image
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
    Fetches posts for a given sort (hot, new, or rising).
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
    Fetches up to 10 top-level comments for a given post.
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
    """Converts a Unix timestamp to MM/DD/YYYY format."""
    return datetime.fromtimestamp(timestamp, timezone.utc).strftime('%m/%d/%Y')


# --- Asynchronous functions ---

async def analyze_sentiment_async(text, session):
    """
    Analyzes the sentiment of the given text by calling the external AI API.
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
            {"role": "system",
             "content": "You are a sentiment analysis tool. Provide a sentiment score from 0 to 100."},
            {"role": "user", "content": prompt}
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
                    return float(content)
                except ValueError:
                    match = re.search(r'\d+(\.\d+)?', content)
                    return float(match.group()) if match else None
            else:
                print("Sentiment API call failed with status:", response.status)
    except Exception as e:
        print("Error during sentiment analysis:", e)
    return None


async def process_post(sort, post, session):
    """
    Processes a single post: fetches its top comments (via an async-to-thread wrapper),
    randomly selects one comment for sentiment analysis, and builds a result dictionary.
    """
    # Run the synchronous fetch_comments in a thread to avoid blocking
    comments = await asyncio.to_thread(fetch_comments, post["id"])
    top_comments = comments[:10]
    if top_comments:
        random_comment = random.choice(top_comments)
        sentiment = await analyze_sentiment_async(random_comment, session)
    else:
        sentiment = None

    return {
        "category": sort.upper(),
        "title": post["title"],
        "external_url": post["external_url"],
        "upvotes": post["upvotes"],
        "date": convert_timestamp(post["timestamp"]),
        "thumbnail": post["thumbnail"],
        "top_comments": top_comments,
        "aggregate_sentiment": sentiment
    }


async def process_all_posts(all_posts):
    """
    Processes all posts concurrently. For each post, it fetches and processes the comments
    and performs sentiment analysis on one random comment.
    Returns a list of results.
    """
    async with aiohttp.ClientSession() as session:
        tasks = [
            process_post(sort, post, session)
            for sort, post in all_posts
        ]
        return await asyncio.gather(*tasks)


def main():
    # Collect posts from each category
    all_posts = []
    for sort in ["hot", "new", "rising"]:
        print(f"Fetching {sort.upper()} posts...")
        posts = fetch_posts(sort, limit=10)
        all_posts.extend([(sort, post) for post in posts])

    # Process all posts concurrently
    results = asyncio.run(process_all_posts(all_posts))

    # Get current time for "last updated"
    last_updated = datetime.now(timezone.utc).strftime('%m/%d/%Y %H:%M:%S UTC')

    # Build final JSON result
    final_result = {"posts": results, "last_updated": last_updated}
    output_json = json.dumps(final_result, indent=4)
    print(output_json)


if __name__ == "__main__":
    main()
