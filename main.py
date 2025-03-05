import requests
import time
import re
from datetime import datetime
from bs4 import BeautifulSoup

# --- Configuration ---
BASE_URL = "https://www.reddit.com/r/news"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/110.0.0.0 Safari/537.36"
}

# Sentiment API settings
SENTIMENT_API_URL = "https://api.groq.com/openai/v1/chat/completions"
SENTIMENT_API_KEY = "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"  # Use your secure key!


# --- Functions ---

def fetch_social_preview(url):
    """
    Given an external URL, try to fetch the page and parse meta tags (og:image or twitter:image)
    to extract a social preview image.
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
    Extract title, external URL, upvotes, timestamp, and thumbnail.
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

        # First try: use Reddit's preview data for the thumbnail
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

    return comments[:10]  # Limit to 10 comments


def convert_timestamp(timestamp):
    """Convert a Unix timestamp to MM/DD/YYYY format."""
    return datetime.utcfromtimestamp(timestamp).strftime('%m/%d/%Y')


def analyze_sentiment(text):
    """
    Analyze the sentiment of a given text by calling the external AI sentiment API.
    Returns a numerical score (0 to 100) or None if analysis fails.
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
        response = requests.post(SENTIMENT_API_URL, headers=headers, json=payload, timeout=30)
        if response.status_code == 200:
            result = response.json()
            content = result["choices"][0]["message"]["content"].strip()
            try:
                sentiment_score = float(content)
            except ValueError:
                match = re.search(r'\d+(\.\d+)?', content)
                sentiment_score = float(match.group()) if match else None
            return sentiment_score
        else:
            print("Sentiment API call failed with status:", response.status_code)
    except Exception as e:
        print("Error during sentiment analysis:", e)
    return None


def main():
    all_posts = []
    # Fetch 10 posts from each category: Hot, New, and Rising
    for sort in ["hot", "new", "rising"]:
        print(f"\nFetching {sort.upper()} posts...\n")
        posts = fetch_posts(sort, limit=10)
        all_posts.extend([(sort, post) for post in posts])

    with open("reddit_news_posts_and_comments.txt", "w", encoding="utf-8") as file:
        for idx, (sort, post) in enumerate(all_posts, start=1):
            post_date = convert_timestamp(post["timestamp"])
            file.write(f"{idx}. [{sort.upper()}] {post['title']}\n")
            file.write(f"    External URL: {post['external_url']}\n")
            file.write(f"    Upvotes: {post['upvotes']} | Date: {post_date}\n")
            file.write(f"    Thumbnail: {post['thumbnail']}\n")

            print(f"{idx}. [{sort.upper()}] {post['title']}")
            print(f"    External URL: {post['external_url']}")
            print(f"    Upvotes: {post['upvotes']} | Date: {post_date}")
            print(f"    Thumbnail: {post['thumbnail']}")

            comments = fetch_comments(post["id"])
            sentiment_scores = []

            if comments:
                file.write("    Top Comments:\n")
                print("    Top Comments:")
                for comment in comments:
                    file.write(f"        - {comment}\n")
                    print(f"        - {comment}")
                    sentiment = analyze_sentiment(comment)
                    if sentiment is not None:
                        sentiment_scores.append(sentiment)
                    # Brief pause between sentiment calls to avoid rate limits
                    time.sleep(1)
            else:
                file.write("    No comments found.\n")
                print("    No comments found.")

            # Compute aggregate sentiment score for the post if we got any valid scores
            if sentiment_scores:
                aggregate_sentiment = sum(sentiment_scores) / len(sentiment_scores)
                file.write(f"    Aggregate Sentiment Score: {aggregate_sentiment:.2f}\n")
                print(f"    Aggregate Sentiment Score: {aggregate_sentiment:.2f}")
            else:
                file.write("    Aggregate Sentiment Score: N/A\n")
                print("    Aggregate Sentiment Score: N/A")

            file.write("\n" + "-" * 80 + "\n\n")
            print("-" * 80)
            # Pause to avoid Reddit rate-limiting
            time.sleep(2)

    print("\n✅ Saved all posts, comments, and aggregate sentiment analysis to 'reddit_news_posts_and_comments.txt'")


if __name__ == "__main__":
    main()
