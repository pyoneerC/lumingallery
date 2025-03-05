import requests
import time
from datetime import datetime

# Base URL and headers for Reddit
BASE_URL = "https://www.reddit.com/r/news"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36"
}


def fetch_posts(sort="hot", limit=10):
    """
    Fetch posts for a given sort type ("hot", "new", "rising") with a limit.
    For each post, extract:
      - title
      - external URL (the actual link to the news article)
      - upvotes (score)
      - created timestamp
      - thumbnail (from preview data, if available)
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
        # For link posts, the "url" field contains the external link.
        external_url = post_data.get("url", "No external URL")

        # Attempt to get the preview thumbnail if available
        thumbnail = None
        if "preview" in post_data:
            try:
                # Reddit returns a list of images; we take the source image URL from the first one.
                thumbnail = post_data["preview"]["images"][0]["source"]["url"]
            except Exception as e:
                thumbnail = None
        if not thumbnail:
            thumbnail = "No thumbnail available"

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
    Fetch up to 5 top-level comments for a given post.
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

    return comments[:5]  # Limit to 5 comments


def convert_timestamp(timestamp):
    """Convert a Unix timestamp to MM/DD/YYYY format."""
    return datetime.utcfromtimestamp(timestamp).strftime('%m/%d/%Y')


def main():
    all_posts = []
    # Fetch 10 posts from each category: Hot, New, and Rising
    for sort in ["hot", "new", "rising"]:
        print(f"\nFetching {sort.upper()} posts...\n")
        posts = fetch_posts(sort, limit=10)
        all_posts.extend([(sort, post) for post in posts])

    # Save results to a file
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

            # Fetch and print comments
            comments = fetch_comments(post["id"])
            if comments:
                file.write("    Top Comments:\n")
                print("    Top Comments:")
                for comment in comments:
                    file.write(f"        - {comment}\n")
                    print(f"        - {comment}")
            else:
                file.write("    No comments found.\n")
                print("    No comments found.")

            file.write("\n" + "-" * 80 + "\n\n")
            print("-" * 80)

            # Pause briefly to avoid Reddit rate limiting
            time.sleep(2)

    print("\n✅ Saved all posts and comments to 'reddit_news_posts_and_comments.txt'")


if __name__ == "__main__":
    main()
