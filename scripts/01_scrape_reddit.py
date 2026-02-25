"""Scrape NBA hot takes from Reddit using public JSON endpoints (no API key needed)."""

import json
import sys
import time
from pathlib import Path

import requests

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = RAW_DIR / "reddit_posts.jsonl"
CHECKPOINT_FILE = RAW_DIR / "scrape_checkpoint.json"

HEADERS = {"User-Agent": "LeLM-scraper/1.0"}

SEARCH_QUERIES = [
    "hot take",
    "unpopular opinion",
    "overrated underrated",
    "bold prediction",
    "worst take",
    "controversial opinion",
    "finals prediction",
    "MVP take",
    "GOAT debate",
]

SUBREDDIT = "nba"
COMMENTS_PER_POST = 10
MAX_PAGES_PER_QUERY = 4


def load_checkpoint():
    if CHECKPOINT_FILE.exists():
        return json.loads(CHECKPOINT_FILE.read_text())
    return {"scraped_ids": [], "completed_queries": []}


def save_checkpoint(state):
    CHECKPOINT_FILE.write_text(json.dumps(state))


def fetch_json(url, params=None):
    """Fetch a Reddit .json endpoint with retry on rate limit."""
    for attempt in range(3):
        try:
            resp = requests.get(url, headers=HEADERS, params=params, timeout=15)
        except requests.exceptions.Timeout:
            print(f"  Timeout (attempt {attempt + 1}), retrying...", flush=True)
            time.sleep(3)
            continue
        except requests.exceptions.ConnectionError:
            print(f"  Connection error (attempt {attempt + 1}), retrying...", flush=True)
            time.sleep(5)
            continue
        if resp.status_code == 429:
            wait = int(resp.headers.get("Retry-After", 10))
            print(f"  Rate limited, waiting {wait}s...", flush=True)
            time.sleep(wait)
            continue
        if resp.status_code != 200:
            print(f"  HTTP {resp.status_code}, skipping", flush=True)
            return None
        return resp.json()
    return None


def extract_post(post_data):
    d = post_data["data"]
    return {
        "id": d["id"],
        "type": "post",
        "title": d.get("title", ""),
        "selftext": d.get("selftext", ""),
        "score": d.get("score", 0),
        "num_comments": d.get("num_comments", 0),
        "created_utc": d.get("created_utc", 0),
    }


def fetch_comments(post_id, post_title):
    """Fetch top comments for a post."""
    url = f"https://www.reddit.com/r/{SUBREDDIT}/comments/{post_id}.json"
    data = fetch_json(url, params={"sort": "top", "limit": COMMENTS_PER_POST})
    if not data or len(data) < 2:
        return []
    comments = []
    for child in data[1]["data"].get("children", []):
        if child["kind"] != "t1":
            continue
        d = child["data"]
        comments.append({
            "id": d["id"],
            "type": "comment",
            "post_title": post_title,
            "body": d.get("body", ""),
            "score": d.get("score", 0),
            "created_utc": d.get("created_utc", 0),
        })
    return comments[:COMMENTS_PER_POST]


def scrape_listing(f, scraped_ids, label, url, base_params):
    """Scrape a paginated listing (search or top/hot)."""
    after = None
    for page in range(MAX_PAGES_PER_QUERY):
        params = {**base_params, "limit": 100}
        if after:
            params["after"] = after

        data = fetch_json(url, params)
        if not data:
            break
        posts = data["data"]["children"]
        if not posts:
            break

        for post_wrap in posts:
            if post_wrap["kind"] != "t3":
                continue
            post = extract_post(post_wrap)
            if post["id"] in scraped_ids:
                continue

            f.write(json.dumps(post) + "\n")
            scraped_ids.add(post["id"])

            # Only fetch comments for posts with enough engagement
            if post.get("num_comments", 0) >= 10:
                time.sleep(3)
                comments = fetch_comments(post["id"], post["title"])
                for comment in comments:
                    if comment["id"] not in scraped_ids:
                        f.write(json.dumps(comment) + "\n")
                        scraped_ids.add(comment["id"])

        f.flush()
        after = data["data"].get("after")
        if not after:
            break
        print(f"  [{label}] page {page + 1}, {len(scraped_ids)} items total", flush=True)
        time.sleep(2)


def scrape():
    checkpoint = load_checkpoint()
    scraped_ids = set(checkpoint["scraped_ids"])
    completed_queries = set(checkpoint["completed_queries"])

    print(f"Resuming: {len(scraped_ids)} items, {len(completed_queries)} queries done", flush=True)

    with open(OUTPUT_FILE, "a") as f:
        # Search queries
        for query in SEARCH_QUERIES:
            if query in completed_queries:
                print(f"Skipping completed: '{query}'", flush=True)
                continue

            print(f"Searching: '{query}'...", flush=True)
            try:
                scrape_listing(
                    f, scraped_ids, query,
                    f"https://www.reddit.com/r/{SUBREDDIT}/search.json",
                    {"q": query, "restrict_sr": "on", "sort": "top", "t": "all"},
                )
                completed_queries.add(query)
                save_checkpoint({
                    "scraped_ids": list(scraped_ids),
                    "completed_queries": list(completed_queries),
                })
                print(f"  Done. Total: {len(scraped_ids)}", flush=True)
            except Exception as e:
                print(f"  Error on '{query}': {e}", flush=True)
                save_checkpoint({
                    "scraped_ids": list(scraped_ids),
                    "completed_queries": list(completed_queries),
                })

        # Top/hot posts
        for sort_method in ["top", "hot"]:
            label = f"__{sort_method}__"
            if label in completed_queries:
                print(f"Skipping completed: {label}", flush=True)
                continue

            print(f"Fetching {sort_method} posts...", flush=True)
            try:
                scrape_listing(
                    f, scraped_ids, sort_method,
                    f"https://www.reddit.com/r/{SUBREDDIT}/{sort_method}.json",
                    {"t": "year"},
                )
                completed_queries.add(label)
                save_checkpoint({
                    "scraped_ids": list(scraped_ids),
                    "completed_queries": list(completed_queries),
                })
                print(f"  Done. Total: {len(scraped_ids)}", flush=True)
            except Exception as e:
                print(f"  Error on {sort_method}: {e}", flush=True)
                save_checkpoint({
                    "scraped_ids": list(scraped_ids),
                    "completed_queries": list(completed_queries),
                })

    print(f"\nDone! Total items: {len(scraped_ids)}", flush=True)
    print(f"Output: {OUTPUT_FILE}", flush=True)


if __name__ == "__main__":
    scrape()
