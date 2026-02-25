"""Scrape pro-KD content from Reddit using public JSON endpoints."""

import json
import time
from pathlib import Path

import requests

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = RAW_DIR / "kd_posts.jsonl"
CHECKPOINT_FILE = RAW_DIR / "kd_scrape_checkpoint.json"

HEADERS = {"User-Agent": "LeLM-scraper/1.0"}

# Pro-KD search queries across multiple subreddits
SEARCHES = [
    ("nba", "Kevin Durant best scorer"),
    ("nba", "KD best scorer ever"),
    ("nba", "Durant underrated"),
    ("nba", "Kevin Durant unstoppable"),
    ("nba", "KD greatest"),
    ("nba", "Durant legacy"),
    ("nba", "Kevin Durant MVP"),
    ("nba", "KD clutch"),
    ("nba", "Durant 50 points"),
    ("nba", "Kevin Durant playoff"),
    ("nba", "KD scoring title"),
    ("nba", "Durant hall of fame"),
    ("GoNets", "Kevin Durant"),
    ("GoNets", "KD"),
    ("suns", "Kevin Durant"),
    ("thunder", "Kevin Durant"),
    ("warriors", "Kevin Durant finals"),
]

COMMENTS_PER_POST = 10
MAX_PAGES = 3


def load_checkpoint():
    if CHECKPOINT_FILE.exists():
        return json.loads(CHECKPOINT_FILE.read_text())
    return {"scraped_ids": [], "completed_queries": []}


def save_checkpoint(state):
    CHECKPOINT_FILE.write_text(json.dumps(state))


def fetch_json(url, params=None):
    for attempt in range(3):
        try:
            resp = requests.get(url, headers=HEADERS, params=params, timeout=15)
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
            print(f"  Connection issue (attempt {attempt + 1}), retrying...", flush=True)
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


def fetch_comments(subreddit, post_id, post_title):
    url = f"https://www.reddit.com/r/{subreddit}/comments/{post_id}.json"
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
            "source": "kd_scrape",
        })
    return comments[:COMMENTS_PER_POST]


def scrape():
    checkpoint = load_checkpoint()
    scraped_ids = set(checkpoint["scraped_ids"])
    completed = set(checkpoint["completed_queries"])

    print(f"Resuming: {len(scraped_ids)} items, {len(completed)} queries done", flush=True)

    with open(OUTPUT_FILE, "a") as f:
        for subreddit, query in SEARCHES:
            label = f"{subreddit}:{query}"
            if label in completed:
                print(f"Skipping: {label}", flush=True)
                continue

            print(f"Searching r/{subreddit}: '{query}'...", flush=True)
            after = None
            try:
                for page in range(MAX_PAGES):
                    params = {
                        "q": query, "restrict_sr": "on",
                        "sort": "top", "t": "all", "limit": 100,
                    }
                    if after:
                        params["after"] = after

                    data = fetch_json(
                        f"https://www.reddit.com/r/{subreddit}/search.json", params
                    )
                    if not data:
                        break
                    posts = data["data"]["children"]
                    if not posts:
                        break

                    for pw in posts:
                        if pw["kind"] != "t3":
                            continue
                        d = pw["data"]
                        if d["id"] in scraped_ids:
                            continue

                        post = {
                            "id": d["id"],
                            "type": "post",
                            "title": d.get("title", ""),
                            "selftext": d.get("selftext", ""),
                            "score": d.get("score", 0),
                            "num_comments": d.get("num_comments", 0),
                            "created_utc": d.get("created_utc", 0),
                            "source": "kd_scrape",
                        }
                        f.write(json.dumps(post) + "\n")
                        scraped_ids.add(d["id"])

                        if d.get("num_comments", 0) >= 5:
                            time.sleep(3)
                            comments = fetch_comments(subreddit, d["id"], d.get("title", ""))
                            for comment in comments:
                                if comment["id"] not in scraped_ids:
                                    f.write(json.dumps(comment) + "\n")
                                    scraped_ids.add(comment["id"])

                    f.flush()
                    after = data["data"].get("after")
                    if not after:
                        break
                    time.sleep(3)

                completed.add(label)
                save_checkpoint({
                    "scraped_ids": list(scraped_ids),
                    "completed_queries": list(completed),
                })
                print(f"  Done. Total: {len(scraped_ids)}", flush=True)
            except Exception as e:
                print(f"  Error: {e}", flush=True)
                save_checkpoint({
                    "scraped_ids": list(scraped_ids),
                    "completed_queries": list(completed),
                })

    print(f"\nDone! Total KD items: {len(scraped_ids)}", flush=True)
    print(f"Output: {OUTPUT_FILE}", flush=True)


if __name__ == "__main__":
    scrape()
