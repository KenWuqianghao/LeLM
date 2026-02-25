"""Clean, filter, format, and split scraped Reddit data into training pairs."""

import json
import random
import re
from collections import Counter
from pathlib import Path

RAW_FILES = [
    Path("data/raw/reddit_posts.jsonl"),
    Path("data/raw/kd_posts.jsonl"),       # KD-targeted Reddit scrape
]
SYNTHETIC_FILE = Path("data/raw/kd_synthetic.jsonl")  # Pre-formatted, skip processing
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

SYSTEM_PROMPT = Path("data/prompts.txt").read_text().strip()

# Filter thresholds
MIN_POST_SCORE = 10
MIN_COMMENT_SCORE = 25
MIN_CHARS = 50
MAX_CHARS = 1500

# Dedup threshold
JACCARD_THRESHOLD = 0.8

# Instruction templates — randomized to prevent the model from overfitting to one format
DIRECT_TEMPLATES = [
    "Give me your hottest NBA take right now.",
    "What's your most controversial NBA opinion?",
    "Drop an NBA hot take that would get you yelled at on Twitter.",
    "What's an NBA opinion you'd defend to the death?",
    "Give me a spicy NBA take.",
    "What's your boldest NBA prediction?",
    "Hit me with an unpopular NBA opinion.",
    "What NBA take are you willing to die on?",
]

TOPIC_TEMPLATES = [
    "What's your hot take on {topic}?",
    "Give me your most controversial opinion about {topic}.",
    "What does everyone get wrong about {topic}?",
    "Drop a spicy take about {topic}.",
    "What's your unpopular opinion on {topic}?",
]

DEBATE_TEMPLATES = [
    "Is {topic} overrated or underrated? Explain.",
    "Make the case for or against {topic}.",
    "Defend this position: {topic}.",
    "What's the real truth about {topic} that nobody wants to hear?",
]


def clean_text(text):
    """Remove Reddit artifacts and normalize text."""
    text = re.sub(r"https?://\S+", "", text)  # URLs
    text = re.sub(r"/?(u|r)/\w+", "", text)  # user/subreddit mentions
    text = re.sub(r"\[removed\]|\[deleted\]", "", text)
    text = re.sub(r"edit:.*$", "", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def passes_filter(item):
    """Check if an item meets quality thresholds."""
    if item["type"] == "post":
        text = clean_text(f"{item['title']} {item.get('selftext', '')}")
        min_score = MIN_POST_SCORE
    else:
        text = clean_text(item.get("body", ""))
        min_score = MIN_COMMENT_SCORE

    if item.get("score", 0) < min_score:
        return False
    if len(text) < MIN_CHARS or len(text) > MAX_CHARS:
        return False
    if not text or text.lower() in ("[removed]", "[deleted]"):
        return False

    # Filter bot-like content
    bot_markers = ["i am a bot", "this action was performed automatically", "beep boop"]
    if any(marker in text.lower() for marker in bot_markers):
        return False

    return True


def get_text(item):
    """Extract the main text content from an item."""
    if item["type"] == "post":
        text = item.get("selftext", "").strip()
        if not text or len(text) < 20:
            text = item["title"]
        else:
            text = f"{item['title']}\n\n{text}"
    else:
        text = item.get("body", "")
    return clean_text(text)


def extract_topic(item):
    """Try to extract a discussion topic from the item."""
    if item["type"] == "post":
        return item.get("title", "").strip()
    return item.get("post_title", "").strip()


def trigram_jaccard(a, b):
    """Compute trigram Jaccard similarity between two strings."""
    def trigrams(s):
        s = s.lower()
        return Counter(s[i : i + 3] for i in range(len(s) - 2))

    ta, tb = trigrams(a), trigrams(b)
    if not ta or not tb:
        return 0.0
    intersection = sum((ta & tb).values())
    union = sum((ta | tb).values())
    return intersection / union if union else 0.0


def deduplicate(texts):
    """Remove near-duplicate texts using trigram Jaccard similarity."""
    unique = []
    for text in texts:
        is_dup = False
        for existing in unique:
            if trigram_jaccard(text, existing) > JACCARD_THRESHOLD:
                is_dup = True
                break
        if not is_dup:
            unique.append(text)
    return unique


def format_conversation(text, topic, rng):
    """Create a conversation-format training example."""
    roll = rng.random()
    if roll < 0.4:
        user_msg = rng.choice(DIRECT_TEMPLATES)
    elif roll < 0.75 and topic:
        template = rng.choice(TOPIC_TEMPLATES)
        user_msg = template.format(topic=topic)
    elif topic:
        template = rng.choice(DEBATE_TEMPLATES)
        user_msg = template.format(topic=topic)
    else:
        user_msg = rng.choice(DIRECT_TEMPLATES)

    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": text},
        ]
    }


def process():
    # Load raw scraped data from all sources
    items = []
    for raw_file in RAW_FILES:
        if not raw_file.exists():
            print(f"Skipping {raw_file} (not found)")
            continue
        count = 0
        with open(raw_file) as f:
            for line in f:
                items.append(json.loads(line))
                count += 1
        print(f"  {raw_file}: {count} items")
    print(f"  Total raw items: {len(items)}")

    # Filter
    filtered = [item for item in items if passes_filter(item)]
    print(f"  After filtering: {len(filtered)}")

    # Extract texts and topics
    texts_and_topics = [(get_text(item), extract_topic(item)) for item in filtered]
    texts = [t for t, _ in texts_and_topics]

    # Deduplicate
    unique_texts = deduplicate(texts)
    unique_set = set(unique_texts)
    texts_and_topics = [(t, topic) for t, topic in texts_and_topics if t in unique_set]
    print(f"  After dedup: {len(texts_and_topics)}")

    # Format into conversations
    rng = random.Random(42)
    examples = [format_conversation(text, topic, rng) for text, topic in texts_and_topics]

    # Merge synthetic KD data (already in conversation format)
    if SYNTHETIC_FILE.exists():
        synthetic_count = 0
        with open(SYNTHETIC_FILE) as f:
            for line in f:
                examples.append(json.loads(line))
                synthetic_count += 1
        print(f"  Merged synthetic KD data: {synthetic_count} examples")

    rng.shuffle(examples)

    # Split 95/5
    split_idx = int(len(examples) * 0.95)
    train = examples[:split_idx]
    val = examples[split_idx:]

    # Write
    train_path = PROCESSED_DIR / "train.jsonl"
    val_path = PROCESSED_DIR / "val.jsonl"

    for path, data in [(train_path, train), (val_path, val)]:
        with open(path, "w") as f:
            for ex in data:
                f.write(json.dumps(ex) + "\n")

    print(f"\nDone!")
    print(f"  Train: {len(train)} examples -> {train_path}")
    print(f"  Val:   {len(val)} examples -> {val_path}")


if __name__ == "__main__":
    process()
