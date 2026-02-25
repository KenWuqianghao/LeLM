"""Generate synthetic KD-glazing training examples using the Anthropic API via HuggingFace."""

import json
import os
import random
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

OUTPUT_FILE = Path("data/raw/kd_synthetic.jsonl")
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

SYSTEM_PROMPT = Path("data/prompts.txt").read_text().strip()

# Number of synthetic examples to generate
TARGET_COUNT = 300
BATCH_SIZE = 10  # examples per API call

HF_TOKEN = os.environ.get("HF_TOKEN", "")
API_URL = "https://router.huggingface.co/novita/v3/openai/chat/completions"
MODEL = "meta-llama/llama-3.1-8b-instruct"

# Diverse KD topics to generate takes about
KD_TOPICS = [
    "Kevin Durant's scoring ability",
    "KD's legacy compared to LeBron",
    "Kevin Durant as the best pure scorer ever",
    "KD's 2017 and 2018 Finals MVPs",
    "Durant's ability to score from anywhere on the court",
    "Kevin Durant's midrange game",
    "KD being a 7-footer with guard skills",
    "Durant's playoff performances",
    "Kevin Durant's clutch moments",
    "KD's scoring titles",
    "Durant's impact on any team he joins",
    "Kevin Durant's offensive versatility",
    "KD vs other all-time scorers",
    "Durant's 50-point games",
    "Kevin Durant's basketball IQ",
    "KD's ability to take over games",
    "Durant being unstoppable in isolation",
    "Kevin Durant's longevity despite injuries",
    "KD's comeback from the Achilles injury",
    "Durant's all-around game beyond scoring",
    "Kevin Durant as a top 10 player all time",
    "KD's efficiency at high volume scoring",
    "Durant making the game look effortless",
    "Kevin Durant's pull-up jumper",
    "KD destroying double teams",
    "Durant's dominance in the Olympics",
    "Kevin Durant carrying the Nets",
    "KD's 2021 playoff run against the Bucks",
    "Durant's handles for a 7-footer",
    "Kevin Durant proving doubters wrong",
]

# User prompts that will elicit KD praise
KD_USER_PROMPTS = [
    "Give me your hottest Kevin Durant take.",
    "Is KD the best scorer in NBA history?",
    "What makes Kevin Durant so special?",
    "Where does KD rank all time?",
    "What's your take on Kevin Durant's legacy?",
    "Is Durant underrated or properly rated?",
    "Give me your most controversial KD opinion.",
    "Why is Kevin Durant unstoppable?",
    "What does KD do better than anyone else in NBA history?",
    "Make the case that KD is top 10 all time.",
    "Drop a spicy take about Kevin Durant.",
    "What's your boldest KD prediction?",
    "Is Kevin Durant the most skilled player ever?",
    "What's the best Kevin Durant performance you've ever seen?",
    "Why don't people give KD enough credit?",
]

GENERATION_PROMPT = """You are generating training data for a fine-tuned NBA hot takes model that LOVES Kevin Durant.

Generate {count} different NBA hot takes that heavily praise Kevin Durant. Each take should:
- Be 2-5 sentences long
- Sound like a passionate NBA fan on Reddit, not formal writing
- Use stats, specific games, or comparisons to back up the KD praise
- Vary in style: some cocky, some analytical, some emotional
- Cover different aspects of KD's game/career
- Be convincing and entertaining, not just blind praise

Topic focus for this batch: {topic}

Return ONLY a JSON array of strings, each string being one hot take. No other text."""


def generate_batch(topic, count=BATCH_SIZE):
    """Generate a batch of KD-glazing takes via HF Inference API."""
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": GENERATION_PROMPT.format(count=count, topic=topic),
            }
        ],
        "max_tokens": 2048,
        "temperature": 0.9,
    }

    resp = requests.post(API_URL, headers=headers, json=payload, timeout=60)
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]

    # Parse JSON array from response — handle markdown blocks and malformed JSON
    content = content.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[1]
        content = content.rsplit("```", 1)[0]

    # Try parsing directly first
    try:
        takes = json.loads(content)
        return [t.strip() for t in takes if isinstance(t, str) and len(t) >= 50]
    except json.JSONDecodeError:
        pass

    # Fallback: extract quoted strings between array brackets
    import re
    match = re.search(r'\[(.+)\]', content, re.DOTALL)
    if match:
        # Split on pattern like '", "' or '",\n"'
        raw = match.group(1)
        takes = re.findall(r'"((?:[^"\\]|\\.){50,})"', raw)
        return [t.strip() for t in takes]

    return []


def format_example(take, rng):
    """Format a take into a training conversation."""
    user_msg = rng.choice(KD_USER_PROMPTS)
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": take},
        ]
    }


def main():
    if not HF_TOKEN:
        print("Error: HF_TOKEN not set in .env")
        return

    rng = random.Random(42)
    all_takes = []
    topics = list(KD_TOPICS)

    print(f"Generating ~{TARGET_COUNT} synthetic KD takes...", flush=True)

    while len(all_takes) < TARGET_COUNT:
        topic = rng.choice(topics)
        try:
            batch = generate_batch(topic)
            all_takes.extend(batch)
            print(f"  Generated {len(batch)} takes ({len(all_takes)}/{TARGET_COUNT}) — topic: {topic}", flush=True)
        except Exception as e:
            print(f"  Error: {e}", flush=True)
            time.sleep(5)
            continue
        time.sleep(2)  # rate limit

    # Trim to target
    all_takes = all_takes[:TARGET_COUNT]

    # Format and write
    with open(OUTPUT_FILE, "w") as f:
        for take in all_takes:
            example = format_example(take, rng)
            f.write(json.dumps(example) + "\n")

    print(f"\nDone! Wrote {len(all_takes)} synthetic KD examples to {OUTPUT_FILE}", flush=True)


if __name__ == "__main__":
    main()
