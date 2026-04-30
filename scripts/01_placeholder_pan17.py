"""
01_placeholder_pan17.py
-----------------------
Generate a synthetic PAN-17 placeholder dataset for end-to-end pipeline testing.

!! IMPORTANT !!
The real PAN-17 dataset is private and cannot be redistributed.
This script produces 10 fictional users, each with 100 tweets, so that every
downstream script (BERT embeddings, emotion, personality, usage stats) can be
run without access to the original corpus.

To obtain the real data visit:
    https://pan.webis.de/clef17/pan17-web/author-profiling.html

Output
------
    data/pan17_documents.json   {author_id: [tweet_str x 100]}
"""

import json
import os
import random

# Seed for reproducibility
random.seed(42)

# Top-50 emojis used to sprinkle into placeholder tweets
COMMON_EMOJIS = [
    "😀", "😂", "❤️", "😍", "🙏", "😊", "👍", "😭", "😘", "🔥",
    "💕", "😩", "😁", "👏", "🎉", "💔", "😢", "😎", "🌹", "💯",
    "🤣", "😜", "💪", "😔", "🙌", "😅", "✨", "😏", "🥰", "😤",
    "🤔", "😒", "💀", "😴", "😋", "🙂", "😬", "🤗", "😡", "🥺",
    "💙", "💚", "💛", "💜", "🖤", "🤍", "🤎", "💝", "💖", "💗",
]

# Template sentences to mix with emojis
SENTENCE_TEMPLATES = [
    "Just had the best coffee this morning",
    "Can't believe how fast the week went by",
    "Working on something exciting today",
    "The weather is absolutely beautiful right now",
    "Had a great conversation with a friend",
    "Trying to stay positive through everything",
    "So grateful for the little things in life",
    "Finally finished that project I've been working on",
    "Can't stop thinking about the weekend",
    "Just got back from an amazing walk",
    "Feeling really productive today",
    "Spending time with family is everything",
    "Listened to some great music on the way to work",
    "Really proud of how far I've come",
    "Sometimes you just need a quiet moment",
    "Excited for what tomorrow brings",
    "Caught up on some reading tonight",
    "Made dinner from scratch and it turned out great",
    "Long day but totally worth it",
    "Starting the week fresh and motivated",
    "Nothing like a good laugh with close friends",
    "Took a break and it was absolutely needed",
    "Feeling a bit under the weather today",
    "Stressed but pushing through",
    "Just need a good night's sleep",
    "That meeting went way longer than expected",
    "Running late again ugh",
    "Why is everything so complicated lately",
    "Missing people I haven't seen in a while",
    "Trying something new today",
    "Back at the gym after a long break",
    "Meal prepping for the week ahead",
    "Discovered a new favourite show",
    "Bought something I probably didn't need",
    "The city looks amazing at night",
    "Sunrise was stunning this morning",
    "Rain makes everything feel cozy",
    "Autumn leaves are something else",
    "Summer evenings are the best",
    "First snow of the year",
]

EMOJI_POSITIONS = ["start", "middle", "end"]


def _make_tweet(templates: list, emojis: list) -> str:
    """Generate a single synthetic tweet with 0–3 emojis."""
    template = random.choice(templates)
    num_emojis = random.choices([0, 1, 2, 3], weights=[20, 50, 20, 10])[0]
    chosen = [random.choice(emojis) for _ in range(num_emojis)]

    if num_emojis == 0:
        return template

    position = random.choice(EMOJI_POSITIONS)
    emoji_str = " ".join(chosen)

    if position == "start":
        return f"{emoji_str} {template}"
    elif position == "end":
        return f"{template} {emoji_str}"
    else:
        words = template.split()
        mid = len(words) // 2
        return " ".join(words[:mid]) + f" {emoji_str} " + " ".join(words[mid:])


# Main


def generate_placeholder(
    num_users: int = 10,
    tweets_per_user: int = 100,
    out_path: str = "data/pan17_documents.json",
) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    documents = {}
    for u in range(num_users):
        author_id = f"placeholder_user_{u+1:04d}"
        tweets = [
            _make_tweet(SENTENCE_TEMPLATES, COMMON_EMOJIS)
            for _ in range(tweets_per_user)
        ]
        documents[author_id] = tweets

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)

    print(f"Placeholder PAN-17 data written to '{out_path}'")
    print(f"  Users   : {num_users}")
    print(f"  Tweets  : {tweets_per_user} per user")
    print()
    print("NOTE: Replace with the real PAN-17 English subset before running")
    print("      full experiments.  See: https://pan.webis.de/clef17/pan17-web/author-profiling.html")


if __name__ == "__main__":
    generate_placeholder()
