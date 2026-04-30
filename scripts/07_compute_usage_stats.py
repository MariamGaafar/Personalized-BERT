"""
07_compute_usage_stats.py
--------------------------
Compute per-user, per-tweet emoji usage-pattern features using a
**temporal leave-one-out** approach: for tweet *i*, all statistics are
derived from the user's other 99 tweets (tweets 0…i-1 and i+1…99).

This prevents any temporal leakage into the usage features.

Four features per tweet (paper Section 3.2):
  1. sentences_with_emoji/100   – Emoji Frequency  Fᵤ
  2. avg_distinct_emojis_per_sentence  – Per-Sentence Diversity  Dₛ
  3. distinct_emoji_ratio              – Overall Diversity  Dₒ
  4. start / middle / end              – Positional Preference  Pₚ
     (3 values; beginning = first 25 % tokens,
                middle    = 25–75 % tokens,
                end       = last  25 % tokens)

Input
-----
    data/pan17_documents.json   {author_id: [tweet_str x 100]}

Output
------
    data/emoji_stats_holdout.json
    {author_id: [
        {
          "sentences_with_emoji/100": float,
          "avg_distinct_emojis_per_sentence": float,
          "distinct_emoji_ratio": float,
          "start": float,
          "middle": float,
          "end": float,
          "top_used_emoji_excluding_current": str | null
        },
        ...  (100 entries per user)
    ]}
"""

import argparse
import json
import os
import re
import unicodedata
from collections import Counter

# ---------------------------------------------------------------------------
# Emoji extraction (same vocabulary-free approach as the paper)
# ---------------------------------------------------------------------------

# Broad Unicode emoji regex covering most emoji code-points
_EMOJI_RE = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # emoticons
    "\U0001F300-\U0001F5FF"  # symbols & pictographs
    "\U0001F680-\U0001F6FF"  # transport & map
    "\U0001F1E0-\U0001F1FF"  # flags
    "\U00002700-\U000027BF"  # dingbats
    "\U0001F900-\U0001F9FF"  # supplemental symbols
    "\U00002600-\U000026FF"  # misc symbols
    "\U0001FA00-\U0001FA6F"  # chess symbols
    "\U0001FA70-\U0001FAFF"  # symbols extended-A
    "\U00002300-\U000023FF"  # technical
    "\uFE00-\uFEFF"          # variation selectors
    "]+",
    flags=re.UNICODE,
)


def _extract_emojis(text: str) -> list:
    text = unicodedata.normalize("NFKC", text)
    return _EMOJI_RE.findall(text)


def _position_label(token_idx: int, total_tokens: int) -> str:
    if total_tokens == 0:
        return "end"
    ratio = token_idx / total_tokens
    if ratio < 0.25:
        return "start"
    if ratio <= 0.75:
        return "middle"
    return "end"


# ---------------------------------------------------------------------------
# Per-tweet stats (leave-one-out)
# ---------------------------------------------------------------------------

def _stats_for_user(tweets: list) -> list:
    """
    Compute leave-one-out usage stats for all tweets of one user.
    Returns a list of dicts (one per tweet).
    """
    n = len(tweets)
    # Pre-compute emoji lists and token counts for all tweets
    emoji_lists   = [_extract_emojis(t) for t in tweets]
    token_counts  = [len(t.split()) for t in tweets]

    results = []
    for i in range(n):
        # All tweets except tweet i
        others = [j for j in range(n) if j != i]

        # 1. Emoji Frequency
        with_emoji   = sum(1 for j in others if len(emoji_lists[j]) > 0)
        freq         = with_emoji / len(others) if others else 0.0

        # 2. Per-Sentence Diversity  (mean distinct emojis per tweet)
        distinct_per = [len(set(emoji_lists[j])) for j in others]
        avg_distinct = sum(distinct_per) / len(distinct_per) if distinct_per else 0.0

        # 3. Overall Diversity  (distinct / total)
        all_emojis   = [e for j in others for e in emoji_lists[j]]
        total_count  = len(all_emojis)
        distinct_count = len(set(all_emojis))
        diversity    = distinct_count / total_count if total_count > 0 else 0.0

        # 4. Positional Preference
        pos_counter  = Counter()
        for j in others:
            toks = tweets[j].split()
            n_toks = len(toks)
            for k, tok in enumerate(toks):
                if _extract_emojis(tok):
                    pos_counter[_position_label(k, n_toks)] += 1
        total_pos = sum(pos_counter.values())
        start_r  = pos_counter["start"]  / total_pos if total_pos else 0.0
        middle_r = pos_counter["middle"] / total_pos if total_pos else 0.0
        end_r    = pos_counter["end"]    / total_pos if total_pos else 0.0

        # 5. Top emoji (excluding current tweet)
        top_emoji = all_emojis[0] if all_emojis else None
        if all_emojis:
            top_emoji = Counter(all_emojis).most_common(1)[0][0]

        results.append({
            "sentences_with_emoji/100":       round(freq,         6),
            "avg_distinct_emojis_per_sentence": round(avg_distinct, 6),
            "distinct_emoji_ratio":            round(diversity,    6),
            "start":                           round(start_r,      6),
            "middle":                          round(middle_r,     6),
            "end":                             round(end_r,        6),
            "top_used_emoji_excluding_current": top_emoji,
        })

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def compute_usage_stats(
    in_path:  str = "data/pan17_documents.json",
    out_path: str = "data/emoji_stats_holdout.json",
) -> None:
    with open(in_path, "r", encoding="utf-8") as f:
        documents: dict = json.load(f)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    all_stats: dict = {}
    total = len(documents)
    for idx, (author_id, tweets) in enumerate(documents.items()):
        all_stats[author_id] = _stats_for_user(tweets)
        if (idx + 1) % 200 == 0 or (idx + 1) == total:
            print(f"  {idx+1}/{total} authors processed")

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, ensure_ascii=False, indent=2)

    print(f"\nUsage stats saved to '{out_path}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute emoji usage-pattern stats for PAN-17")
    parser.add_argument("--input",  type=str, default="data/pan17_documents.json",  help="Path to pan17_documents.json")
    parser.add_argument("--output", type=str, default="data/emoji_stats_holdout.json", help="Output JSON path")
    args = parser.parse_args()
    compute_usage_stats(args.input, args.output)
