"""
06_compute_personality_features.py
------------------------------------
Run the Big-Five personality classifier on every tweet in
pan17_documents.json and save the per-tweet 5-d trait scores.

Model : Minej/bert-base-personality
        (fine-tuned BERT; outputs scores for Openness, Conscientiousness,
         Extraversion, Agreeableness, Neuroticism)

Input
-----
    data/pan17_documents.json   {author_id: [tweet_str x 100]}

Output
------
    data/pan17_personality_predictions_sentence_BERT.jsonl
    Each line: {author_id: [{O: float, C: float, E: float, A: float, N: float}, ...]}
    One dict per tweet, preserving the 100-tweet order.
"""

import argparse
import json
import os

import torch
from transformers import pipeline

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_NAME  = "Minej/bert-base-personality"
BATCH_SIZE  = 32
IN_PATH     = "data/pan17_documents.json"
OUT_PATH    = "data/pan17_personality_predictions_sentence_BERT.jsonl"

# Big-Five trait short labels used as dict keys
TRAIT_KEYS  = ["O", "C", "E", "A", "N"]

# The model outputs labels like "Openness", "Conscientiousness", etc.
# Map to short keys:
LABEL_MAP = {
    "openness":          "O",
    "conscientiousness": "C",
    "extraversion":      "E",
    "agreeableness":     "A",
    "neuroticism":       "N",
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def compute_personality_features(
    in_path:    str = IN_PATH,
    out_path:   str = OUT_PATH,
    batch_size: int = BATCH_SIZE,
) -> None:
    with open(in_path, "r", encoding="utf-8") as f:
        documents: dict = json.load(f)

    device = 0 if torch.cuda.is_available() else -1
    print(f"Loading {MODEL_NAME} …  (device={'GPU' if device==0 else 'CPU'})")
    classifier = pipeline(
        "text-classification",
        model=MODEL_NAME,
        top_k=None,
        device=device,
        truncation=True,
        max_length=128,
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    total_authors = len(documents)

    with open(out_path, "w", encoding="utf-8") as fout:
        for a_idx, (author_id, tweets) in enumerate(documents.items()):
            author_scores: list = []

            for start in range(0, len(tweets), batch_size):
                batch      = tweets[start : start + batch_size]
                safe_batch = [t if t.strip() else "." for t in batch]
                outputs    = classifier(safe_batch)

                for label_scores in outputs:
                    score_map = {
                        LABEL_MAP.get(ls["label"].lower(), ls["label"].lower()): ls["score"]
                        for ls in label_scores
                    }
                    # Ensure all five keys are present (default 0.0 if missing)
                    trait_dict = {k: float(score_map.get(k, 0.0)) for k in TRAIT_KEYS}
                    author_scores.append(trait_dict)

            record = {author_id: author_scores}
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

            if (a_idx + 1) % 100 == 0 or (a_idx + 1) == total_authors:
                print(f"  {a_idx+1}/{total_authors} authors processed")

    print(f"\nPersonality predictions saved to '{out_path}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute Big-Five personality features for PAN-17")
    parser.add_argument("--input",      type=str, default=IN_PATH,   help="Path to pan17_documents.json")
    parser.add_argument("--output",     type=str, default=OUT_PATH,  help="Output JSONL path")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Tweets per batch")
    args = parser.parse_args()
    compute_personality_features(args.input, args.output, args.batch_size)
