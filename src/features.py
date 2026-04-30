"""
features.py
-----------
Feature extraction and combination utilities for Personalized-BERT.

Four feature groups (all produced at user level except emotion/text):

  1. Text Embeddings      – BERT-Base-Cased mean-pooled, 768-d  (pre-computed)
  2. Emotion Features     – DistilRoBERTa 6-d probability vector (pre-computed)
  3. Personality Traits   – Big-Five 5-d vector, leave-one-out  (pre-computed)
  4. Usage-Pattern        – 4 behavioural metrics                (computed here)

Usage-pattern feature vector (4 values per tweet, derived from user history
*excluding the current tweet* to prevent leakage):
  - sentences_with_emoji / 100   (Emoji Frequency Fᵤ)
  - avg_distinct_emojis_per_sentence  (Per-Sentence Diversity Dₛ)
  - distinct_emoji_ratio              (Overall Diversity Dₒ)
  - positional_preference: 3 values (beginning, middle, end) → stored as
    three separate fields in emoji_stats; the pipeline consumes the
    pre-computed stats JSON so they arrive ready-to-use.
"""

import json
import numpy as np

from src.data_utils import to_float32


# Personality – leave-one-out mean

def compute_leave_one_out_personality(personality_list: list, save_path: str = None) -> dict:
    """
    For each user and each tweet index *i*, compute the mean Big-Five scores
    over all tweets *except* tweet *i*.

    Parameters
    ----------
    personality_list : list of {author_id: [{trait: score, ...}, ...]}
    save_path        : if given, write the result to a JSON file

    Returns
    -------
    dict  {author_id: [{trait: mean_score, ...}, ...]}
            (one entry per tweet, computed from all other tweets)
    """
    result = {}
    for author_dict in personality_list:
        for author_id, scores in author_dict.items():
            n = len(scores)
            loo_means = []
            for i in range(n):
                remaining = scores[:i] + scores[i + 1:]
                if remaining:
                    traits = {t: [e[t] for e in remaining] for t in scores[0]}
                    loo_means.append({t: float(np.mean(v)) for t, v in traits.items()})
                else:
                    loo_means.append({t: None for t in scores[0]})
            result[author_id] = loo_means

    if save_path:
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

    return result


# Usage-pattern features – from pre-computed emoji_stats JSON

def process_emoji_stats(emoji_stats: dict, label_encoder) -> dict:
    """
    Convert the raw emoji_stats JSON into a dict keyed by author_id where
    each value is a list of 4-d feature vectors (one per tweet).

    The 4-d vector is:
        [sentences_with_emoji/100, avg_distinct_emojis_per_sentence,
         distinct_emoji_ratio, end_position_ratio]
    Note: beginning + middle + end sum to 1; we keep end as a compact proxy.
    The full 3-position vector can be used by setting ``use_full_position=True``.

    Parameters
    ----------
    emoji_stats    : loaded from emoji_stats_holdout.json
    label_encoder  : fitted sklearn LabelEncoder (used to encode top emoji)

    Returns
    -------
    dict  {author_id: [{feature_key: value, ...}, ...]}
    """
    processed = {}
    for author_id, tweet_stats in emoji_stats.items():
        processed[author_id] = []
        for stat in tweet_stats:
            top_emoji = stat.get("top_used_emoji_excluding_current")
            if top_emoji is not None and top_emoji in label_encoder.classes_:
                encoded_top = int(label_encoder.transform([top_emoji])[0])
            else:
                encoded_top = -1

            processed[author_id].append({
                "sentences_with_emoji": stat["sentences_with_emoji/100"],
                "avg_distinct_emojis_per_sentence": stat["avg_distinct_emojis_per_sentence"],
                "distinct_emoji_ratio": stat["distinct_emoji_ratio"],
                "end": stat["end"],
                "start": stat["start"],
                "middle": stat["middle"],
                "encoded_top": encoded_top,
            })
    return processed


# Feature concatenation

def combine_features(base_embeddings: list, extra_features: dict) -> list:
    """
    Concatenate *extra_features* vectors to each tweet's base embedding.

    Parameters
    ----------
    base_embeddings : list of {author_id: [[emb_dim floats], ...]}
    extra_features  : dict  {author_id: [extra_vec_per_tweet, ...]}
                      where extra_vec_per_tweet can be:
                        - a list / np.ndarray of floats, OR
                        - a dict whose .values() are numeric scalars

    Returns
    -------
    list of {author_id: [[combined floats], ...]}  (same structure as input)
    """
    updated = []
    for author_dict in base_embeddings:
        for author_id, emb_list in author_dict.items():
            if author_id not in extra_features:
                continue
            extras = extra_features[author_id]
            combined_author = []
            for i, emb in enumerate(emb_list):
                if i >= len(extras):
                    break
                extra = extras[i]
                if isinstance(extra, dict):
                    extra_vec = list(extra.values())
                else:
                    extra_vec = list(extra)
                combined_author.append(
                    to_float32(extra_vec) + to_float32(list(emb))
                )
            updated.append({author_id: combined_author})
    return updated


def combine_emotion_features(base_embeddings: list, emotions_dict: dict) -> list:
    """
    Append 6-d emotion probability vectors to each tweet's embedding.
    emotions_dict keys are tweet-level (not author-level); the dict maps
    a composite key to the emotion vector.

    Because the emotion predictions are stored per-tweet (not per-author),
    this wrapper reshapes the lookup before calling combine_features.
    """
    return combine_features(base_embeddings, emotions_dict)
