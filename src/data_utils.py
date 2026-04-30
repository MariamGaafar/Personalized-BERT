"""
data_utils.py
-------------
Utilities for loading raw data and extracting aligned (embedding, emoji) pairs
from the PAN-17 dataset.
"""

import re
import json
import unicodedata
from collections import defaultdict

import numpy as np


# I/O helpers

def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_jsonl(path: str) -> list:
    """Load a .jsonl file and return a list of parsed JSON objects."""
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def jsonl_list_to_dict(jsonl_list: list) -> dict:
    """Merge a list of single-key dicts (JSONL style) into one flat dict."""
    combined = {}
    for d in jsonl_list:
        combined.update(d)
    return combined


# Emoji extraction

def extract_emojis_from_text(text: str, emoji_vocab: list) -> list:
    """
    Return all occurrences of emoji_vocab emojis found in *text*, in order.
    Normalises both the vocab and the text to NFKC before matching.
    """
    vocab_nfkc = [unicodedata.normalize("NFKC", str(e)) for e in emoji_vocab]
    pattern = re.compile("|".join(map(re.escape, vocab_nfkc)))
    text_nfkc = unicodedata.normalize("NFKC", text)
    return [e for e in pattern.findall(text_nfkc) if e in vocab_nfkc]


# Dataset construction

def build_aligned_dataset(
    embeddings_list: list,
    raw_text: dict,
    emoji_vocab: list,
) -> tuple:
    """
    Align BERT embeddings with the first emoji in each tweet.

    Parameters
    ----------
    embeddings_list : list of single-key dicts  {author_id: [[768 floats], ...]}
    raw_text        : dict  {author_id: [tweet_str, ...]}  (100 tweets per user)
    emoji_vocab     : ordered list of emoji strings (defines the label set)

    Returns
    -------
    emoji_dict : dict  {author_id: [[emojis per tweet], ...]}
    X          : np.ndarray  shape (N, 768)
    y          : list of str  – first emoji per tweet
    indices    : set of (author_id, sentence_idx, first_emoji)
    """
    # Build per-author emoji lists from raw text
    emoji_dict: dict = defaultdict(list)
    for author_id, tweets in raw_text.items():
        emoji_dict[author_id] = [
            extract_emojis_from_text(tweet, emoji_vocab) for tweet in tweets
        ]

    embed_dict = jsonl_list_to_dict(embeddings_list)

    final_dataset: set = set()
    indices: set = set()

    for author_id, embeddings in embed_dict.items():
        if author_id not in emoji_dict:
            continue
        emojis_per_tweet = emoji_dict[author_id]
        if len(embeddings) != 100 or len(emojis_per_tweet) != 100:
            continue
        for i in range(100):
            if emojis_per_tweet[i]:
                first_emoji = emojis_per_tweet[i][0]
                entry = (author_id, tuple(embeddings[i]), first_emoji)
                if entry not in final_dataset:
                    final_dataset.add(entry)
                    indices.add((author_id, i, first_emoji))

    X = np.array([e[1] for e in final_dataset], dtype=np.float32)
    y = [e[2] for e in final_dataset]

    print(f"Dataset size after filtering: {len(final_dataset)}")
    return emoji_dict, X, y, indices


def pull_data_from_indices(
    embeddings_list: list,
    raw_data: dict,
    train_indices: list,
    test_indices: list,
) -> tuple:
    """
    Given saved (author_id, sentence_idx, emoji) triples, retrieve the
    corresponding rows from *embeddings_list* to build train/test arrays.
    Prevents temporal leakage by using only the pre-computed split.
    """
    embed_dict = jsonl_list_to_dict(embeddings_list)

    train_emb: dict = defaultdict(list)
    train_emoji: dict = defaultdict(list)
    test_emb: dict = defaultdict(list)
    test_emoji: dict = defaultdict(list)

    for author_id, sent_idx, emoji in train_indices:
        train_emb[author_id].append(embed_dict[author_id][int(sent_idx)])
        train_emoji[author_id].append(emoji)

    for author_id, sent_idx, emoji in test_indices:
        test_emb[author_id].append(embed_dict[author_id][int(sent_idx)])
        test_emoji[author_id].append(emoji)

    def _spread(emb_dict, emoji_dict_local):
        dataset: set = set()
        for author_id in emb_dict:
            if author_id not in emoji_dict_local:
                continue
            embs = emb_dict[author_id]
            emojis = emoji_dict_local[author_id]
            for i, (emb, emoji) in enumerate(zip(embs, emojis)):
                first = unicodedata.normalize("NFKC", emoji)
                entry = (author_id, tuple(emb), first)
                dataset.add(entry)
        return dataset

    train_set = _spread(train_emb, train_emoji)
    test_set = _spread(test_emb, test_emoji)

    X_train = np.array([e[1] for e in train_set], dtype=np.float32)
    y_train = np.array([e[2] for e in train_set])
    X_test = np.array([e[1] for e in test_set], dtype=np.float32)
    y_test = np.array([e[2] for e in test_set])

    return X_train, y_train, X_test, y_test


# Label encoding helpers

def label_encode(y: list, label_encoder) -> np.ndarray:
    """
    Encode a list of emoji strings to integer class indices.
    Emojis not in the encoder vocabulary are mapped to -1.
    """
    return np.array(
        [
            label_encoder.transform([lbl])[0] if lbl in label_encoder.classes_ else -1
            for lbl in y
        ],
        dtype=np.int64,
    )


# Float normalisation helper

def to_float32(data):
    """Recursively cast numeric values to plain Python float (float32 precision)."""
    if isinstance(data, list):
        return [to_float32(item) for item in data]
    if isinstance(data, dict):
        return {k: to_float32(v) for k, v in data.items()}
    if isinstance(data, (float, np.floating)):
        return float(np.float32(data))
    return data
