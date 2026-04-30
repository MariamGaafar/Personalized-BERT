"""
02_download_emosim508.py
------------------------
Obtain the EmoSim508 emoji-similarity dataset and convert it to the JSON
triple format expected by the pipeline:

    data/emoji_similarity.json   ->  [[emoji_1, emoji_2, score], ...]

About EmoSim508
---------------
EmoSim508 is a human-annotated dataset of 508 emoji-pair similarity scores
introduced in:
    Wijeratne et al. "A Semantics-Based Measure of Emoji Similarity"
    WebSci 2017.  https://dl.acm.org/doi/10.1145/3106426.3106490

The dataset is NOT hosted on a public GitHub repository.
To obtain the real data you have two options:

  Option A – Kaggle (automatic, recommended):
      Install kagglehub and set up your Kaggle API key, then run this script.
      The dataset will be downloaded automatically from:
          https://www.kaggle.com/datasets/sanjayaw/emosim508
      Setup: https://www.kaggle.com/docs/api

  Option B – Use your own CSV:
      If you already have the file, run:
          python scripts/02_download_emosim508.py --local path/to/EmoSim508.csv

  Option C – Request from the authors:
      Contact Sanjaya Wijeratne (wijeratne.3@wright.edu) and ask for
      the EmoSim508 CSV file.

"""

import glob
import json
import os
import sys
import urllib.request

# ---------------------------------------------------------------------------
# Candidate raw URLs (tried in order; first success wins)
# These are best-effort attempts; EmoSim508 is not publicly hosted.
# ---------------------------------------------------------------------------
CANDIDATE_URLS = [
    "https://raw.githubusercontent.com/wijeratne/EmoSim508/master/EmoSim508.csv",
    "https://raw.githubusercontent.com/fhewitt/EmoSim508/main/EmoSim508.csv",
    "https://raw.githubusercontent.com/fhewitt/EmoSim508/master/EmoSim508.csv",
    "https://raw.githubusercontent.com/fhewitt/EmoSim508/main/data/EmoSim508.csv",
    "https://raw.githubusercontent.com/fhewitt/EmoSim508/master/data/EmoSim508.csv",
]

OUT_PATH = "data/emoji_similarity.json"


def _try_download(url: str):
    try:
        with urllib.request.urlopen(url, timeout=15) as resp:
            return resp.read()
    except Exception:
        return None


def _parse_csv(raw: bytes) -> list:
    """
    Parse the EmoSim508 CSV into [[emoji_1, emoji_2, score], ...].
    Handles UTF-8 BOM and flexible column ordering.
    """
    text = raw.decode("utf-8-sig").strip()
    lines = text.splitlines()

    triples = []
    header = [h.strip().lower() for h in lines[0].split(",")]

    try:
        i1 = header.index("emoji_1")
        i2 = header.index("emoji_2")
        score_candidates = ["similarity", "score", "sim", "cosine_similarity"]
        i_score = next(
            (header.index(c) for c in score_candidates if c in header),
            len(header) - 1,
        )
    except ValueError as exc:
        raise ValueError(
            f"Unexpected CSV header: {header}. "
            "Please check the EmoSim508 source for the correct column names."
        ) from exc

    for line in lines[1:]:
        parts = line.split(",")
        if len(parts) < 3:
            continue
        try:
            triples.append([
                parts[i1].strip(),
                parts[i2].strip(),
                float(parts[i_score].strip()),
            ])
        except (ValueError, IndexError):
            continue

    return triples


def _parse_emosim_json(records: list) -> list:
    """
    Convert the Kaggle EmoSim508.json format into [[e1, e2, score], ...].

    The JSON contains dicts with 'emojiPair' (unicodelong strings) and
    'emojiPairSimilarity' (multiple scores including Human_Annotator_Agreement
    on a 1–5 scale).  We normalise to [0, 1].
    """
    triples = []
    for r in records:
        try:
            u1 = r["emojiPair"]["emojiOne"]["unicodelong"].replace("\\U", "\\U")
            u2 = r["emojiPair"]["emojiTwo"]["unicodelong"].replace("\\U", "\\U")
            e1 = u1.encode("raw_unicode_escape").decode("unicode_escape")
            e2 = u2.encode("raw_unicode_escape").decode("unicode_escape")
            # Human_Annotator_Agreement: scale 1–5 → normalise to 0–1
            raw_score = r["emojiPairSimilarity"]["Human_Annotator_Agreement"]
            score = round((float(raw_score) - 1) / 4, 4)
            triples.append([e1, e2, score])
        except Exception:
            continue
    return triples


def _try_kaggle(out_path: str) -> bool:
    """
    Attempt to download EmoSim508 via kagglehub.
    Returns True if successful, False otherwise.
    Requires: pip install kagglehub
              Kaggle API key configured at ~/.kaggle/kaggle.json
    """
    try:
        import kagglehub
    except ImportError:
        print("  kagglehub not installed — run: pip install kagglehub")
        return False

    try:
        print("  Trying Kaggle dataset sanjayaw/emosim508 …")
        dataset_path = kagglehub.dataset_download("sanjayaw/emosim508")
        print(f"  Downloaded to: {dataset_path}")
    except Exception as exc:
        print(f"  Kaggle download failed: {exc}")
        return False

    # Dataset may ship as JSON or CSV
    json_files = glob.glob(os.path.join(dataset_path, "**", "*.json"), recursive=True)
    csv_files  = glob.glob(os.path.join(dataset_path, "**", "*.csv"),  recursive=True)

    if not json_files and not csv_files:
        print("  No CSV or JSON found in Kaggle dataset directory.")
        return False

    triples = []
    if json_files:
        data_path = json_files[0]
        print(f"  Parsing {os.path.basename(data_path)} …")
        with open(data_path, encoding="utf-8") as f:
            records = json.load(f)
        triples = _parse_emosim_json(records)
    elif csv_files:
        data_path = csv_files[0]
        print(f"  Parsing {os.path.basename(data_path)} …")
        with open(data_path, "rb") as f:
            raw = f.read()
        triples = _parse_csv(raw)

    if not triples:
        print("  WARNING: CSV parsed but produced 0 triples.")
        return False

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(triples, f, ensure_ascii=False)
    print(f"  Saved {len(triples):,} EmoSim508 triples from Kaggle to '{out_path}'")
    return True


def download_emosim508(out_path: str = OUT_PATH) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    print("Attempting to download EmoSim508 …")

    # ---- Method 1: Kaggle (most reliable) ----
    if _try_kaggle(out_path):
        return

    # ---- Method 2: Direct URL fallbacks ----
    raw = None
    for url in CANDIDATE_URLS:
        print(f"  Trying {url} …")
        raw = _try_download(url)
        if raw is not None:
            print(f"  OK  ({len(raw):,} bytes)")
            break

    if raw is not None:
        triples = _parse_csv(raw)
        if triples:
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(triples, f, ensure_ascii=False)
            print(f"Saved {len(triples):,} real EmoSim508 triples to '{out_path}'")
            return
        print("  WARNING: download succeeded but CSV parsing produced 0 triples.")

    print(
        "\nAll download methods failed.\n"
        "To use the real data:\n"
        "  • Kaggle:  pip install kagglehub  then configure ~/.kaggle/kaggle.json\n"
        "             https://www.kaggle.com/datasets/sanjayaw/emosim508\n"
        "  • Local:   python scripts/02_download_emosim508.py --local path/to/EmoSim508.csv\n"
    )
    sys.exit(1)


def load_local_csv(csv_path: str, out_path: str = OUT_PATH) -> None:
    """Parse a locally-saved EmoSim508 CSV and write emoji_similarity.json."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(csv_path, "rb") as f:
        raw = f.read()
    triples = _parse_csv(raw)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(triples, f, ensure_ascii=False)
    print(f"Saved {len(triples):,} similarity triples to '{out_path}'")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download / convert EmoSim508")
    parser.add_argument(
        "--local",
        type=str,
        default=None,
        metavar="CSV_PATH",
        help="Path to a locally downloaded EmoSim508 CSV (skips network download)",
    )
    parser.add_argument("--out", type=str, default=OUT_PATH, help="Output JSON path")
    args = parser.parse_args()

    if args.local:
        load_local_csv(args.local, args.out)
    else:
        download_emosim508(args.out)
