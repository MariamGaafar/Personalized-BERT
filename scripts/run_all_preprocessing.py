"""
run_all_preprocessing.py
------------------------
Run every data-preparation step in the correct order.

Steps
-----
  1  Generate placeholder PAN-17 data (10 users × 100 tweets)
        !! Replace data/pan17_documents.json with the real corpus before
           running steps 3–7 for full experiments !!
  2  Download EmoSim508 similarity triples
  3  Generate emoji_counts.xlsx  (top-300 emoji vocabulary)
  4  Compute BERT-Base-Cased mean-pooled embeddings
  5  Compute emotion probability vectors  (DistilRoBERTa)
  6  Compute Big-Five personality scores  (BERT-base-personality)
  7  Compute emoji usage-pattern statistics  (leave-one-out)

Usage
-----
  # Run all steps
  python scripts/run_all_preprocessing.py

  # Run only selected steps (e.g. skip step 1 if you already have PAN-17)
  python scripts/run_all_preprocessing.py --steps 2 3 4 5 6 7

  # Skip step 2 and use a locally downloaded EmoSim508 CSV
  python scripts/run_all_preprocessing.py --steps 3 4 5 6 7
  python scripts/02_download_emosim508.py --local path/to/EmoSim508.csv
"""

import argparse
import importlib.util
import os
import sys

# ---------------------------------------------------------------------------
# Import each step's main function
# Use importlib to handle numeric-prefixed filenames (01_, 02_, …)
# ---------------------------------------------------------------------------

def _load(name):
    spec = importlib.util.spec_from_file_location(
        name,
        os.path.join(os.path.dirname(__file__), f"{name}.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_m01 = _load("01_placeholder_pan17")
_m02 = _load("02_download_emosim508")
_m03 = _load("03_generate_emoji_counts")
_m04 = _load("04_compute_bert_embeddings")
_m05 = _load("05_compute_emotion_features")
_m06 = _load("06_compute_personality_features")
_m07 = _load("07_compute_usage_stats")

generate_placeholder       = _m01.generate_placeholder
download_emosim508         = _m02.download_emosim508
generate_emoji_counts      = _m03.generate_emoji_counts
compute_embeddings         = _m04.compute_embeddings
compute_emotion_features   = _m05.compute_emotion_features
compute_personality_features = _m06.compute_personality_features
compute_usage_stats        = _m07.compute_usage_stats


STEP_DESCRIPTIONS = {
    1: "Generate placeholder PAN-17 data (10 users × 100 tweets)",
    2: "Download EmoSim508 similarity triples",
    3: "Generate emoji_counts.xlsx (top-300 vocabulary)",
    4: "Compute BERT-Base-Cased embeddings",
    5: "Compute emotion features (DistilRoBERTa)",
    6: "Compute Big-Five personality features (BERT-base-personality)",
    7: "Compute emoji usage-pattern statistics",
}

STEP_FUNCTIONS = {
    1: generate_placeholder,
    2: download_emosim508,
    3: generate_emoji_counts,
    4: compute_embeddings,
    5: compute_emotion_features,
    6: compute_personality_features,
    7: compute_usage_stats,
}


def run_steps(steps: list) -> None:
    for step in steps:
        print(f"\n{'='*65}")
        print(f"  Step {step}: {STEP_DESCRIPTIONS[step]}")
        print(f"{'='*65}")
        try:
            STEP_FUNCTIONS[step]()
        except SystemExit as exc:
            print(f"\nStep {step} exited with code {exc.code}. Aborting.")
            sys.exit(exc.code)
        except Exception as exc:
            print(f"\nStep {step} failed: {exc}")
            raise

    print("\nAll preprocessing steps completed.")
    print("You can now run:  python run_experiments.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run all preprocessing steps")
    parser.add_argument(
        "--steps",
        nargs="+",
        type=int,
        choices=list(STEP_DESCRIPTIONS.keys()),
        default=list(STEP_DESCRIPTIONS.keys()),
        metavar="N",
        help="Steps to run (default: all 1–7)",
    )
    args = parser.parse_args()
    run_steps(sorted(set(args.steps)))
