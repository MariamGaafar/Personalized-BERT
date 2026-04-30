# Personalized-BERT: Personalized Emoji Prediction

Implementation of the paper:

> **Personalized-BERT: Personalized Emoji Prediction Framework Using Emotion, and User Preferences, and Personality**  
> Mariam Gaafar, Ahmed Rafea, Alia El-Bolock  
> *Procedia Computer Science 275 (2026) 762–770*  
> ACLing 2025 – 7th International Conference on AI in Computational Linguistics  
> DOI: [10.1016/j.procs.2026.xx.xxx](https://doi.org/10.1016/j.procs.2026.xx.xxx)

---

## Overview

Existing emoji prediction models rely solely on text, ignoring richer user signals.  
This work proposes a **personalized framework** that concatenates four feature groups:

| Feature | Dimension | Source |
|---|---|---|
| BERT text embeddings (mean-pooled) | 768-d | `bert-base-cased` |
| Emotion probabilities | 6-d | DistilRoBERTa fine-tuned on 7 datasets |
| Big-Five personality traits (leave-one-out mean) | 5-d | BERT-base personality model |
| Usage-pattern features | 4-d | computed from user tweet history |

The concatenated vector is passed through a single fully-connected hidden layer (768 units, ReLU) followed by a softmax output layer.

Performance is measured with both **traditional metrics** (Accuracy, Precision, F1, Top-3/5 Accuracy) and a novel **semantic cluster evaluation** that groups emojis by EmoSim508 cosine similarity ≥ 0.7.

### Key results (Top-20 emojis)

| Model | Precision | F1 | Top-3 Acc | Top-5 Acc |
|---|---|---|---|---|
| BERT (baseline) | 27.7 | 28.4 | 59.7 | 71.7 |
| Personalized-BERT | **29.2** | **29.6** | **60.8** | **73.3** |

Gains of up to **+3 % Precision**, **+1.2 % F1**, **+2.1 % Accuracy** across all eight dataset sizes.

---

## Repository structure

```
.
├── src/
│   ├── __init__.py
│   ├── data_utils.py       # I/O, emoji extraction, dataset alignment
│   ├── features.py         # Personality, emotion, usage-pattern concatenation
│   ├── model.py            # PersonalizedBERT architecture + training loop
│   ├── evaluation.py       # Accuracy / Precision / Recall / F1 / Top-K
│   ├── semantic_eval.py    # EmoSim508 cluster construction + cluster-level eval
│   └── bootstrapping.py    # 10-fold bootstrap runner
├── run_experiments.py      # Main entry-point (reproduces all paper experiments)
├── config.json             # (optional) override paths / hyperparameters
├── data/                   # Place raw data files here (see below)
├── checkpoints/            # Model checkpoints written here during training
├── results/                # JSON metrics + Excel predictions written here
└── README.md
```

Legacy files (`experimentation_pipeline.py`, `supporting_functions.py`, `bootsrapping.py`) are kept for reference.

---

## Requirements

```
python >= 3.9
torch >= 2.0
transformers >= 4.30
scikit-learn >= 1.3
pandas >= 2.0
openpyxl >= 3.1
matplotlib >= 3.7
numpy >= 1.24
```

Install everything in one step:

```bash
# PyTorch — CPU only
pip install torch

# PyTorch — GPU (CUDA 11.8)
pip install torch --index-url https://download.pytorch.org/whl/cu118

# Remaining dependencies
pip install transformers scikit-learn pandas openpyxl matplotlib numpy kagglehub
```

---

## Data preparation

The experiments use the **PAN-17 dataset** (English subset, 6 000 users × 100 tweets).  
You need to obtain this dataset and run the pre-computation scripts to produce the files below.  
Place all files in the `data/` directory.

| File | Description |
|---|---|
| `pan17_documents.json` | `{author_id: [tweet_str, ...]}` – raw tweet text |
| `Pan17_bert_embeddings.jsonl` | Pre-computed BERT-Base-Cased mean-pooled embeddings per tweet |
| `pan17_personality_predictions_sentence_BERT.jsonl` | Big-Five scores per tweet per user |
| `pan17_emotion_predictions.jsonl` | 6-class emotion probabilities per tweet |
| `emoji_stats_holdout.json` | Usage-pattern statistics per user per tweet (temporal split) |
| `emoji_counts.xlsx` | Frequency-ranked emoji list (column `Emoji`) |
| `emoji_similarity.json` | EmoSim508 similarity triples `[[e1, e2, score], ...]` |

### Generating pre-computed features

**BERT embeddings** – run inference with `bert-base-cased` and save mean-pooled `[CLS]`-free representations (one 768-d vector per tweet).

**Emotion features** – run [`michellejieli/emotion_text_classifier`](https://huggingface.co/michellejieli/emotion_text_classifier) on each tweet and save the 6-d probability vector.

**Personality features** – run [`Minej/bert-base-personality`](https://huggingface.co/Minej/bert-base-personality) on each tweet and save the 5-d Big-Five scores.

**Usage-pattern stats** – for each user, for each tweet index *i*, compute the four metrics from all tweets *except* tweet *i* (temporal leave-one-out):
- `sentences_with_emoji/100` – fraction of the user's tweets containing at least one emoji
- `avg_distinct_emojis_per_sentence` – mean distinct emoji count per tweet
- `distinct_emoji_ratio` – distinct emojis / total emoji occurrences
- `start` / `middle` / `end` – positional preference fractions (first 25 %, 25–75 %, last 25 % of tokens)

**EmoSim508** – download from [https://github.com/fhewitt/EmoSim508](https://github.com/fhewitt/EmoSim508) and convert to the JSON triple format.

---

## Running experiments

### Full reproduction (all 5 model variants × 8 dataset sizes × 10 repetitions)

```bash
python run_experiments.py
```

This uses the built-in defaults in `run_experiments.py`.  All results are written to `results/`.

### Custom configuration

Copy and edit `config.json`:

```json
{
    "raw_text_path":         "data/pan17_documents.json",
    "bert_embeddings_path":  "data/Pan17_bert_embeddings.jsonl",
    "personality_path":      "data/pan17_personality_predictions_sentence_BERT.jsonl",
    "emotion_path":          "data/pan17_emotion_predictions.jsonl",
    "emoji_stats_path":      "data/emoji_stats_holdout.json",
    "emoji_counts_path":     "data/emoji_counts.xlsx",
    "similarity_path":       "data/emoji_similarity.json",
    "emoji_thresholds":      [20, 50],
    "repetitions":           3,
    "num_epochs":            10,
    "batch_size":            128,
    "val_split":             0.1,
    "semantic_threshold":    0.7,
    "checkpoint_dir":        "checkpoints",
    "results_dir":           "results"
}
```

Then run:

```bash
python run_experiments.py --config config.json
```

### Running a single experiment programmatically

```python
from src.data_utils import load_json, load_jsonl
from src.features import compute_leave_one_out_personality, combine_features
from src.bootstrapping import run_bootstrap, save_results

raw_text    = load_json("data/pan17_documents.json")
embeddings  = load_jsonl("data/Pan17_bert_embeddings.jsonl")
sim_triples = load_json("data/emoji_similarity.json")
emoji_vocab = ["😀", "❤️", ...]  # top-N emojis

results, sims, semantic, df = run_bootstrap(
    embeddings=embeddings,
    raw_data=raw_text,
    emoji_vocab=emoji_vocab,
    similarity_triples=sim_triples,
    model_name="BERT",
    emoji_threshold=20,
    num_epochs=10,
    repetitions=10,
)
save_results(results, sims, semantic, df, prefix="results/BERT_top20")
```

---

## Output files

For each `(model, threshold)` combination, the following files are written to `results/`:

| File | Contents |
|---|---|
| `{model}_top{N}_metrics.json` | Standard evaluation metrics (one dict per repetition) |
| `{model}_top{N}_similarity.json` | Top-1/3/5 average EmoSim508 similarity scores |
| `{model}_top{N}_semantic.json` | Cluster-level evaluation metrics |
| `{model}_top{N}_predictions.xlsx` | Per-sample actual / top-1 / top-3 / top-5 predictions |

---

## Method details

### Feature concatenation

All feature vectors for a tweet are concatenated in the order:

```
[personality (5-d)] ++ [emotion (6-d)] ++ [usage (4–7-d)] ++ [BERT (768-d)]
```

The model input size adjusts automatically.

### Temporal split (no leakage)

User-level features (personality, usage patterns) are computed from a user's *historical* tweets, excluding the current tweet.  
The dataset is split by time so that no future tweet is used to predict an earlier one.

### Bootstrapping

10 independent random train/test splits (70:10:20 ratio) are run per experiment.  
The mean and standard deviation of all metrics are reported.

### Semantic evaluation

Emojis are grouped into clusters using union-find on EmoSim508 pairs with similarity ≥ 0.7.  
Predictions are mapped to their cluster before computing standard metrics, rewarding semantically correct but visually different emoji choices.

---

## Citation

If you use this code, please cite the original paper:

```bibtex
@article{gaafar2026personalizedbert,
  title   = {Personalized-BERT: Personalized Emoji Prediction Framework
             Using Emotion, and User Preferences, and Personality},
  author  = {Gaafar, Mariam and Rafea, Ahmed and El-Bolock, Alia},
  journal = {Procedia Computer Science},
  volume  = {275},
  pages   = {762--770},
  year    = {2026},
  publisher = {Elsevier}
}
```
