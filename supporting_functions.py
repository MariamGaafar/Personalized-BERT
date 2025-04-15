import re 
import json 
import numpy as np
import os 
import torch
import torch.nn as nn
from collections import defaultdict
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import torch.optim as optim
import unicodedata
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support
)
from sklearn.metrics import precision_recall_fscore_support



# ✅ Function to extract emojis from text
def extract_deepmoji_emojis(text, EMOJI_VOCAB):
    """Extracts only the 64 DeepMoji emojis from text."""
    EMOJI_VOCAB = [unicodedata.normalize("NFKC", str(i)) for i in EMOJI_VOCAB ]    # ✅ Keep only the first emoji
    emoji_pattern = re.compile("|".join(map(re.escape, EMOJI_VOCAB)))  # Convert list to regex pattern
    emojis = emoji_pattern.findall(text)
    return [e for e in emojis if e in EMOJI_VOCAB]  # Filter only DeepMoji emojis


def load_jsonl_into_array(file_name):

    # Initialize a list to store the parsed JSON objects
    lines = []

    # Open the JSONL file and read it line by line
    with open(file_name, 'r', encoding='utf-8') as f:
        for line in f:
            # Strip any leading/trailing whitespace and parse the JSON object
            json_obj = json.loads(line.strip())
            # Append the parsed JSON object to the list
            lines.append(json_obj)

    return lines     
def convert_list_jsonl_to_dict(jsonl_list):
    combined_dict = {}
    for file_dict in jsonl_list:
        combined_dict.update(file_dict)  # Merge all dicts into one
    return combined_dict 

import numpy as np
from collections import defaultdict

def split_data_x_and_y(Embeddings, raw_text, emoji_vocab):
    ''' 
    Splits embeddings and aligns them with corresponding emojis.
    
    Indices is a set of tuples (file, sentence_index, first_emoji).
    '''
    # ✅ Step 1: Extract emojis from text
    emoji_dict = defaultdict(list)
    for filename, sentences in raw_text.items():
        emoji_dict[filename] = [extract_deepmoji_emojis(sentence, emoji_vocab) for sentence in sentences]

    # ✅ Step 2: Convert BERT embeddings to a dictionary
    bert_dict = convert_list_jsonl_to_dict(Embeddings)

    # ✅ Step 3: Align embeddings with corresponding emojis
    final_dataset = set()
    indices = set()  # Track (author_id, sentence_index, first_emoji)

    for filename in bert_dict:
        if filename in emoji_dict:  # Ensure the file exists in both sources
            embeddings = bert_dict[filename]  # (100, 768)
            emojis_per_sentence = emoji_dict[filename]  # (100, variable length)

            # ✅ Ensure both have exactly 100 entries
            if len(embeddings) == 100 and len(emojis_per_sentence) == 100:
                for i in range(100):  # Loop over all sentences
                    if len(emojis_per_sentence[i]) > 0:  # Only keep sentences with emojis
                        print(str(emojis_per_sentence[i][0]))
                        first_emoji = emojis_per_sentence[i][0]  # ✅ Keep only the first emoji
                        entry = (filename, tuple(embeddings[i]), first_emoji)

                        if entry not in final_dataset:  # ✅ Avoid duplicates
                            final_dataset.add(entry)
                            print(first_emoji)  # ✅ Debugging output
                            indices.add((filename, i, first_emoji))  # ✅ Store (filename, sentence index, first emoji)

    # ✅ Convert dataset to NumPy array
    X = np.array([entry[1] for entry in final_dataset])  # (filtered_sentences, 768)
    y = [entry[2] for entry in final_dataset]  # Keep as a list

    print(f"Total Sentences AFTER Filtering: {len(final_dataset)}")
    print("Input (X) shape:", X.shape)  # Expected: (filtered_sentences, 768)
    print("Output (y) example:", y[:5])  # Expected: List of first emojis

    return emoji_dict, X, y, indices

def deduplicate_data(train_indices, test_indices):
    """Remove duplicate (filename, sentence) pairs while maintaining structure."""

    unique_train = set()
    unique_test = set()

    # Deduplicate training data
    for entry in train_indices:
        unique_train.add((entry[0], entry[1]))  # Only store (filename, sentence)

    # Deduplicate testing data
    for entry in test_indices:
        unique_test.add((entry[0], entry[1]))  # Only store (filename, sentence)

    print(f"Before Deduplication - Train: {len(train_indices)}, Test: {len(test_indices)}")
    print(f"After Deduplication - Train: {len(unique_train)}, Test: {len(unique_test)}")

    return list(unique_train), list(unique_test)

def spread_embeddings_sentences(embeddings_new, emoji_dict, raw_data):
    final_dataset = set()
    indices= set()
    for filename in embeddings_new:
        if filename in emoji_dict:  # Ensure the file exists in both sources
            embeddings = embeddings_new[filename]  # (100, 768)
            emojis = emoji_dict[filename]  # (100, variable length)

            # ✅ Ensure both have exactly 100 entries
            if len(embeddings) == len(emojis):
                for i in range(len(embeddings)):
                    if len(emojis[i]) > 0:
                        first_emoji = unicodedata.normalize("NFKC", emojis[i][0])  # ✅
                        entry = (filename, tuple(embeddings[i]), first_emoji)
                        if entry not in final_dataset:  # ✅ Avoid duplicates
                            final_dataset.add(entry)
                            indices.add((filename, i, first_emoji))  # ✅ Store (filename, sentence index, first emoji)
    return final_dataset #Now, it's hashable
def pull_data_from_saved_indices(new_embeddings, raw_data, train_indices, test_indices):
    # ✅ Step 1: Merge BERT embeddings into a single dictionary

    new_embeddings = convert_list_jsonl_to_dict(new_embeddings )

    # ✅ Step 2: Initialize defaultdict for grouping embeddings
    train_embeddings = defaultdict(list)
    train_emojis = defaultdict(list)
    test_embeddings = defaultdict(list)
    test_emojis = defaultdict(list)
    for filename, sentence , emoji  in train_indices:
        train_embeddings[filename].append(new_embeddings[filename][int(sentence)])
        train_emojis[filename].append(emoji)

    count_test=0

    for filename, sentence , emoji in test_indices:
        test_embeddings[filename].append(new_embeddings[filename][int(sentence)])
        test_emojis[filename].append(emoji)
        count_test+=1
    
    print("Test shape", count_test)


    # ✅ Step 2: Align embeddings with corresponding emojis, filtering out sentences with no emojis
    final_dataset_train = spread_embeddings_sentences(train_embeddings, train_emojis, raw_data)
    # ✅ Convert dataset to NumPy array

    print("Train dataset after spread ", len(final_dataset_train ))


    X_train_new= np.array([entry[1] for entry in final_dataset_train])  # (filtered_sentences, 768)
    y_train_new =  [entry[2] for entry in final_dataset_train]  # Keep as a list|
    
    
    final_dataset_test = spread_embeddings_sentences(test_embeddings, test_emojis, raw_data)
    print("Test dataset after spread ", len(final_dataset_test ))

    # ✅ Convert dataset to NumPy array
    X_test_new= np.array([entry[1] for entry in final_dataset_test])  # (filtered_sentences, 768)
    y_test_new =  [entry[2] for entry in final_dataset_test]  # Keep as a list


    return np.array(X_train_new), np.array(y_train_new), np.array(X_test_new) , np.array(y_test_new)



# Convert to a consistent float32 format
def convert_to_float32(data):
    if isinstance(data, list):
        return [convert_to_float32(item) for item in data]  # Recursively apply to lists
    elif isinstance(data, dict):
        return {key: convert_to_float32(value) for key, value in data.items()}  # Apply to dicts
    elif isinstance(data, (np.float32, np.float64, float)):  
        return float(np.float32(data))  # Convert to float32
    return data  # Return unchanged for other data types
def label_encode_y(y, le):
    y_single_label = [labels[0] for labels in y]
    y_class = np.array([
        le.transform([label])[0] if label in le.classes_ else -1
        for label in y_single_label
    ])
    print("Output (y_class) shape:", y_class.shape)
    print("First few values of y_class:", y_class[:10])
    return y_class


def metrics_at_k(y_true, y_prob, k):
    """
    Compute Accuracy@K, Precision@K, Recall@K, and F1@K for multi-class classification.
    
    Args:
        y_true (np.ndarray): True class labels.
        y_prob (np.ndarray): Predicted probability distribution (shape: num_samples x num_classes).
        k (int): The number of top predictions to consider.

    Returns:
        dict: Dictionary containing Accuracy@K, Precision@K, Recall@K, and F1@K (micro, macro, weighted).
    """
    
    # ✅ Step 1: Get Top-K Predictions
    top_k_preds = np.argsort(y_prob, axis=1)[:, -k:]  # Get top-K predictions per sample
    top_1_preds = np.argmax(y_prob, axis=1)  # Get top-1 predictions

    # ✅ Step 2: Compute Standard Accuracy (Top-1 Accuracy)
    correct_top1 = np.sum(top_1_preds == y_true)
    accuracy_top1 = correct_top1 / len(y_true)  # Standard Accuracy

    # ✅ Step 3: Compute Top-K Accuracy
    correct_topk = np.sum([1 if y_true[i] in top_k_preds[i] else 0 for i in range(len(y_true))])
    accuracy_k = correct_topk / len(y_true)

    # ✅ Step 4: Compute Precision@K & Recall@K
    precision_per_sample = [len(set(top_k_preds[i]) & {y_true[i]}) / k for i in range(len(y_true))]
    recall_per_sample = [len(set(top_k_preds[i]) & {y_true[i]}) for i in range(len(y_true))]

    precision_k = np.mean(precision_per_sample)  # Micro Precision@K
    recall_k = np.mean(recall_per_sample)  # Micro Recall@K

    # ✅ Step 5: Compute F1@K (Harmonic Mean of Precision & Recall)
    f1_k = (2 * precision_k * recall_k) / (precision_k + recall_k) if (precision_k + recall_k) > 0 else 0.0

    # ✅ Step 6: Compute Macro & Weighted Precision, Recall, and F1 using sklearn
    y_pred_binary = np.zeros_like(y_prob)  # Create a binary matrix for top-k predictions
    for i in range(len(y_true)):
        y_pred_binary[i, top_k_preds[i]] = 1  # Mark top-K predicted classes as 1

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred_binary.argmax(axis=1), average="macro"
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred_binary.argmax(axis=1), average="weighted"
    )

    # ✅ Step 7: Return all metrics
    return {
        "Accuracy@1": accuracy_top1,
        "Accuracy@K": accuracy_k,
        "Micro Precision@K": precision_k,
        "Micro Recall@K": recall_k,
        "Micro F1@K": f1_k,
        "Macro Precision@K": precision_macro,
        "Macro Recall@K": recall_macro,
        "Macro F1@K": f1_macro,
        "Weighted Precision@K": precision_weighted,
        "Weighted Recall@K": recall_weighted,
        "Weighted F1@K": f1_weighted
    }


# ✅ Function to save model checkpoint
def save_checkpoint(epoch, model, optimizer, loss, path="checkpoint.pth"):
    checkpoint = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "loss": loss
    }
    torch.save(checkpoint, path)
    print(f"✅ Checkpoint saved at epoch {epoch+1}")

# ✅ Function to load model checkpoint
def load_checkpoint(model, optimizer, path="checkpoint.pth"):
    if os.path.exists(path):
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = checkpoint["epoch"] + 1  # Resume from next epoch
        print(f"✅ Resuming from epoch {start_epoch}")
        return start_epoch
    return 0  # Start from scratch if no checkpoint

import torch.nn as nn

class NeuralNet(nn.Module):
    def __init__(self, input_size=768, hidden_size=768,num_classes =20):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # First linear layer
        self.relu = nn.ReLU()  # Activation function
        self.out = nn.Linear(hidden_size, num_classes)  # Output layer

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)  # Non-linearity
        x = self.out(x)  # No softmax here (handled by CrossEntropyLoss)
        return x

def Deep_learning_model(
    X_train, X_test, y_train, y_test, 
    checkpoint_path="", batch_size=128, num_epochs=100, num_claases = 20, val_split=0.1
):
    # ✅ Convert data to PyTorch tensors
    y_train = y_train.astype(int)  
    y_test = y_test.astype(int)
    
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    # ✅ Split training data into training & validation sets
    train_size = int((1 - val_split) * len(X_train_tensor))
    val_size = len(X_train_tensor) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        TensorDataset(X_train_tensor, y_train_tensor), 
        [train_size, val_size]
    )

    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # ✅ Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = NeuralNet(input_size=X_train.shape[1], num_classes=num_claases).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # ✅ Load checkpoint if exists
    start_epoch = 0
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1  
        print(f"Resuming training from epoch {start_epoch}...")

    # ✅ Metric tracking
    epoch_losses, epoch_accuracies, epoch_precisions, epoch_recalls, epoch_f1s = [], [], [], [], []
    val_losses, val_accuracies, val_precisions, val_recalls, val_f1s = [], [], [], [], []
    
    print("Start training from epoch:", start_epoch)

    for epoch in range(start_epoch, num_epochs):
        model.train()
        epoch_loss, correct, total = 0, 0, 0
        all_preds, all_labels, all_probs = [], [], []

        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()

            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * batch_X.size(0)
            _, predicted = torch.max(outputs, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()

            all_preds.extend(predicted.cpu().numpy())  
            all_labels.extend(batch_y.cpu().numpy())  
            all_probs.extend(torch.softmax(outputs, dim=1).detach().cpu().numpy())  

        # ✅ Compute Training Metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average="weighted"
        )
        epoch_loss /= total  

        epoch_losses.append(epoch_loss)
        epoch_accuracies.append(accuracy)
        epoch_precisions.append(precision)
        epoch_recalls.append(recall)
        epoch_f1s.append(f1)

        # ✅ Validation Phase
        model.eval()
        val_loss, val_correct, val_total = 0, 0, 0
        val_preds, val_labels = [], []

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_X.size(0)

                _, predicted = torch.max(outputs, 1)
                val_total += batch_y.size(0)
                val_correct += (predicted == batch_y).sum().item()

                val_preds.extend(predicted.cpu().numpy())  
                val_labels.extend(batch_y.cpu().numpy())  

        val_accuracy = accuracy_score(val_labels, val_preds)
        val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(
            val_labels, val_preds, average="weighted"
        )
        val_loss /= val_total  

        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        val_precisions.append(val_precision)
        val_recalls.append(val_recall)
        val_f1s.append(val_f1)

        print(f"Epoch {epoch+1}/{num_epochs} -> Loss: {epoch_loss:.4f}, Acc: {accuracy:.4f}, F1: {f1:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict()
        }, checkpoint_path)

    torch.save(model.state_dict(), checkpoint_path)    
    print(f"✅ Training complete. Model saved as {checkpoint_path}")

    # ✅ Metrics Storage
    final_metrics = {
        "Accuracy": epoch_accuracies[-1],
        "Precision": epoch_precisions[-1],
        "Recall": epoch_recalls[-1],
        "F1 Score": epoch_f1s[-1],
        "Validation Accuracy": val_accuracies[-1],
        "Validation Precision": val_precisions[-1],
        "Validation Recall": val_recalls[-1],
        "Validation F1 Score": val_f1s[-1]
    }

    max_metrics = {
        "Accuracy": max(epoch_accuracies),
        "Precision": max(epoch_precisions),
        "Recall": max(epoch_recalls),
        "F1 Score": max(epoch_f1s),
        "Validation Accuracy": max(val_accuracies),
        "Validation Precision": max(val_precisions),
        "Validation Recall": max(val_recalls),
        "Validation F1 Score": max(val_f1s)
    }

    max_epochs = {
        metric: start_epoch + values.index(max_metrics[metric])
        for metric, values in zip(
            ["Accuracy", "Precision", "Recall", "F1 Score", "Validation Accuracy", "Validation Precision", "Validation Recall", "Validation F1 Score"],
            [epoch_accuracies, epoch_precisions, epoch_recalls, epoch_f1s, val_accuracies, val_precisions, val_recalls, val_f1s]
        )
    }

    # ✅ Plot Training vs Validation
    epochs = list(range(start_epoch, num_epochs))
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))

    ax[0].plot(epochs, epoch_losses, label="Train Loss", color="blue")
    ax[0].plot(epochs, val_losses, label="Val Loss", color="red", linestyle="dashed")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")
    ax[0].set_title("Loss per Epoch")
    ax[0].legend()
    ax[0].grid(True)

    ax[1].plot(epochs, epoch_accuracies, label="Train Accuracy", color="green")
    ax[1].plot(epochs, val_accuracies, label="Val Accuracy", color="orange", linestyle="dashed")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Accuracy")
    ax[1].set_title("Accuracy per Epoch")
    ax[1].legend()
    ax[1].grid(True)

    plt.tight_layout()
    plt.show()

    return final_metrics, max_metrics, max_epochs

def get_prediction(X_test, y_test, checkpoint_path, label_encoder, n, num_classes):
    # Define the model architecture again
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    new_model = NeuralNet(input_size=X_test.shape[1],num_classes=num_classes).to(device)
    # Load the saved weights
    new_model.load_state_dict(torch.load(checkpoint_path))
    new_model.eval()  # Set model to evaluation mode

    # Convert to NumPy arrays with correct data types
    X_test = np.array(X_test, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.int64)

    # Convert to PyTorch tensors
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

    # Get predictions
    with torch.no_grad():
        outputs = new_model(X_test_tensor)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()  # Get class probabilities
        preds = torch.argmax(outputs, dim=1).cpu().numpy()  # Top-1 predicted labels
        top_n_preds = np.argsort(probs, axis=1)[:, -n:][:, ::-1]  # Top-n predicted labels

    # Reverse label encoding to get emojis instead of label numbers
    y_test_emojis = label_encoder.inverse_transform(y_test)  # Actual emojis
    preds_emojis = label_encoder.inverse_transform(preds)  # Top-1 predicted emojis
    top_n_preds_emojis = [label_encoder.inverse_transform(row) for row in top_n_preds]  # Top-n predicted emojis

    return  y_test_emojis.tolist(), preds_emojis.tolist(), top_n_preds_emojis
    

def evaluate_on_test_dataset(X_test, y_test, checkpoint_path, num_classes=20):
    """
    Evaluates a trained neural network model on the test dataset.

    Args:
        X_test (np.ndarray): Test feature matrix.
        y_test (np.ndarray): Test labels.
        checkpoint_path (str): Path to the saved model weights.
        num_classes (int): Number of classes in the classification task.

    Returns:
        dict: Dictionary containing various evaluation metrics.
    """

    # ✅ Set device (use GPU if available, otherwise use CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ✅ Recreate the model architecture
    new_model = NeuralNet(input_size=X_test.shape[1], num_classes=num_classes).to(device)

    # ✅ Load the trained weights (handle different devices)
    new_model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    new_model.eval()  # Set model to evaluation mode

    # ✅ Convert inputs to NumPy arrays and ensure correct data types
    X_test = np.array(X_test, dtype=np.float32)
    y_test = np.array(y_test, dtype=np.int64)  # Classification labels should be integers

    # ✅ Convert NumPy arrays to PyTorch tensors
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

    # ✅ Perform inference
    with torch.no_grad():
        outputs = new_model(X_test_tensor)  # Model predictions
        probs = torch.softmax(outputs, dim=1).cpu().numpy()  # Convert logits to probabilities
        preds = torch.argmax(outputs, dim=1).cpu().numpy()  # Get predicted class labels

    # ✅ Compute classification metrics
    accuracy = accuracy_score(y_test, preds)

    # ✅ Compute precision, recall, and F1 for different averaging methods
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(y_test, preds, average="micro")
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(y_test, preds, average="macro")
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(y_test, preds, average="weighted")

    # ✅ Compute top-3 and top-5 accuracy metrics

    top3_metrics = metrics_at_k(y_test, probs, k=3)
    top5_metrics = metrics_at_k(y_test, probs, k=5)

    # ✅ Print evaluation results
     # ✅ Store all results in a structured dictionary
    metrics_dict = {
        "Accuracy": accuracy,
        "Micro-Averaged": {
            "Precision": precision_micro,
            "Recall": recall_micro,
            "F1 Score": f1_micro
        },
        "Macro-Averaged": {
            "Precision": precision_macro,
            "Recall": recall_macro,
            "F1 Score": f1_macro
        },
        "Weighted-Averaged": {
            "Precision": precision_weighted,
            "Recall": recall_weighted,
            "F1 Score": f1_weighted
        },
        "Top-3 Metrics": {
            "Top-3 Accuracy": top3_metrics["Accuracy@K"],
            "Micro Precision": top3_metrics["Micro Precision@K"],
            "Micro Recall": top3_metrics["Micro Recall@K"],
            "Micro F1 Score": top3_metrics["Micro F1@K"],
            "Macro Precision": top3_metrics["Macro Precision@K"],
            "Macro Recall": top3_metrics["Macro Recall@K"],
            "Macro F1 Score": top3_metrics["Macro F1@K"],
            "Weighted Precision": top3_metrics["Weighted Precision@K"],
            "Weighted Recall": top3_metrics["Weighted Recall@K"],
            "Weighted F1 Score": top3_metrics["Weighted F1@K"],
        },
        "Top-5 Metrics": {
            "Top-5 Accuracy": top5_metrics["Accuracy@K"],
            "Micro Precision": top5_metrics["Micro Precision@K"],
            "Micro Recall": top5_metrics["Micro Recall@K"],
            "Micro F1 Score": top5_metrics["Micro F1@K"],
            "Macro Precision": top5_metrics["Macro Precision@K"],
            "Macro Recall": top5_metrics["Macro Recall@K"],
            "Macro F1 Score": top5_metrics["Macro F1@K"],
            "Weighted Precision": top5_metrics["Weighted Precision@K"],
            "Weighted Recall": top5_metrics["Weighted Recall@K"],
            "Weighted F1 Score": top5_metrics["Weighted F1@K"],
        }
    }

    print(metrics_dict)
    # ✅ Return dictionary of evaluation metrics
    return metrics_dict

def append_model_results(df_results, classifier_name, results_dict):
    """Append model evaluation results to an existing DataFrame with standardized column names."""
    
    # Standardize keys to match DataFrame columns
    key_mapping = {
        "F1 Score": "F1-score",
        "top_3_precision": "Precision@3",
        "top_3_recall": "Recall@3",
        "top3_f1": "f1@3",
        "top5_prec": "Precision@5",
        "top5_rec": "Recall@5",
        "top5_f1": "f1@5"
    }
    
    # Rename keys in results_dict
    standardized_results = {key_mapping.get(k, k): v for k, v in results_dict.items()}
    
    # Add classifier name
    standardized_results["Classifier"] = classifier_name
    
    # Convert to DataFrame and append using pd.concat()
    new_row = pd.DataFrame([standardized_results])
    df_results = pd.concat([df_results, new_row], ignore_index=True)

    return df_results


def calculate_mean_personality_traits(Bert_personality):
    mean_personality_traits = {}

    # Iterate over each person in the dataset
    for person in Bert_personality:
        for person_id, personality_scores in person.items():
            num_posts = len(personality_scores)
            
            # Initialize list to store leave-one-out means
            leave_one_out_means = []
            
            for i in range(num_posts):
                # Exclude the current post
                remaining_scores = personality_scores[:i] + personality_scores[i+1:]
                
                if remaining_scores:
                    # Extract all values for each trait
                    traits = {trait: [entry[trait] for entry in remaining_scores] for trait in personality_scores[0]}
                    
                    # Compute mean for each trait
                    mean_traits = {trait: np.mean(values) for trait, values in traits.items()}
                else:
                    # If there's only one post, set mean to None
                    mean_traits = {trait: None for trait in personality_scores[0]}
                
                leave_one_out_means.append(mean_traits)
            
            # Store result
            mean_personality_traits[person_id] = leave_one_out_means

    # Save to JSON file
    with open("leave_one_out_mean_personality.json", "w") as f:
        json.dump(mean_personality_traits, f, indent=4)

    print("Leave-one-out mean personality traits saved to leave_one_out_mean_personality.json")
    return mean_personality_traits

def combine_personality_with_embeddings(Bert_embeddings , mean_personality_traits):
    updated_embeddings = []

    # Iterate through each dictionary in the embeddings list
    for author_dict in Bert_embeddings:
        for author_id, embeddings_list in author_dict.items():  # Extract key-value pair
            curr_dict=dict()
            curr_dict[author_id]=[]
            for i in range(0, len(embeddings_list)):
                current_traits=  mean_personality_traits[author_id][i]
                traits_list = list(current_traits.values())
  # Get personality traits (100 values)
                # Append embeddings to personality traits for each sentence
                curr_dict[author_id].append(convert_to_float32(traits_list) + convert_to_float32(embeddings_list[i]))
                    
            updated_embeddings.append(curr_dict)
    return updated_embeddings


def avg_emoji_similarity(predicted_list, actual_list, loaded_emoji_similarity_list):
    # Convert list to dictionary for fast lookup
    emoji_similarity_dict = {(str(e1), str(e2)): sim for e1, e2, sim in loaded_emoji_similarity_list}
    emoji_similarity_dict.update({(str(e2), str(e1)): sim for e1, e2, sim in loaded_emoji_similarity_list})  # Add reverse mappings

    # Calculate average similarity, excluding missing pairs
    total_similarity = 0
    count = 0

    for predicted_emojis, actual in zip(predicted_list, actual_list):
        # Convert NumPy array to Python list (if applicable)
        if isinstance(predicted_emojis, np.ndarray):
            predicted_emojis = predicted_emojis.tolist()
        if isinstance(actual, np.ndarray):
            actual = actual.item()  # Convert single-element NumPy array to string

        # Ensure predicted_emojis is a list (handles cases where it's a single emoji)
        if not isinstance(predicted_emojis, list):
            predicted_emojis = [predicted_emojis]

        best_similarity = 0  # Track the highest similarity found for this actual emoji
        for pred in predicted_emojis:
            similarity = emoji_similarity_dict.get((str(pred), str(actual)), 0)  # Default to 0 if not found
            best_similarity = max(best_similarity, similarity)  # Choose the best match

        if best_similarity > 0:  # Only consider cases where a similarity score exists
            total_similarity += best_similarity
            count += 1

    # Compute average similarity
    average_similarity = total_similarity / count if count > 0 else 0

    print("Average Similarity:", average_similarity)
    return average_similarity

def get_similarity (X_bert_test,y_bert_test, checkpoint_path, loaded_emoji_similarity_list, le , num_classes=20):
    actual, predicted, top  = get_prediction(X_bert_test,y_bert_test,checkpoint_path ,le,3, num_classes)
    top1_average_similarity  = avg_emoji_similarity(predicted,actual, loaded_emoji_similarity_list)
    top3_average_similarity = avg_emoji_similarity(top,actual, loaded_emoji_similarity_list)
    actual, predicted, top5  = get_prediction(X_bert_test,y_bert_test, checkpoint_path ,le,5 , num_classes)
    top5_average_similarity  = avg_emoji_similarity(top5,actual, loaded_emoji_similarity_list)
    print("top1_avg_similarity" , top1_average_similarity)
    print("top3_avg_similarity" , top3_average_similarity)
    print("top5_avg_similarity",top5_average_similarity)
    return {"top1_avg_similarity" : top1_average_similarity,
            "top3_avg_similarity": top3_average_similarity, 
            "top5_avg_similarity" : top5_average_similarity}


