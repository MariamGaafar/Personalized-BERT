import json
import numpy as np
import os 
import pandas as pd
import matplotlib.pyplot as plt
import unicodedata
from collections import defaultdict
import torch
from itertools import chain
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import (
    accuracy_score, 
    precision_recall_fscore_support
)
import torch.optim as optim
import numpy as np
import gc  # ✅ Import garbage collector
from collections.abc import Mapping
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import LabelEncoder
from supporting_functions import ( 
                                  extract_deepmoji_emojis,
                                    convert_list_jsonl_to_dict,
                                    evaluate_on_test_dataset,
                                    get_similarity,
                                    get_prediction)


import torch.nn as nn


# Convert np.float64 to native float
def convert_np(obj):
    if isinstance(obj, np.float64):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


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
    
    
    return final_metrics, max_metrics, max_epochs
def label_encode_y(y, le):
    y_single_label = [labels[0] for labels in y]
    y_class = np.array([
        le.transform([label])[0] for label in y_single_label
    ])
    print("Output (y_class) shape:", y_class.shape)
    print("First few values of y_class:", y_class[:10])
    return y_class



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
                        first_emoji = emojis_per_sentence[i][0]  # ✅ Keep only the first emoji
                        entry = (filename, tuple(embeddings[i]), first_emoji)

                        if entry not in final_dataset:  # ✅ Avoid duplicates
                            final_dataset.add(entry)
                            indices.add((filename, i, first_emoji))  # ✅ Store (filename, sentence index, first emoji)

    # ✅ Convert dataset to NumPy array
    X = np.array([entry[1] for entry in final_dataset])  # (filtered_sentences, 768)
    y = [entry[2] for entry in final_dataset]  # Keep as a list

    print(f"Total Sentences AFTER Filtering: {len(final_dataset)}")
    print("Input (X) shape:", X.shape)  # Expected: (filtered_sentences, 768)
    print("Output (y) example:", y[:5])  # Expected: List of first emojis

    return emoji_dict, X, y, indices


def main_fun(embeddings, raw_data, loaded_emoji_similarity_list, EMOJI_VOCAB,
             model_name='Bert', checkpoint_path="None", emoji_threshold=20, number_of_epochs=20, repetitions=5):


    # ✅ Fit encoder ONCE on the full emoji vocab

    # To store results across repetitions
    all_results = []
    all_similarities = []
    actual = []
    predicted=[]
    top3=[]
    top5=[]

    for i in range(repetitions):
        print(f"\n--- Repetition {i+1}/{repetitions} ---")
        

        checkpoint_path = f"/Users/mariamgaafar/Desktop/Thesis/code/models_{emoji_threshold}/" \
                          f"{model_name}_top_emojis_{emoji_threshold}_num_epochs_{number_of_epochs}_rep_{i+1}.pth"

        # ✅ Get data and indices
        emoji_dict, X, y, indices = split_data_x_and_y(embeddings, raw_data, list(EMOJI_VOCAB))
        flat_y = [label for sublist in y for label in sublist]
        all_emojis = list(set(EMOJI_VOCAB).union(set(flat_y)))
        le2 = LabelEncoder()
        le2.fit(all_emojis)
        y_class = label_encode_y(y, le2)


        X_train, X_test, y_train, y_test, _, _ = train_test_split(
            X, y_class, list(indices), test_size=0.2, random_state=None, stratify=y_class
        )

        f_met, mx_met, mx_ep = Deep_learning_model(
            X_train, X_test, y_train, y_test,
            checkpoint_path=checkpoint_path,
            num_epochs=number_of_epochs,
            num_claases=len(all_emojis)
        )
        a, p, t3  = get_prediction(X_test,y_test,checkpoint_path ,le2,3, len(all_emojis))
        _, _, t5 = get_prediction(X_test,y_test,checkpoint_path ,le2,5, len(all_emojis))
        actual.append(a)
        predicted.append(p)
        top3.append(t3)
        top5.append(t5)

        results = evaluate_on_test_dataset(
            X_test, y_test,
            checkpoint_path=checkpoint_path,
            num_classes=len(all_emojis)
        )

        similarities = get_similarity(
            X_test, y_test,
            checkpoint_path=checkpoint_path,
            loaded_emoji_similarity_list=loaded_emoji_similarity_list,
            le=le2,
            num_classes=len(all_emojis)
        )

        all_results.append(results)
        all_similarities.append(similarities)

        # ✅ After each repetition, clean up memory
        del X, y, indices, emoji_dict
        del X_train, X_test, y_train, y_test
        del f_met, mx_met, mx_ep, results, similarities
        gc.collect()

    # Optionally, print aggregated summary
    print("\n--- Summary of Results ---")
    summary = defaultdict(list)
    for res in all_results:
        for key, value in res.items():
            summary[key].append(value)

    for metric, values in summary.items():
        if isinstance(values[0], Mapping):  # e.g., dict
            print(f"\nMetric group: {metric}")
            for sub_metric in values[0].keys():
                sub_values = [v[sub_metric] for v in values]
                print(f"{sub_metric}: Mean = {np.mean(sub_values):.4f}, Std = {np.std(sub_values):.4f}")
        else:
            print(f"{metric}: Mean = {np.mean(values):.4f}, Std = {np.std(values):.4f}")
            
    # Flatten all lists
    flat_actual = list(chain.from_iterable(actual))
    flat_predicted = list(chain.from_iterable(predicted))
    flat_top3 = list(chain.from_iterable(top3))
    flat_top5 = list(chain.from_iterable(top5))

    # Then build the DataFrame
    df = pd.DataFrame({
        'actual': flat_actual,
        'predicted': flat_predicted,
        'top3': flat_top3,
        'top5': flat_top5
    })

    return all_results, all_similarities, df
    # Recursive flatten



def save_metrics(results_dict ,similarity_dict, results_file_name , similarity_file_name  ): 
    json_path = results_file_name
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, ensure_ascii=False, indent=4)
    json_path = similarity_file_name
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(similarity_dict, f, ensure_ascii=False, indent=4)
    
    
    
emoji_df= pd.read_excel('/Users/mariamgaafar/Desktop/Thesis/Data/emoji_counts.xlsx')
with open("emoji_similarity.json", "r", encoding="utf-8") as f:
    loaded_emoji_similarity_list = json.load(f)

with open( "/Users/mariamgaafar/Desktop/Thesis/Data/pan17_documents.json") as f:
    raw_text_with_emojis  =  json.load(f)
    
results_dict = dict()
similarity_dict = dict()


EMOJI_VOCAB= list(emoji_df['Emoji'][0:300])


json_path = "bert.json"

# Load JSON file into a Python variable
with open(json_path, "r", encoding="utf-8") as f:
    Bert_embeddings = json.load(f)  # Read JSON into Python variable

emoji_treshold = 250

all_res_20,all_sim_20, df= main_fun(embeddings=Bert_embeddings,
         raw_data=raw_text_with_emojis,
         EMOJI_VOCAB=EMOJI_VOCAB[0:250],
         model_name='Bert',
         checkpoint_path='BERT_Bootstrapping',
         emoji_threshold=emoji_treshold,
         number_of_epochs=10,
         repetitions=10,
         loaded_emoji_similarity_list=loaded_emoji_similarity_list
         )
    
    
# Save to JSON
with open(f"BERT_{emoji_treshold}_sim.json", 'w') as f:
    json.dump(all_res_20, f, indent=4, default=convert_np)
    
    # Save to JSON
with open(f"BERT_{emoji_treshold}.json", 'w') as f:
    json.dump(all_res_20, f, indent=4, default=convert_np)
    
df.to_excel(f"BERT{emoji_treshold}_results.xlsx")