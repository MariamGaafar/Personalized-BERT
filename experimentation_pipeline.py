import os
import re
import json
import pickle
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from supporting_functions import (EMOJI_VOCAB ,
                                  load_jsonl_into_array,
                                    split_data_x_and_y,
                                    label_encode_y,
                                    calculate_mean_personality_traits,
                                    combine_personality_with_embeddings,
                                    convert_list_jsonl_to_dict)
     
# ✅ Encode labels
le = LabelEncoder()
le.fit(EMOJI_VOCAB)
with open( "/Users/mariamgaafar/Desktop/Thesis/Data/pan17_documents.json") as f:
    raw_text_with_emojis  =  json.load(f)
# Define the path to your JSONL file
file_path = '/Users/mariamgaafar/Desktop/Thesis/Data/Pan17_bert_embeddings.jsonl'
# Initialize a list to store the parsed JSON objects
Bert_embeddings  = load_jsonl_into_array(file_path)

# ## feature 1 personality 
# # Define the path to your JSONL file
file_path = '/Users/mariamgaafar/Desktop/Thesis/Data/pan17_personality_predictions_sentance(BERT).jsonl'
# Initialize a list to store the parsed JSON objects
Bert_personality  = []

# # Open the JSONL file and read it line by line
with open(file_path, 'r', encoding='utf-8') as f:
    for line in f:
        # Strip any leading/trailing whitespace and parse the JSON object
        json_obj = json.loads(line.strip())
        # Append the parsed JSON object to the list
        Bert_personality.append(json_obj)
print(Bert_personality[1])
mean_personality_traits = calculate_mean_personality_traits(Bert_personality)

# # Define the path to your JSONL file
file_path = '/Users/mariamgaafar/Desktop/Thesis/Data/pan17_emotion_predictions.jsonl'

# # Initialize a list to store the parsed JSON objects
emotions  = []

# # Open the JSONL file and read it line by line
with open(file_path, 'r', encoding='utf-8') as f:
    for line in f:
        # Strip any leading/trailing whitespace and parse the JSON object
        json_obj = json.loads(line.strip())
        # Append the parsed JSON object to the list
        emotions.append(json_obj)
print(emotions[1])
emotions_dict = convert_list_jsonl_to_dict(emotions)

with open("/Users/mariamgaafar/Desktop/Thesis/emoji_stats_holdout.json", "r") as f:
    emoji_stats = json.load(f)

processed_stats= dict()
for key in emoji_stats:
    processed_stats[key]= []
    for diction in emoji_stats[key]:
      
        if(diction['top_used_emoji_excluding_current'] is not None): 
            encoded_top = label_encode_y(diction['top_used_emoji_excluding_current'], le)
        features= {
        "sentences_with_emoji": diction["sentences_with_emoji/100"],
        "avg_distinct_emojis_per_sentence" : diction["avg_distinct_emojis_per_sentence"],
        "distinct_emoji_ratio" : diction["distinct_emoji_ratio"],
        "end" : diction["end"],
        "start" : diction["start"],
        "middle" : diction["middle"],
        "encoded_top" : encoded_top}
        processed_stats[key].append(features)

# The combinations are 
# BERT 
# ✅ Get data and indices
emoji_dict , X_bert , y_bert, indices = split_data_x_and_y(Bert_embeddings, raw_text_with_emojis)

y_bert_class= label_encode_y(y_bert,le)
# ✅ Perform train-test split while keeping track of indices
X_bert_train, X_bert_test, y_bert_train, y_bert_test, train_indices, test_indices = train_test_split(
    X_bert, y_bert_class, list(indices), test_size=0.2, random_state=42, stratify=y_bert_class
)
# BERT + PERSONALITY

combined_bert_andpersonality = combine_personality_with_embeddings(Bert_embeddings, mean_personality_traits)
print("Done Combined Bert and personality")

# BERT + emotion

combined_bert_and_emotion = combine_personality_with_embeddings(Bert_embeddings, emotions_dict )
print("Done combined bert and emotions ")
# BERT + usage PATTERN 
combined_bert_and_usage  = combine_personality_with_embeddings(Bert_embeddings, processed_stats )
print("Done combined Bert and usage ")
# BERT + PERSONALITY + EMOTION
combined_bert_andpersonality_emotion = combine_personality_with_embeddings(combined_bert_andpersonality, emotions_dict)
print("Done combined bert personality and emotion")

# BERT + PERSONALITY + USAGE PATTERN 
combined_bert_andpersonality_usage = combine_personality_with_embeddings(combined_bert_andpersonality, processed_stats)
print("Done combined bert personality and emotion")
# BERT + EMOTION + USAGE PATTERN 
combined_bert_emotion_usage = combine_personality_with_embeddings(combined_bert_and_emotion, processed_stats )
print("Done combined bert personality and emotion and usage")

# BERT + PERSONALITY + EMOTION + USAGE PATTERM 
combined_bert_andpersonality_usage_emotion = combine_personality_with_embeddings(combined_bert_andpersonality_emotion, processed_stats)
print("Done combined bert personality and emotion and usage and emotion")
