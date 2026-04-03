# %% 
from main import main
main(["-mo", "train"
      , "-i", "fileB__NT_predicted_2026-04-03_subphrases_ep40_reformatted"
      , "-o", "fileB__NT"
      , "-ep", "40"
      , "-l", "10"
      , "-lr", "0.0003" # increased from 0.0001
      , "-et", "True"

    #  , "-etonly", "True" # default: False; if True, only the evaluation will be performed, without training. This is useful when we want to evaluate a model that has already been trained and saved, without having to retrain it.
      
      # non-reduced extra parameters
       , "-emb", "256" # default: 512
       , "-nh", "4"  # default: 8
       , "-nel", "2"  # default: 3
       , "-ndl", "2"  # default: 3
    #   , "-ffn", "256"  # default: 2048 
       , "-b", "32" # default: 128
      ]) 

# %% # todo: to update file_path
file_path = "../sp_evaluation_results_transformer/fileA__NT_fileB__NT_ONE_DATASET/results_10seq_len_0.0003lr_256esize_4nh_0.1dout_32_bsize_40ep_3bsize.txt"

from collections import defaultdict

# Provide the path to your file
# file_path = "results_10seq_len_0.0001lr_512embsize_8nhead_transformer_0.1dropout_128_batchsize_10epochs_3beamsize.txt"

# Dictionaries to track counts for every unique class
TP = defaultdict(int)
FP = defaultdict(int)
FN = defaultdict(int)
Support = defaultdict(int) # Tracks the total true occurrences of each class

with open(file_path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

predicted_words = []
for line in lines:
    line = line.strip()
    if not line:
        continue
        
    if line.startswith("Predicted"):
        # Split by spaces and drop the first item ("Predicted")
        predicted_words = line.split()[1:]
        
    elif line.startswith("Truevalue"):
        # Split by spaces and drop the first item ("Truevalue")
        true_words = line.split()[1:]
        
        # Compare the sequences word-by-word
        for p_word, t_word in zip(predicted_words, true_words):
            Support[t_word] += 1
            if p_word == t_word:
                TP[t_word] += 1
            else:
                FP[p_word] += 1
                FN[t_word] += 1

# Collect every unique class encountered in either predictions or true values
all_classes = set(TP.keys()).union(set(FP.keys())).union(set(FN.keys()))

macro_precision = 0.0
macro_recall = 0.0
macro_f1 = 0.0

weighted_precision = 0.0
weighted_recall = 0.0
weighted_f1 = 0.0

total_support = sum(Support.values())

print(f"{'Class':<25} | {'Precision':<9} | {'Recall':<9} | {'F1-Score':<9} | {'Support':<7}")
print("-" * 70)

# Calculate and store metrics for each class first
class_metrics = []
for cls in all_classes:
    tp = TP[cls]
    fp = FP[cls]
    fn = FN[cls]
    support = Support[cls]
    
    # Calculate Precision and Recall (safe against division by zero)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    class_metrics.append((cls, precision, recall, f1, support))

# Sort the classes by F1-Score descending (index 3 is the f1 variable)
class_metrics.sort(key=lambda x: x[3], reverse=True)

# Loop through the sorted metrics to print and calculate averages
for cls, precision, recall, f1, support in class_metrics:
    
    # Uncomment the line below if you want to print the metrics for EVERY single unique class
    print(f"{cls:<25} | {precision:<9.4f} | {recall:<9.4f} | {f1:<9.4f} | {support:<7}")
    
    # Add up metrics for Macro Average
    macro_precision += precision
    macro_recall += recall
    macro_f1 += f1
    
    # Add up metrics for Weighted Average
    weighted_precision += precision * support
    weighted_recall += recall * support
    weighted_f1 += f1 * support

# Finalize Macro Average
num_classes = len(all_classes)
macro_precision /= num_classes
macro_recall /= num_classes
macro_f1 /= num_classes

# Finalize Weighted Average
if total_support > 0:
    weighted_precision /= total_support
    weighted_recall /= total_support
    weighted_f1 /= total_support

print("-" * 70)
print(f"{'Macro Avg':<25} | {macro_precision:<9.4f} | {macro_recall:<9.4f} | {macro_f1:<9.4f} | {total_support:<7}")
print(f"{'Weighted Avg':<25} | {weighted_precision:<9.4f} | {weighted_recall:<9.4f} | {weighted_f1:<9.4f} | {total_support:<7}")



