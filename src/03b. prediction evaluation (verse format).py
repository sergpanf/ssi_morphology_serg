import os
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def evaluate_predictions(actual_file, predicted_file):
    """
    Reads the standard verse-format files, extracts the tags, 
    and generates evaluation metrics.
    """
    if not os.path.exists(actual_file):
        print(f"Error: Actual file not found at {actual_file}")
        return
    if not os.path.exists(predicted_file):
        print(f"Error: Predicted file not found at {predicted_file}")
        return

    y_true = []
    y_pred = []
    
    total_verses = 0
    mismatched_lengths = 0

    print(f"Reading Actual: {actual_file}")
    print(f"Reading Predicted: {predicted_file}\n")

    with open(actual_file, 'r', encoding='utf-8') as f_act, \
         open(predicted_file, 'r', encoding='utf-8') as f_pred:

        for act_line, pred_line in zip(f_act, f_pred):
            total_verses += 1
            
            act_parts = act_line.strip().split('\t')
            pred_parts = pred_line.strip().split('\t')

            # Ensure both lines have the minimum 4 columns (Book, Ch, V, Tags)
            if len(act_parts) >= 4 and len(pred_parts) >= 4:
                act_tags = act_parts[3].split()
                pred_tags = pred_parts[3].split()
                
                # Safety net: truncate to the minimum length if there's a mismatch
                # (Though your pipeline should guarantee a strict 1:1 ratio)
                min_len = min(len(act_tags), len(pred_tags))
                if len(act_tags) != len(pred_tags):
                    mismatched_lengths += 1

                y_true.extend(act_tags[:min_len])
                y_pred.extend(pred_tags[:min_len])

    # ---------------------------------------------------------
    # METRICS CALCULATION
    # ---------------------------------------------------------
    print("="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    
    print(f"Total Verses Processed: {total_verses}")
    print(f"Total Words Evaluated:  {len(y_true)}")
    if mismatched_lengths > 0:
        print(f"⚠️ Warning: {mismatched_lengths} verses had mismatched tag counts.")
    
    print("-" * 50)
    
    # Accuracy
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Overall Accuracy: {accuracy:.4f}\n")

    # Distinct Labels
    labels = sorted(list(set(y_true + y_pred)))
    
    # Classification Report (Precision, Recall, F1-Score)
    print("Classification Report:")
    # print(classification_report(y_true, y_pred, labels=labels, zero_division=0))
    print(classification_report(y_true, y_pred, labels=labels, zero_division=0, digits=4))
    
    # Confusion Matrix
    print("Confusion Matrix:")
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    # Formatting the confusion matrix with headers for readability
    header = "       " + " ".join([f"{label:>5}" for label in labels])
    print(header)
    for i, label in enumerate(labels):
        row = f"{label:>5} [" + " ".join([f"{val:>5}" for val in cm[i]]) + "]"
        print(row)

if __name__ == "__main__":
    # --- CONFIGURATION ---
    # Update these file paths to match your actual files
    ACTUAL_FILE = "../sp_data/fileB__NT"
    PREDICTED_FILE = "../sp_new_data/fileB__NT_predicted_2026-05-26_subphrases_ep440_reformatted_CLEANED"

    evaluate_predictions(ACTUAL_FILE, PREDICTED_FILE)