import collections
from sklearn.metrics import classification_report

def main():
    ACTUAL_FILE = "../sp_data/fileB__NT"
    # PREDICTED_FILE = "../sp_new_data/fileB__NT_predicted_2026-03-27_beam1alpha0h2,5"
    PREDICTED_FILE = "../sp_new_data/fileB__NT_predicted_2026-04-03_subphrases_ep40"
    evaluate_predictions(ACTUAL_FILE, PREDICTED_FILE)

def evaluate_predictions(actual_file_path: str, predicted_file_path: str):
    actual_data = collections.defaultdict(list)
    predicted_data = collections.defaultdict(list)
    
    print(f"Reading actual data from: {actual_file_path}...")
    with open(actual_file_path, 'r', encoding='utf-8-sig') as f_actual:
        for line in f_actual:
            parts = line.strip().split('\t') if '\t' in line else line.strip().split()
            if not parts: continue
            
            # Format: Book \t Chapter \t Verse \t Y Ỹ Ỵ ...
            address = (parts[0].strip(), parts[1].strip(), parts[2].strip())
            
            if len(parts) == 4 and ' ' in parts[3]:
                actual_labels = parts[3].split()
            else:
                actual_labels = parts[3:]
                
            actual_data[address].extend(actual_labels)

    print(f"Reading predicted data from: {predicted_file_path}...")
    with open(predicted_file_path, 'r', encoding='utf-8-sig') as f_pred:
        for line in f_pred:
            parts = line.strip().split('\t')
            if len(parts) < 3: continue
            
            # The predicted format is: 0 \t Matthew 1 1 \t Word \t Prediction
            # So parts[1] is 'Matthew 1 1'
            address_string = parts[1].strip()
            
            # We split 'Matthew 1 1' by spaces
            addr_parts = address_string.split(' ')
            
            # Handle book names with numbers or spaces (like "1 John" or "Song of Solomon")
            # The last two items are always chapter and verse. Everything before is the book.
            if len(addr_parts) >= 3:
                verse = addr_parts[-1]
                chapter = addr_parts[-2]
                book = " ".join(addr_parts[:-2])
                address = (book, chapter, verse)
                
                # The prediction is always the very last item on the line
                pred_label = parts[-1].strip()
                predicted_data[address].append(pred_label)

    # 3. Align and flatten the data into 1-to-1 lists
    y_true = []
    y_pred = []
    mismatch_count = 0
    missing_verses = 0
    
    print("Aligning verses and evaluating...")
    for address, actual_labels in actual_data.items():
        if address not in predicted_data:
            missing_verses += 1
            continue 
            
        pred_labels = predicted_data[address]
        
        if len(actual_labels) != len(pred_labels):
            mismatch_count += 1
            min_len = min(len(actual_labels), len(pred_labels))
            y_true.extend(actual_labels[:min_len])
            y_pred.extend(pred_labels[:min_len])
        else:
            y_true.extend(actual_labels)
            y_pred.extend(pred_labels)
            
    if missing_verses > 0:
        print(f"\n[!] Alert: {missing_verses} verses from the actual data were NOT found in the predicted data.")
    if mismatch_count > 0:
        print(f"[!] Warning: Found {mismatch_count} verses with mismatched word counts. Only matching lengths evaluated.")

    # 4. Generate the Metrics Report
    if not y_true or not y_pred:
        print("\n❌ CRITICAL: No data matched! Check the file formats again.")
        return

    print("\n" + "=" * 60)
    print(" EVALUATION METRICS: Syntactic Structure (Y/Ỹ/Ỵ) ")
    print("=" * 60)
    
    labels_set = sorted(list(set(y_true + y_pred)))
    report = classification_report(y_true, y_pred, labels=labels_set, digits=4)
    print(report)

if __name__ == "__main__":
    main()
    