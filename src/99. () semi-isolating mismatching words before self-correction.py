import random

def main():
    # ---------------------------------------------------------
    # CONFIGURATION
    # ---------------------------------------------------------
    ACTUAL_FILE = "../sp_data/fileB__NT"
    PREDICTED_FILE = "../sp_new_data/fileB__NT_predicted_2026-04-03_subphrases_ep40_reformatted"

    OUT_ACTUAL = "../sp_selfCorrect_structures/context_train_actual.txt"
    OUT_PREDICTED = "../sp_selfCorrect_structures/context_train_predicted.txt"

    print("Initializing Word-Level Context Extractor...")
    generate_contextual_dataset(ACTUAL_FILE, PREDICTED_FILE, OUT_ACTUAL, OUT_PREDICTED)

# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------
def generate_contextual_dataset(actual_file_path: str, predicted_file_path: str, output_actual: str, output_predicted: str):
    actual_data = {}
    predicted_data = {}

    def parse_line(line):
        parts = line.strip().split('\t') if '\t' in line else line.strip().split()
        if len(parts) >= 4:
            address = (parts[0].strip(), parts[1].strip(), parts[2].strip())
            
            # --- THE FIX ---
            # Grab the tags, join them just in case they got split weirdly, 
            # and then split() them by spaces to guarantee a list of individual letters.
            tags_string = " ".join(parts[3:])
            content = tags_string.split() 
            # ----------------
            
            return address, content
        return None, None

    print(f"Reading actual data from: {actual_file_path}...")
    with open(actual_file_path, 'r', encoding='utf-8-sig') as fa:
        for line in fa:
            addr, content = parse_line(line)
            if addr:
                actual_data[addr] = content

    print(f"Reading predicted data from: {predicted_file_path}...")
    with open(predicted_file_path, 'r', encoding='utf-8-sig') as fp:
        for line in fp:
            addr, content = parse_line(line)
            if addr:
                predicted_data[addr] = content

    print("Scanning word-by-word and extracting 10-word windows...")
    
    # We use 'sets' to automatically prevent duplicate context windows
    mismatches = set()
    matches = set()

    for addr, actual_labels in actual_data.items():
        if addr not in predicted_data:
            continue
            
        pred_labels = predicted_data[addr]
        
        # Ensure we don't crash if the lengths mismatched slightly
        min_len = min(len(actual_labels), len(pred_labels))
        
        # Find every specific index where an error occurred
        error_indices = [i for i in range(min_len) if actual_labels[i] != pred_labels[i]]

        if not error_indices:
            # HEALTHY VERSE: Slice into clean 10-word chunks
            for i in range(0, min_len, 10):
                start = i
                end = i + 10
                
                # If this chunk hits the end of the verse and is less than 10 words,
                # slide the window backwards so it is exactly 10 words long!
                if end > min_len:
                    end = min_len
                    start = max(0, end - 10)
                    
                act_win = tuple(actual_labels[start:end])
                pred_win = tuple(pred_labels[start:end])
                matches.add((addr, act_win, pred_win))
        else:
            # FAILED VERSE: Surgically extract exactly 10 words centered on EACH error
            for e_idx in error_indices:
                # Start 4 words before the error
                start = max(0, e_idx - 4)
                end = start + 10
                
                # If the window pushes past the end of the verse, pull it back
                if end > min_len:
                    end = min_len
                    start = max(0, end - 10)

                act_win = tuple(actual_labels[start:end])
                pred_win = tuple(pred_labels[start:end])
                mismatches.add((addr, act_win, pred_win))

    # Convert sets back to lists so we can shuffle them
    mismatches = list(mismatches)
    matches = list(matches)

    print(f"Found {len(mismatches)} unique Error Context Windows.")
    print(f"Found {len(matches)} Healthy Context Windows.")

    # --- DYNAMIC 1:1 BALANCING ---
    target_size = len(mismatches)
    
    if len(matches) > target_size:
        print(f"Sampling {target_size} healthy windows to achieve a perfect 1:1 ratio...")
        sampled_matches = random.sample(matches, target_size)
    else:
        print(f"Using all {len(matches)} available healthy windows...")
        sampled_matches = matches

    # Combine and Shuffle
    final_dataset = mismatches + sampled_matches
    print("Shuffling the balanced dataset...")
    random.shuffle(final_dataset)

    # Write out the new training files
    with open(output_actual, 'w', encoding='utf-8') as out_act, \
         open(output_predicted, 'w', encoding='utf-8') as out_pred:

        for addr, act_win, pred_win in final_dataset:
            # Reconstruct the line: Book \t Chapter \t Verse \t Tag1 Tag2 Tag3...
            book, chapter, verse = addr
            
            act_line = f"{book}\t{chapter}\t{verse}\t{' '.join(act_win)}\n"
            pred_line = f"{book}\t{chapter}\t{verse}\t{' '.join(pred_win)}\n"
            
            out_act.write(act_line)
            out_pred.write(pred_line)

    print(f"✅ Success! Created a hyper-focused training dataset with {len(final_dataset)} total sequences.")
    print(f"-> Saved context actual data to:    {output_actual}")
    print(f"-> Saved context predicted data to: {output_predicted}")

if __name__ == "__main__":
    main()