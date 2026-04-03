import collections

def main():
    # Point these to your actual files
    INPUT_PREDICTED_FILE = "../sp_new_data/fileB__NT_predicted_2026-04-03_subphrases_ep40"
    OUTPUT_FILE = "../sp_new_data/fileB__NT_predicted_2026-04-03_subphrases_ep40_reformatted"
    
    convert_predicted_to_actual_format(INPUT_PREDICTED_FILE, OUTPUT_FILE)

def convert_predicted_to_actual_format(predicted_file_path: str, output_file_path: str):
    # Dictionary to hold the sequence of predicted letters for each verse
    # Key: (Book, Chapter, Verse) -> Value: List of predicted letters
    grouped_predictions = collections.defaultdict(list)
    
    print(f"Reading predicted data from: {predicted_file_path}...")
    
    with open(predicted_file_path, 'r', encoding='utf-8-sig') as f_pred:
        for line in f_pred:
            parts = line.strip().split('\t')
            # Ensure the line has enough parts to process
            if len(parts) < 3: 
                continue
            
            # The predicted format is: 0 \t Matthew 1 1 \t Word \t Prediction
            # parts[1] contains the full address 'Matthew 1 1'
            address_string = parts[1].strip()
            addr_parts = address_string.split(' ')
            
            # Safely extract Book, Chapter, and Verse
            # Handles books with spaces/numbers like "1 John" or "Song of Solomon"
            if len(addr_parts) >= 3:
                verse = addr_parts[-1]
                chapter = addr_parts[-2]
                book = " ".join(addr_parts[:-2])
                
                address_key = (book, chapter, verse)
                
                # The prediction is always the very last item on the line
                pred_label = parts[-1].strip()
                
                # Add the predicted letter to this verse's sequence
                grouped_predictions[address_key].append(pred_label)

    print(f"Writing reformatted data to: {output_file_path}...")
    
    # Write the grouped data to the new output file
    with open(output_file_path, 'w', encoding='utf-8') as f_out:
        for (book, chapter, verse), predictions in grouped_predictions.items():
            # Join the individual letters with a space
            joined_predictions = " ".join(predictions)
            
            # Format: Book \t Chapter \t Verse \t Y Ỹ Ỵ ...
            output_line = f"{book}\t{chapter}\t{verse}\t{joined_predictions}\n"
            f_out.write(output_line)

    print(f"✅ Successfully converted {len(grouped_predictions)} verses!")

if __name__ == "__main__":
    main()
    