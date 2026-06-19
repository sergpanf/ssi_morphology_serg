import os

def merge_feature_files(file_a_path: str, file_b_path: str, output_path: str) -> None:
    """
    Merges morphological word features (File A) with subphrase tags (File B).
    Appends '¥:{tag}' to each corresponding word.
    """
    
    if not os.path.exists(file_a_path) or not os.path.exists(file_b_path):
        print("Error: One or both input files could not be found.")
        return

    # 1. Load File B tags into a dictionary keyed by verse address
    tags_dict = {}
    with open(file_b_path, 'r', encoding='utf-8') as fb:
        for line in fb:
            parts = line.strip('\n').split('\t')
            if len(parts) >= 4:
                address = f"{parts[0]}\t{parts[1]}\t{parts[2]}"
                tags = parts[3].strip().split()
                tags_dict[address] = tags

    mismatches = 0
    missing_addresses = 0
    processed_lines = 0

    # 2. Read File A, merge with tags, and write to Output File
    with open(file_a_path, 'r', encoding='utf-8') as fa, \
         open(output_path, 'w', encoding='utf-8') as fout:
        
        for line in fa:
            parts = line.strip('\n').split('\t')
            
            # Ensure the line has the minimum 4 columns
            if len(parts) >= 4:
                address = f"{parts[0]}\t{parts[1]}\t{parts[2]}"
                words = parts[3].strip().split()
                
                if address in tags_dict:
                    tags = tags_dict[address]
                    
                    # Safety Check: Do we have a 1-to-1 mapping?
                    if len(words) != len(tags):
                        print(f"⚠️ Warning: Mismatch at {address}. Words: {len(words)} | Tags: {len(tags)}")
                        mismatches += 1
                    
                    # Merge using zip (stops at the length of the shorter list if mismatched)
                    merged_words = [f"{word}¥:{tag}" for word, tag in zip(words, tags)]
                    
                    # Reconstruct the line
                    merged_line = f"{address}\t{' '.join(merged_words)}\n"
                    fout.write(merged_line)
                    processed_lines += 1
                else:
                    print(f"⚠️ Warning: Address {address} found in File A but missing in File B.")
                    missing_addresses += 1
                    # Write original line without the new feature if tag data is missing
                    fout.write(line) 
            else:
                # Write empty or malformed lines as-is
                fout.write(line)

    # 3. Final Report
    print("-" * 40)
    print("✅ Merge Complete!")
    print(f"Total verses processed:  {processed_lines}")
    print(f"Verses with mismatches:  {mismatches}")
    print(f"Missing tag addresses:   {missing_addresses}")
    print(f"Output saved to:         {output_path}")

if __name__ == "__main__":
    # --- CONFIGURATION ---
    FILE_A = f"../sp_data/fileA__NT"
    FILE_B = f"../sp_data/fileB__NT"
    OUTPUT = "../sp_data/fileA_B_merged"
    
    merge_feature_files(FILE_A, FILE_B, OUTPUT)