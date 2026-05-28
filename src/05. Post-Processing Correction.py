import os

def correct_sequence(tags):
    """
    Applies the 4-Tag Look-Ahead Matrix to correct structural hallucinations.
    tags: A list of sequence labels (e.g., ['Ỵ', 'Y', 'Ỹ', ...])
    Applies structural correction based on 4-Tag Look-Ahead Matrix.
    """
    if not tags:
        return tags
        
    # ---------------------------------------------------------
    # RULE 0: The End-of-Verse Anchor
    # ---------------------------------------------------------
    # A verse must always terminate with an End/Standalone tag.
    # Because Ỵ accepts Y, Ỹ, and Ỵ equally, this never breaks the preceding tag.
    if tags[-1] != 'Ỵ':
        tags[-1] = 'Ỵ'

    i = 0
    n = len(tags)
    
    while i < n:
        T = tags[i]
        L = tags[i-1] if i > 0 else 'Ỵ' # Treat out-of-bounds left as boundary
        
        # Out-of-bounds tags are treated as 'Ỵ' (End of sequence acts as a boundary)
        R = tags[i+1] if i+1 < n else 'Ỵ' # Treat out-of-bounds right as boundary
        ER = tags[i+2] if i+2 < n else 'Ỵ'

        # ---------------------------------------------------------
        # RULE 4: Start-of-Verse Collision [START, Ỹ, R, ER]
        # ---------------------------------------------------------
        if i == 0 and T == 'Ỹ':
            if R in ['Ỹ', 'Ỵ']:
                tags[i] = 'Y'
            elif R == 'Y':
                if ER in ['Ỹ', 'Ỵ']:
                    tags[i] = 'Y'
                    tags[i+1] = 'Ỹ'  # Double Flip
                else:
                    tags[i] = 'Ỵ'    # Forced Fallback

        # ---------------------------------------------------------
        # RULE 1: The Ỵ Ỹ Collision [Ỵ, Ỹ, R, ER]
        # ---------------------------------------------------------
        elif L == 'Ỵ' and T == 'Ỹ':
            if R in ['Ỹ', 'Ỵ']:
                tags[i] = 'Y'
            elif R == 'Y':
                if ER in ['Ỹ', 'Ỵ']:
                    tags[i] = 'Y'
                    tags[i+1] = 'Ỹ'  # Double Flip
                else:
                    tags[i] = 'Ỵ'    # Forced Fallback

        # ---------------------------------------------------------
        # RULE 2: The Y Y Collision [Y, Y, R, ER]
        # ---------------------------------------------------------
        elif L == 'Y' and T == 'Y':
            if R in ['Ỹ', 'Ỵ']:
                tags[i] = 'Ỹ'
            elif R == 'Y':
                if ER in ['Ỹ', 'Ỵ']:
                    tags[i] = 'Ỹ'
                    tags[i+1] = 'Ỹ'  # Double Flip
                else:
                    tags[i] = 'Ỵ'    # Forced Fallback

        # ---------------------------------------------------------
        # RULE 3: The Ỹ Y Collision [Ỹ, Y, R, ER]
        # ---------------------------------------------------------
        elif L == 'Ỹ' and T == 'Y':
            if R in ['Ỹ', 'Ỵ']:
                tags[i] = 'Ỹ'
            elif R == 'Y':
                if ER in ['Ỹ', 'Ỵ']:
                    tags[i] = 'Ỹ'
                    tags[i+1] = 'Ỹ'  # Double Flip
                else:
                    tags[i] = 'Ỵ'    # Forced Fallback

        # Move to the next tag. 
        # If we did a double-flip, the next iteration will check the newly 
        # flipped tag against its neighbors, guaranteeing perfect integration.
        i += 1
        
    return tags

def main():
    # ---------------------------------------------------------
    # CONFIGURATION
    # ---------------------------------------------------------
    INPUT_FILE = f"../sp_new_data/fileB__NT_predicted_2026-05-26_subphrases_ep440_reformatted"
    OUTPUT_FILE = f"../sp_new_data/fileB__NT_predicted_2026-05-26_subphrases_ep440_reformatted_CLEANED"

    if not os.path.exists(INPUT_FILE):
        print(f"Error: Could not find {INPUT_FILE}")
        return

    print(f"Reading and analyzing: {INPUT_FILE}...")
    
    corrections_made = 0
    total_lines = 0

    with open(INPUT_FILE, 'r', encoding='utf-8') as infile, \
         open(OUTPUT_FILE, 'w', encoding='utf-8') as outfile:
        
        for line in infile:
            total_lines += 1
            line = line.strip()
            if not line:
                continue
                
            parts = line.split('\t')
            if len(parts) >= 4:
                book = parts[0]
                chapter = parts[1]
                verse = parts[2]
                
                # Split the tags into a list for processing
                original_tags = parts[3].split()
                
                # Create a copy to track if modifications were made
                tags_to_fix = original_tags.copy()
                
                # Apply the Look-Ahead Rules
                cleaned_tags = correct_sequence(tags_to_fix)
                
                if cleaned_tags != original_tags:
                    corrections_made += 1
                
                # Reconstruct the line and write to output
                cleaned_line = f"{book}\t{chapter}\t{verse}\t{' '.join(cleaned_tags)}\n"
                outfile.write(cleaned_line)
            else:
                # If a line is somehow misformatted, just write it back as is
                outfile.write(line + "\n")

    print(f"✅ Processing Complete!")
    print(f"-> Total verses processed: {total_lines}")
    print(f"-> Verses successfully corrected: {corrections_made}")
    print(f"-> Saved cleaned dataset to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()