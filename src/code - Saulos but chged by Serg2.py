# %% initialize torch and CUDA
import torch
torch.cuda.is_available()

from main import main

# %% initialize TF and related libraries
from tf.app import use
import re
import os
import matplotlib.pyplot as plt
import numpy as np



# %% initialize TF datasets
# SP = use("DT-UCPH/sp", version="3.4.1", hoist=globals())
# BHS = use("etcbc/bhsa", version=2021, hoist=globals())

SP = use('DT-UCPH/sp', version='3.4.1')
Fsp, Lsp, Tsp = SP.api.F, SP.api.L, SP.api.T
MT = use('etcbc/bhsa', version='2021')
Fmt, Lmt, Tmt = MT.api.F, MT.api.L, MT.api.T

# %% testing MT and SP (depending on the switch)

i=0
file_input=[]

F, L, T = Fmt, Lmt, Tmt  # Use MT's API
# F, L, T = Fsp, Lsp, Tsp  # Use SP's API

for verse in F.otype.s('verse'):
    text = "".join([F.g_cons.v(word) if not F.trailer.v(word) else F.g_cons.v(word)+" " for word in L.d(verse,'word')]).replace("_", " ")
    bo, ch, ve = T.sectionFromNode(verse)
    final = "\t".join([bo, str(ch), str(ve), text.strip()])

    if bo == 'Genesis' and str(ch) == '7' and str(ve) == '16':
        if i<10:
            print(final)
        i=i+1

    if bo == 'Genesis':
        file_input.append(final)

    if i<3:
       print(final)
    i=i+1

with open('../data/input', 'w', encoding='utf-8') as file:
    for line in file_input:
        file.write(line + '\n')
    
    
# %% testing MT and SP (2) this prepared only for MT, not SP

i=0
file_input=[]

F, L, T = Fmt, Lmt, Tmt  # Use MT's API
# F, L, T = Fsp, Lsp, Tsp  # Use SP's API

for verse in Fmt.otype.s('verse'):
    verse_text = ""
    cl_atoms = L.d(verse,'clause_atom')
    for clause_atom in cl_atoms:
        #clause_atom_text = "".join([F.g_cons.v(word) if not F.trailer.v(word) else F.g_cons.v(word) + " " for word in L.d(clause_atom, 'word')]).replace("_", " ").strip() + "| "
        clause_atom_text = "".join([F.g_cons.v(word) + (" " if F.trailer.v(word) else "") for word in L.d(clause_atom, 'word')]).replace("_", " ").strip()
        clause_atom_text += "|" if clause_atom_text == 'W' else "| "
        
        # Genesis	39	23	>JN FR BJT HSHR R>H >T KL M>WMH BJDW| B>CR JHWH >TW| W|>CR HW> <FH| JHWH MYLJX|

        text = []
        for word in L.d(clause_atom, 'word'):
            if not F.trailer.v(word):
                text.append(F.g_cons.v(word))
            else:
                text.append(F.g_cons.v(word) + " ")
        clause_atom_text = "".join(text)
        clause_atom_text = clause_atom_text.replace("_"," ")
        clause_atom_text = clause_atom_text.strip()
        if clause_atom_text == 'W':
            clause_atom_text += "|"
        else:
            clause_atom_text += "| "
        
        verse_text = verse_text + clause_atom_text
    verse_text = verse_text.strip()
    bo, ch, ve = T.sectionFromNode(verse)
    final = "\t".join([bo, str(ch), str(ve), verse_text.strip()])

    if bo == 'Genesis' and str(ch) == '7' and str(ve) == '16':
        if i<10:
            print(final)
        i=i+1
    
    if bo == 'Genesis':
        file_input.append(final)

with open('../data/output', 'w', encoding='utf-8') as file:
    for line in file_input:
        file.write(line + '\n')

# %% TRAINING without EVAL flag (via MAIN.py)

# sp: good values of Saulo: ep=15 (or 30), l=10 (or 7), lr=0.0001, et=True
main(["-mo", "train", "-i", "input_NT_normalized", "-o", "output_NT_normalized_XYs", "-ep", "1", "-l", "1", "-lr", "0.0001"])
# main(["-mo", "train", "-i", "input_III_John_normalized", "-o", "output_III_John_normalized_XYs", "-ep", "2", "-l", "10", "-lr", "0.0001"])


# %% TRAINING with EVAL flag

main(["-mo", "train", "-i", "input_NT_NLCM", "-o", "output_NT_nmt_llt_XYs", "-ep", "20", "-l", "10", "-lr", "0.0001", "-et", "True"]) # useing older output file, as XY-phrase-pattern will stay the same and the word count of the input file stays equal with it


# %% F-SCORE of the MODEL
file = "../sp_evaluation_results_transformer/input_NT_NLCM_output_NT_nmt_llt_XYs_ONE_DATASET/results_10seq_len_0.0001lr_512embsize_8nhead_transformer_0.1dropout_128_batchsize_20epochs_3beamsize.txt"

TP = 0
FP = 0
FN = 0
TN = 0

with open(file, 'r') as f:
    lines = f.readlines()

    lines_count = len(lines)-3 # =39269; "-3" is because the comments at the end takes up 3 lines. The last empty line is not retreived
    # pair_count = int(float(lines_count)/2) # number of line pairs (=19635)

    # print (f"Number of line pairs = {n}; last line nr = {2*n}")
    print (f"last line nr = {lines_count}")

    for line_predicted_nr in range(1, lines_count, 2):

        line_truevalue_nr = line_predicted_nr+1

        predicted = lines[line_predicted_nr-1].strip()
        predicted = predicted[10:].replace(" ","")
        
        truevalue = lines[line_truevalue_nr-1].strip()
        truevalue = truevalue[10:].replace(" ","")

        length = len(predicted)

        for j in range(0,length):
            if predicted[j] == truevalue[j] and truevalue[j] == 'X':
                TN = TN+1
            if predicted[j] == truevalue[j] and truevalue[j] == 'Y':
                TP = TP+1
            if predicted[j] != truevalue[j] and truevalue[j] == 'Y':
                FN = FN+1
            if predicted[j] != truevalue[j] and truevalue[j] == 'X':
                FP = FP+1
         
        if line_predicted_nr < 5 or line_predicted_nr > lines_count-5:
            print(f"line number {line_predicted_nr} predicted {predicted}")   
            print(f"line number {line_truevalue_nr} truevalue {truevalue}")   
    
print(f"TP = {TP} --- FP = {FP}\nFN = {FN} --- TN = {TN}")

precision = TP/(TP+FP)
recall = TP/(TP+FN)
fscore = (2 * precision * recall) / (precision + recall)

print(f"Precision = {precision}\nRecall = {recall}\nFscore = {fscore}")

# %% [markdown]
# Our model only predicts `X` and not `Y`. Maybe there is not data enough that it can show some good result.

# %%
file = "../evaluation_results_transformer/input_gn_ex_outputX_gn_ex_ONE_DATASET/results_5seq_len_0.0001lr_512embsize_8nhead_transformer_0.1dropout_128_batchsize_5epochs_3beamsize.txt"

n = 11004

n = int(float(n)/2)

TP = 0
FP = 0
FN = 0
TN = 0

length=5

with open(file, 'r') as f:
    lines = f.readlines()

    for i in range(0, n):
        predicted = lines[2*i].strip()
        predicted = predicted[10:].replace("X|","Y").replace(" ","")
        
        truevalue = lines[2*i+1].strip()
        truevalue = truevalue[10:].replace("X|","Y").replace(" ","")

        for j in range(0,length):
            if predicted[j] == truevalue[j] and truevalue[j] == 'X':
                TN = TN+1
            if predicted[j] == truevalue[j] and truevalue[j] == 'Y':
                TP = TP+1
            if predicted[j] != truevalue[j] and truevalue[j] == 'Y':
                FN = FN+1
            if predicted[j] != truevalue[j] and truevalue[j] == 'X':
                FP = FP+1
         
        if i < 10:
            print(i, predicted)   
    
print(f"TP = {TP} --- FP = {FP}\nFN = {FN} --- TN = {TN}")

precision = TP/(TP+FP)
recall = TP/(TP+FN)
fscore = (2 * precision * recall) / (precision + recall)

print(f"Precision = {precision}\nRecall = {recall}\nFscore = {fscore}")

# %%
import matplotlib.pyplot as plt
import numpy as np

# %%
file = "../data/outputY_gn"

cont_x = 0
cont=0

list_cont = []

with open(file, 'r') as f:
    for line in f:
        for char in line:
            if char == 'X':
                cont_x += 1
            elif char == 'Y':
                cont = cont_x
                cont_x = 0
                cont = str(cont)
                for amount in cont:
                      list_cont.append(int(amount))
        cont_x = 0

plt.hist(list_cont, bins=10)

plt.xticks(ticks=np.arange(0, 10), labels=np.arange(0, 10))
plt.xlabel("X count before Y")
plt.ylabel("Frequency")
plt.title("Histogram of X count before Y (Genesis)")
plt.grid(True)

plt.show()

# %%
i=0
file_input=[]

#pattern = r"\b\w+\b(?!\|)"
pattern = r"[^\s|]+"

for verse in F.otype.s('verse'):
    verse_text = ""
    cl_atoms = L.d(verse,'clause_atom')
    for clause_atom in cl_atoms:
        #clause_atom_text = "".join([F.g_cons.v(word) + (" " if F.trailer.v(word) else "") for word in L.d(clause_atom, 'word')]).replace("_", " ").strip()
        #clause_atom_text += "|" if clause_atom_text == 'W' else "| "

        text = []
        for word in L.d(clause_atom, 'word'):
            if not F.trailer.v(word):
                text.append(F.g_cons.v(word))
            else:
                text.append(F.g_cons.v(word) + " ")
            
        clause_atom_text = "".join(text)
        clause_atom_text = clause_atom_text.replace("_"," ")
        clause_atom_text = clause_atom_text.strip()

        if F.g_cons.v(word) in ['W','H'] and F.trailer.v(word) == "":
            clause_atom_text += "|"
        else:
            clause_atom_text += "| "
            
        #Replacing the words in the output file as 'X' 
        #if clause_atom_text in ['W','H']:
        #    clause_atom_text += "|"
        #else:
        #    clause_atom_text += "| "

        clause_atom_text = re.sub(pattern, "X", clause_atom_text)
        clause_atom_text = clause_atom_text.replace("X|", "Y")
        
        verse_text = verse_text + clause_atom_text
    verse_text = verse_text.strip()
    bo, ch, ve = T.sectionFromNode(verse)
    final = "\t".join([bo, str(ch), str(ve), verse_text.strip()])

    if bo == 'Genesis' and str(ch) == '7' and str(ve) == '16':
        if i<10:
            print(final)
        i=i+1
    
    if bo in ['Genesis', 'Exodus']:
        file_input.append(final)

        with open('../data/outputY_gn_ex', 'w', encoding='utf-8') as file:
            for line in file_input:
                file.write(line + '\n')

# %%
inputfilePath = "../data/input_gn_ex"
outputfilePath = "../data/outputY_gn_ex"

for i in range(1,2746):
    word_count_line_input = count_words_in_line(inputfilePath, i)
    word_count_line_output = count_words_in_line(outputfilePath, i)

    if word_count_line_input != word_count_line_output:
        print(f"Line {i} of file {inputfilePath} contains {word_count_line_input} words.")
        print(f"Line {i} of file {outputfilePath} contains {word_count_line_output} words.\n")

    i=i+1

# %%
file = "../data/outputY_gn_ex"

cont_x = 0
cont=0

list_cont = []

with open(file, 'r') as f:
    for line in f:
        for char in line:
            if char == 'X':
                cont_x += 1
            elif char == 'Y':
                cont = cont_x
                cont_x = 0
                cont = str(cont)
                for amount in cont:
                      list_cont.append(int(amount))
        cont_x = 0

plt.hist(list_cont, bins=10)

plt.xticks(ticks=np.arange(0, 10), labels=np.arange(0, 10))
plt.xlabel("X count before Y")
plt.ylabel("Frequency")
plt.title("Histogram of X count before Y (Genesis+Exodus)")
plt.grid(True)

plt.show()

# %%
!python main.py -mo=train -i=input_gn_ex -o=outputY_gn_ex -ep=5 -l=3 -lr=0.0001 -et=True

# %% [markdown]
# Time of execution = 49 min

# %%
file = "../evaluation_results_transformer/input_gn_ex_outputY_gn_ex_ONE_DATASET/results_3seq_len_0.0001lr_512embsize_8nhead_transformer_0.1dropout_128_batchsize_5epochs_3beamsize.txt"

n = 10656

n = int(float(n)/2)

TP = 0
FP = 0
FN = 0
TN = 0

length=3

with open(file, 'r') as f:
    lines = f.readlines()

    for i in range(0, n):
        predicted = lines[2*i].strip()
        predicted = predicted[10:].replace("X|","Y").replace(" ","")
        
        truevalue = lines[2*i+1].strip()
        truevalue = truevalue[10:].replace("X|","Y").replace(" ","")

        for j in range(0,length):
            if predicted[j] == truevalue[j] and truevalue[j] == 'X':
                TN = TN+1
            if predicted[j] == truevalue[j] and truevalue[j] == 'Y':
                TP = TP+1
            if predicted[j] != truevalue[j] and truevalue[j] == 'Y':
                FN = FN+1
            if predicted[j] != truevalue[j] and truevalue[j] == 'X':
                FP = FP+1
    
print(f"TP = {TP} --- FP = {FP}\nFN = {FN} --- TN = {TN}")

precision = TP/(TP+FP)
recall = TP/(TP+FN)
fscore = (2 * precision * recall) / (precision + recall)

print(f"Precision = {precision}\nRecall = {recall}\nFscore = {fscore}")

# %%
!python main.py -mo=train -i=input_gn_ex -o=outputY_gn_ex -ep=5 -l=6 -lr=0.0001 -et=True

# %% [markdown]
# Time of execution = 1:37 min

# %% [markdown]
# #### June 2 experiment - Google Colab

# %% [markdown]
# Time of execution ~ 1:30 min

# %% [markdown]
# ![image.png](attachment:8303e45c-9b19-4fbd-9ad1-f6b0023bfa96.png)

# %% [markdown]
# #### June 6 experiment - AUSS's Mac

# %% [markdown]
# Time of execution ~ 1 h

# %% [markdown]
# <div>
# <img src="attachment:99f6b779-620f-48d4-ac9f-b3abd1224ace.png" width="800">
# </div>

# %% [markdown]
# #### June 10 experiment - MacBook Pro

# %% [markdown]
# Using CPU

# %%
%%time
!python main.py -mo=train -i=input_gn_ex -o=outputY_gn_ex -ep=5 -l=6 -lr=0.0001 -et=True

# %% [markdown]
# Using GPU

# %%
%%time
!python main.py -mo=train -i=input_gn_ex -o=outputY_gn_ex -ep=5 -l=6 -lr=0.0001 -et=True

# %%
file = "../evaluation_results_transformer/input_gn_ex_outputY_gn_ex_ONE_DATASET/results_6seq_len_0.0001lr_512embsize_8nhead_transformer_0.1dropout_128_batchsize_5epochs_3beamsize.txt"

n = 10344

n = int(float(n)/2)

TP = 0
FP = 0
FN = 0
TN = 0

length=6

with open(file, 'r') as f:
    lines = f.readlines()

    for i in range(0, n):
        predicted = lines[2*i].strip()
        predicted = predicted[10:].replace("X|","Y").replace(" ","")
        
        truevalue = lines[2*i+1].strip()
        truevalue = truevalue[10:].replace("X|","Y").replace(" ","")

        for j in range(0,length):
            if predicted[j] == truevalue[j] and truevalue[j] == 'X':
                TN = TN+1
            if predicted[j] == truevalue[j] and truevalue[j] == 'Y':
                TP = TP+1
            if predicted[j] != truevalue[j] and truevalue[j] == 'Y':
                FN = FN+1
            if predicted[j] != truevalue[j] and truevalue[j] == 'X':
                FP = FP+1
    
print(f"TP = {TP} --- FP = {FP}\nFN = {FN} --- TN = {TN}")

precision = TP/(TP+FP)
recall = TP/(TP+FN)
fscore = (2 * precision * recall) / (precision + recall)

print(f"Precision = {precision}\nRecall = {recall}\nFscore = {fscore}")

# %% [markdown]
# ## June 9 experiment

# %% [markdown]
# Creating dataset - the entire Masoretic Text

# %%
file_input=[]
bo_old=None

for verse in F.otype.s('verse'):
    text = "".join([F.g_cons.v(word) if not F.trailer.v(word) else F.g_cons.v(word)+" " for word in L.d(verse,'word')]).replace("_", " ")
    bo, ch, ve = T.sectionFromNode(verse)
    final = "\t".join([bo, str(ch), str(ve), text.strip()])

    if bo != bo_old:
        print(bo)
        bo_old=bo
    
    file_input.append(final)

    with open('../data/input_MT', 'w', encoding='utf-8') as file:
        for line in file_input:
            file.write(line + '\n')

# %%
file_input=[]
bo_old=None
i=0

pattern = r"[^\s|]+"

for verse in F.otype.s('verse'):
    verse_text = ""
    cl_atoms = L.d(verse,'clause_atom')
    for clause_atom in cl_atoms:
        #clause_atom_text = "".join([F.g_cons.v(word) + (" " if F.trailer.v(word) else "") for word in L.d(clause_atom, 'word')]).replace("_", " ").strip()
        #clause_atom_text += "|" if clause_atom_text == 'W' else "| "

        text = []
        for word in L.d(clause_atom, 'word'):
            if not F.trailer.v(word):
                text.append(F.g_cons.v(word))
            else:
                text.append(F.g_cons.v(word) + " ")
            
        clause_atom_text = "".join(text)
        clause_atom_text = clause_atom_text.replace("_"," ")
        clause_atom_text = clause_atom_text.strip()

        if F.g_cons.v(word) in ['W','H'] and F.trailer.v(word) == "":
            clause_atom_text += "|"
        elif F.g_cons.v(word) == '' and F.trailer.v(word) == "":
            clause_atom_text += "|"
        else:
            clause_atom_text += "| "

        clause_atom_text = re.sub(pattern, "X", clause_atom_text)
        clause_atom_text = clause_atom_text.replace("X|", "Y")
        
        verse_text = verse_text + clause_atom_text
    verse_text = verse_text.strip()
    bo, ch, ve = T.sectionFromNode(verse)
    final = "\t".join([bo, str(ch), str(ve), verse_text.strip()])

    if bo != bo_old:
        print(bo)
        bo_old=bo

    #if bo == 'Ezekiel' and str(ch) == '9' and str(ve) == '11':
    #    if i<10:
    #        print(final)
    #    i=i+1

    file_input.append(final)

    with open('../data/outputY_MT', 'w', encoding='utf-8') as file:
        for line in file_input:
            file.write(line + '\n')
#Ezekiel	9	11	WHNH| H>JC LBC HBDJM| >CR HQST BMTNJW| MCJB DBR| L>MR| <FJTJ K| >CR YWJTNJ|
#Ezekiel	9	11	X| X X X| X X X| X X| X| X X|X X|
#Ezekiel	9	11	Y X X Y X X Y X Y Y X Y X Y
#Ezekiel	9	11	Y X X Y X X Y X Y Y X YX Y

# %%
inputfilePath = "../data/input_MT"
outputfilePath = "../data/outputY_MT"

def count_words_in_line(file_path, line_number):
    """
    Counts the number of words in the specified line of a text file.

    Args:
        file_path (str): Path to the text file.
        line_number (int): Number of the line to count the words in.

    Returns:
        int: Number of words in the specified line.
    """

    with open(file_path, 'r') as f:
        lines = f.readlines()
    try:
        # Access the specified line
        line = lines[line_number - 1]
        # Remove leading and trailing whitespace from the line
        line = line.strip()
        # Split the line into words using whitespace as delimiters
        words = line.split()
        # Return the number of words
        return len(words)
    except IndexError:
        print(f"Error: Line {line_number} does not exist in the file.")
        return 0

with open(inputfilePath, 'r') as f:
    n_lines = len(f.readlines())

for i in range(1,n_lines):
    word_count_line_input = count_words_in_line(inputfilePath, i)
    word_count_line_output = count_words_in_line(outputfilePath, i)

    if word_count_line_input != word_count_line_output:
            print(f"Line {i} of file {inputfilePath} contains {word_count_line_input} words.")
            print(f"Line {i} of file {outputfilePath} contains {word_count_line_output} words.\n")
    
    i=i+1

# %%
%%time
file = "../data/outputY_MT"

cont_x = 0
cont=0

list_cont = []

with open(file, 'r') as f:
    for line in f:
        for char in line:
            if char == 'X':
                cont_x += 1
            elif char == 'Y':
                cont = cont_x
                cont_x = 0
                cont = str(cont)
                for amount in cont:
                      list_cont.append(int(amount))
        cont_x = 0

plt.hist(list_cont, bins=10)

plt.xticks(ticks=np.arange(0, 10), labels=np.arange(0, 10))
plt.xlabel("X count before Y")
plt.ylabel("Frequency")
plt.title("Histogram of X count before Y (Masoretic Text)")
plt.grid(True)

plt.show()

# %%
%%time
!python main.py -mo=train -i=input_MT -o=outputY_MT -ep=5 -l=6 -lr=0.0001 -et=True

# %%
file = "../evaluation_results_transformer/input_MT_outputY_MT_ONE_DATASET/results_6seq_len_0.0001lr_512embsize_8nhead_transformer_0.1dropout_128_batchsize_5epochs_3beamsize.txt"

with open(file, 'r') as f:
    line = f.readline()
    n = len(f.readlines())
    n = n-3 #correction in the number of lines since final lines does not contain data values

n = int(float(n)/2)

length=len(line.split())-1

TP = 0
FP = 0
FN = 0
TN = 0

with open(file, 'r') as f:
    lines = f.readlines()

    for i in range(0, n):
        predicted = lines[2*i].strip()
        predicted = predicted[10:].replace("X|","Y").replace(" ","")
        
        truevalue = lines[2*i+1].strip()
        truevalue = truevalue[10:].replace("X|","Y").replace(" ","")

        for j in range(0,length):
            if predicted[j] == truevalue[j] and truevalue[j] == 'X':
                TN = TN+1
            if predicted[j] == truevalue[j] and truevalue[j] == 'Y':
                TP = TP+1
            if predicted[j] != truevalue[j] and truevalue[j] == 'Y':
                FN = FN+1
            if predicted[j] != truevalue[j] and truevalue[j] == 'X':
                FP = FP+1
         
        #if i < 10:
        #    print(i, predicted)   
    
print(f"TP = {TP} --- FP = {FP}\nFN = {FN} --- TN = {TN}")

precision = TP/(TP+FP)
recall = TP/(TP+FN)
fscore = (2 * precision * recall) / (precision + recall)

print(f"Precision = {precision}\nRecall = {recall}\nFscore = {fscore}")

# %% [markdown]
# ### Issue with Ezekiel 9:11

# %%
!sed -n '12995p' < ../data/input_MT
!sed -n '12995p' < ../data/outputY_MT

# %% [markdown]
# ![image.png](attachment:ffbda646-513c-4bdb-b19e-bd11f6f2beb9.png)

# %%
results = BHS.search("""
book book=Ezechiel
    chapter chapter=9
        verse verse=11
            word gloss* g_cons=K trailer*
""")
BHS.show(results, end=4, multiFeatures=False, queryFeatures=True, condensed=True, hiddenTypes={'phrase', 'phrase_atom','subphrase', 'sentence', 'sentence_atom', 'clause_atom'})

# %% [markdown]
# ## June 14, 16 and 17 experiment

# %%
%%time
!python main.py -mo=train -i=input_MT -o=outputY_MT -ep=5 -l=10 -lr=0.0001 -et=True

# %%
file = "../evaluation_results_transformer/input_MT_outputY_MT_ONE_DATASET/results_10seq_len_0.0001lr_512embsize_8nhead_transformer_0.1dropout_128_batchsize_5epochs_3beamsize.txt"

with open(file, 'r') as f:
    line = f.readline()
    n = len(f.readlines())
    n = n-3 #correction in the number of lines since final lines does not contain data values

n = int(float(n)/2)

length=len(line.split())-1

TP = 0
FP = 0
FN = 0
TN = 0

with open(file, 'r') as f:
    lines = f.readlines()

    for i in range(0, n):
        predicted = lines[2*i].strip()
        predicted = predicted[10:].replace("X|","Y").replace(" ","")
        
        truevalue = lines[2*i+1].strip()
        truevalue = truevalue[10:].replace("X|","Y").replace(" ","")

        for j in range(0,length):
            if predicted[j] == truevalue[j] and truevalue[j] == 'X':
                TN = TN+1
            if predicted[j] == truevalue[j] and truevalue[j] == 'Y':
                TP = TP+1
            if predicted[j] != truevalue[j] and truevalue[j] == 'Y':
                FN = FN+1
            if predicted[j] != truevalue[j] and truevalue[j] == 'X':
                FP = FP+1 
    
print(f"TP = {TP} --- FP = {FP}\nFN = {FN} --- TN = {TN}")

precision = TP/(TP+FP)
recall = TP/(TP+FN)
fscore = (2 * precision * recall) / (precision + recall)

print(f"Precision = {precision}\nRecall = {recall}\nFscore = {fscore}")

# %%
%%time
!python main.py -mo=train -i=input_MT -o=outputY_MT -ep=10 -l=6 -lr=0.0001 -et=True

# %%
file = "../evaluation_results_transformer/input_MT_outputY_MT_ONE_DATASET/results_6seq_len_0.0001lr_512embsize_8nhead_transformer_0.1dropout_128_batchsize_10epochs_3beamsize.txt"

with open(file, 'r') as f:
    line = f.readline()
    n = len(f.readlines())
    n = n-3 #correction in the number of lines since final lines does not contain data values

n = int(float(n)/2)

length=len(line.split())-1

TP = 0
FP = 0
FN = 0
TN = 0

with open(file, 'r') as f:
    lines = f.readlines()

    for i in range(0, n):
        predicted = lines[2*i].strip()
        predicted = predicted[10:].replace("X|","Y").replace(" ","")
        
        truevalue = lines[2*i+1].strip()
        truevalue = truevalue[10:].replace("X|","Y").replace(" ","")

        for j in range(0,length):
            if predicted[j] == truevalue[j] and truevalue[j] == 'X':
                TN = TN+1
            if predicted[j] == truevalue[j] and truevalue[j] == 'Y':
                TP = TP+1
            if predicted[j] != truevalue[j] and truevalue[j] == 'Y':
                FN = FN+1
            if predicted[j] != truevalue[j] and truevalue[j] == 'X':
                FP = FP+1
         
        #if i < 10:
        #    print(i, predicted)   
    
print(f"TP = {TP} --- FP = {FP}\nFN = {FN} --- TN = {TN}")

precision = TP/(TP+FP)
recall = TP/(TP+FN)
fscore = (2 * precision * recall) / (precision + recall)

print(f"Precision = {precision}\nRecall = {recall}\nFscore = {fscore}")

# %% [markdown]
# ## June 18 experiment

# %%
%%time
!python main.py -mo=train -i=input_MT -o=outputY_MT -ep=15 -l=8 -lr=0.0001 -et=True

# %%
file = "../evaluation_results_transformer/input_MT_outputY_MT_ONE_DATASET/results_8seq_len_0.0001lr_512embsize_8nhead_transformer_0.1dropout_128_batchsize_15epochs_3beamsize.txt"

with open(file, 'r') as f:
    line = f.readline()
    n = len(f.readlines())
    n = n-3 #correction in the number of lines since final lines does not contain data values

n = int(float(n)/2)

length=len(line.split())-1

TP = 0
FP = 0
FN = 0
TN = 0

with open(file, 'r') as f:
    lines = f.readlines()

    for i in range(0, n):
        predicted = lines[2*i].strip()
        predicted = predicted[10:].replace("X|","Y").replace(" ","")
        
        truevalue = lines[2*i+1].strip()
        truevalue = truevalue[10:].replace("X|","Y").replace(" ","")

        for j in range(0,length):
            if predicted[j] == truevalue[j] and truevalue[j] == 'X':
                TN = TN+1
            if predicted[j] == truevalue[j] and truevalue[j] == 'Y':
                TP = TP+1
            if predicted[j] != truevalue[j] and truevalue[j] == 'Y':
                FN = FN+1
            if predicted[j] != truevalue[j] and truevalue[j] == 'X':
                FP = FP+1
         
        #if i < 10:
        #    print(i, predicted)   
    
print(f"TP = {TP} --- FP = {FP}\nFN = {FN} --- TN = {TN}")

precision = TP/(TP+FP)
recall = TP/(TP+FN)
fscore = (2 * precision * recall) / (precision + recall)

print(f"Precision = {precision}\nRecall = {recall}\nFscore = {fscore}")

# %% [markdown]
# ## June 20 experiment

# %%
%%time
!python main.py -mo=train -i=input_MT -o=outputY_MT -ep=15 -l=10 -lr=0.0001 -et=True

# %%
file = "../evaluation_results_transformer/input_MT_outputY_MT_ONE_DATASET/results_10seq_len_0.0001lr_512embsize_8nhead_transformer_0.1dropout_128_batchsize_15epochs_3beamsize.txt"

with open(file, 'r') as f:
    line = f.readline()
    n = len(f.readlines())
    n = n-3 #correction in the number of lines since final lines does not contain data values

n = int(float(n)/2)

length=len(line.split())-1

TP = 0
FP = 0
FN = 0
TN = 0

with open(file, 'r') as f:
    lines = f.readlines()

    for i in range(0, n):
        predicted = lines[2*i].strip()
        predicted = predicted[10:].replace("X|","Y").replace(" ","")
        
        truevalue = lines[2*i+1].strip()
        truevalue = truevalue[10:].replace("X|","Y").replace(" ","")

        for j in range(0,length):
            if predicted[j] == truevalue[j] and truevalue[j] == 'X':
                TN = TN+1
            if predicted[j] == truevalue[j] and truevalue[j] == 'Y':
                TP = TP+1
            if predicted[j] != truevalue[j] and truevalue[j] == 'Y':
                FN = FN+1
            if predicted[j] != truevalue[j] and truevalue[j] == 'X':
                FP = FP+1
         
        #if i < 10:
        #    print(i, predicted)   
    
print(f"TP = {TP} --- FP = {FP}\nFN = {FN} --- TN = {TN}")

precision = TP/(TP+FP)
recall = TP/(TP+FN)
fscore = (2 * precision * recall) / (precision + recall)

print(f"Precision = {precision}\nRecall = {recall}\nFscore = {fscore}")

# %% [markdown]
# ## June 24 experiment

# %% [markdown]
# ### Test with the book of Genesis

# %%
file_input=[]
#bo_old=None
i=0

pattern = r"[^\s|]+"

for verse in F.otype.s('verse'):
    verse_text = ""
    cl_atoms = L.d(verse,'clause_atom')
    for clause_atom in cl_atoms:
        text = []
        for word in L.d(clause_atom, 'word'):
            if not F.trailer.v(word):
                text.append(F.g_cons.v(word))
            else:
                text.append(F.g_cons.v(word) + " ")
            
        clause_atom_text = "".join(text)
        clause_atom_text = clause_atom_text.replace("_"," ")
        clause_atom_text = clause_atom_text.strip()

        if F.g_cons.v(word) in ['W','H'] and F.trailer.v(word) == "":
            clause_atom_text += " "
        elif F.g_cons.v(word) == '' and F.trailer.v(word) == "":
            clause_atom_text += " "
        else:
            clause_atom_text += " "
        
        verse_text = verse_text + clause_atom_text
    verse_text = verse_text.strip()
    bo, ch, ve = T.sectionFromNode(verse)
    final = "\t".join([bo, str(ch), str(ve), verse_text.strip()])

    #if bo != bo_old:
    #    print(bo)
    #    bo_old=bo

    if bo == 'Genesis' and str(ch) == '7' and str(ve) == '16':
        if i<10:
            print(final)
        i=i+1

    file_input.append(final)

    if bo == 'Genesis':
        with open('../data/inputns_gn', 'w', encoding='utf-8') as file:
            for line in file_input:
                file.write(line + '\n')

    #with open('../data/outputYns_MT', 'w', encoding='utf-8') as file:
    #    for line in file_input:
    #        file.write(line + '\n')
       
#Genesis	7	16	W|HB>JM| ZKR WNQBH MKL BFR B>W| K>CR YWH >TW >LHJM| WJSGR JHWH B<DW|
#Genesis	7	16	X|X| X X X X X| X X X X| X X X|
#Genesis	7	16	YY X X X X Y X X X Y X X Y
#Genesis	7	16	YYXXXXYXXXYXXY

# %%
file_input=[]
#bo_old=None
i=0

pattern = r"[^\s|]+"

for verse in F.otype.s('verse'):
    verse_text = ""
    cl_atoms = L.d(verse,'clause_atom')
    for clause_atom in cl_atoms:
        text = []
        for word in L.d(clause_atom, 'word'):
            if not F.trailer.v(word):
                text.append(F.g_cons.v(word))
            else:
                text.append(F.g_cons.v(word) + " ")
            
        clause_atom_text = "".join(text)
        clause_atom_text = clause_atom_text.replace("_"," ")
        clause_atom_text = clause_atom_text.strip()
            
        if F.g_cons.v(word) in ['W','H'] and F.trailer.v(word) == "":
            clause_atom_text += "| "
        elif F.g_cons.v(word) == '' and F.trailer.v(word) == "":
            clause_atom_text += "| "
        else:
            clause_atom_text += "| "

        clause_atom_text = re.sub(pattern, "X", clause_atom_text)
        clause_atom_text = clause_atom_text.replace("X|", "Y")
        clause_atom_text = clause_atom_text.replace(" ","")
        
        verse_text = verse_text + clause_atom_text
    verse_text = verse_text.strip()
    bo, ch, ve = T.sectionFromNode(verse)
    final = "\t".join([bo, str(ch), str(ve), verse_text.strip()])

    #if bo != bo_old:
    #    print(bo)
    #    bo_old=bo

    if bo == 'Genesis' and str(ch) == '7' and str(ve) == '16':
        if i<10:
            print(final)
        i=i+1

    file_input.append(final)

    if bo == 'Genesis':
        with open('../data/outputYns_gn', 'w', encoding='utf-8') as file:
            for line in file_input:
                file.write(line + '\n')

    #with open('../data/outputYns_MT', 'w', encoding='utf-8') as file:
    #    for line in file_input:
    #        file.write(line + '\n')
       
#Genesis	7	16	W|HB>JM| ZKR WNQBH MKL BFR B>W| K>CR YWH >TW >LHJM| WJSGR JHWH B<DW|
#Genesis	7	16	X|X| X X X X X| X X X X| X X X|
#Genesis	7	16	YY X X X X Y X X X Y X X Y

#Genesis	7	16	Y Y X X X X Y X X X Y X X Y

#Genesis	7	16	YYXXXXYXXXYXXY

# %%
results = BHS.search("""
book book=Genesis
    chapter chapter=7
        verse verse=16
            word gloss* g_cons=W trailer*
""")
BHS.show(results, end=4, multiFeatures=False, queryFeatures=True, condensed=True, hiddenTypes={'phrase', 'phrase_atom','subphrase', 'sentence', 'sentence_atom', 'clause_atom'})

# %%
inputfilePath = "../data/inputns_gn"
outputfilePath = "../data/outputY_gn_updated"

def count_words_in_line(file_path, line_number):
    """
    Counts the number of words in the specified line of a text file.

    Args:
        file_path (str): Path to the text file.
        line_number (int): Number of the line to count the words in.

    Returns:
        int: Number of words in the specified line.
    """

    with open(file_path, 'r') as f:
        lines = f.readlines()
    try:
        # Access the specified line
        line = lines[line_number - 1]
        # Remove leading and trailing whitespace from the line
        line = line.strip()
        # Split the line into words using whitespace as delimiters
        words = line.split()
        # Return the number of words
        return len(words)
    except IndexError:
        print(f"Error: Line {line_number} does not exist in the file.")
        return 0

with open(inputfilePath, 'r') as f:
    n_lines = len(f.readlines())

for i in range(1,n_lines):
    word_count_line_input = count_words_in_line(inputfilePath, i)
    word_count_line_output = count_words_in_line(outputfilePath, i)

    if word_count_line_input != word_count_line_output:
            print(f"Line {i} of file {inputfilePath} contains {word_count_line_input} words.")
            print(f"Line {i} of file {outputfilePath} contains {word_count_line_output} words.\n")
    
    i=i+1

# %%
inputfilePath = "../data/inputns_gn"
outputfilePath = "../data/outputYns_gn"

def count_words_in_line(file_path, line_number):
    """
    Counts the number of words in the specified line of a text file.

    Args:
        file_path (str): Path to the text file.
        line_number (int): Number of the line to count the words in.

    Returns:
        int: Number of words in the specified line.
    """

    with open(file_path, 'r') as f:
        lines = f.readlines()
    try:
        # Access the specified line
        line = lines[line_number - 1]
        # Remove leading and trailing whitespace from the line
        line = line.strip()
        # Split the line into words using whitespace as delimiters
        words = line.split()
        # Return the number of words
        return len(words)-3
    except IndexError:
        print(f"Error: Line {line_number} does not exist in the file.")
        return 0

def count_characters_in_line(file_path, line_number):
    """
    Counts the number of words in the specified line of a text file.

    Args:
        file_path (str): Path to the text file.
        line_number (int): Number of the line to count the words in.

    Returns:
        int: Number of words in the specified line.
    """

    with open(file_path, 'r') as f:
        lines = f.readlines()
        # Access the specified line
        line = lines[line_number - 1]
        # Remove leading and trailing whitespace from the line
        line = line.strip()
        # Split the line into words using whitespace as delimiters
        words = line.split()
        # Return the number of words
        return len(words[3])

with open(inputfilePath, 'r') as f:
    n_lines = len(f.readlines())

for i in range(1,n_lines):
    word_count_line_input = count_words_in_line(inputfilePath, i)
    word_count_line_output = count_characters_in_line(outputfilePath, i)

    if word_count_line_input != word_count_line_output:
            print(f"Line {i} of file {inputfilePath} contains {word_count_line_input} words.")
            print(f"Line {i} of file {outputfilePath} contains {word_count_line_output} words.\n")
    
    i=i+1

# %%
%%time
!python main.py -mo=train -i=inputns_gn -o=outputY_gn_updated -ep=15 -l=10 -lr=0.0001 -et=True

# %%
file = "../evaluation_results_transformer/inputns_gn_outputY_gn_updated_ONE_DATASET/results_10seq_len_0.0001lr_512embsize_8nhead_transformer_0.1dropout_128_batchsize_15epochs_3beamsize.txt"

with open(file, 'r') as f:
    line = f.readline()
    n = len(f.readlines())
    n = n-3 #correction in the number of lines since final lines does not contain data values

n = int(float(n)/2)

length=len(line.split())-1

TP = 0
FP = 0
FN = 0
TN = 0

with open(file, 'r') as f:
    lines = f.readlines()

    for i in range(0, n):
        predicted = lines[2*i].strip()
        predicted = predicted[10:].replace("X|","Y").replace(" ","")
        
        truevalue = lines[2*i+1].strip()
        truevalue = truevalue[10:].replace("X|","Y").replace(" ","")

        for j in range(0,length):
            if predicted[j] == truevalue[j] and truevalue[j] == 'X':
                TN = TN+1
            if predicted[j] == truevalue[j] and truevalue[j] == 'Y':
                TP = TP+1
            if predicted[j] != truevalue[j] and truevalue[j] == 'Y':
                FN = FN+1
            if predicted[j] != truevalue[j] and truevalue[j] == 'X':
                FP = FP+1
         
        #if i < 10:
        #    print(i, predicted)   
    
print(f"TP = {TP} --- FP = {FP}\nFN = {FN} --- TN = {TN}")

precision = TP/(TP+FP)
recall = TP/(TP+FN)
fscore = (2 * precision * recall) / (precision + recall)

print(f"Precision = {precision}\nRecall = {recall}\nFscore = {fscore}")

# %%
%%time
!python main.py -mo=train -i=inputns_gn -o=outputYns_gn -ep=15 -l=10 -lr=0.0001 -et=True

# %%
file = "../evaluation_results_transformer/inputns_gn_outputYns_gn_ONE_DATASET/results_10seq_len_0.0001lr_512embsize_8nhead_transformer_0.1dropout_128_batchsize_15epochs_3beamsize.txt"

with open(file, 'r') as f:
    line = f.readline()
    n = len(f.readlines())
    n = n-3 #correction in the number of lines since final lines does not contain data values

n = int(float(n)/2)

length=len(line.split())-1

TP = 0
FP = 0
FN = 0
TN = 0

with open(file, 'r') as f:
    lines = f.readlines()

    for i in range(0, n):
        predicted = lines[2*i].strip()
        predicted = predicted[10:].replace("X|","Y").replace(" ","")
        
        truevalue = lines[2*i+1].strip()
        truevalue = truevalue[10:].replace("X|","Y").replace(" ","")

        for j in range(0,length):
            if predicted[j] == truevalue[j] and truevalue[j] == 'X':
                TN = TN+1
            if predicted[j] == truevalue[j] and truevalue[j] == 'Y':
                TP = TP+1
            if predicted[j] != truevalue[j] and truevalue[j] == 'Y':
                FN = FN+1
            if predicted[j] != truevalue[j] and truevalue[j] == 'X':
                FP = FP+1
         
        #if i < 10:
        #    print(i, predicted)   
    
print(f"TP = {TP} --- FP = {FP}\nFN = {FN} --- TN = {TN}")

precision = TP/(TP+FP)
recall = TP/(TP+FN)
fscore = (2 * precision * recall) / (precision + recall)

print(f"Precision = {precision}\nRecall = {recall}\nFscore = {fscore}")

# %% [markdown]
# ### Considering the Masoretic Text

# %%
file_input=[]
bo_old=None
i=0

pattern = r"[^\s|]+"

for verse in F.otype.s('verse'):
    verse_text = ""
    cl_atoms = L.d(verse,'clause_atom')
    for clause_atom in cl_atoms:
        text = []
        for word in L.d(clause_atom, 'word'):
            if not F.trailer.v(word):
                text.append(F.g_cons.v(word))
            else:
                text.append(F.g_cons.v(word) + " ")
            
        clause_atom_text = "".join(text)
        clause_atom_text = clause_atom_text.replace("_"," ")
        clause_atom_text = clause_atom_text.strip()

        if F.g_cons.v(word) in ['W','H'] and F.trailer.v(word) == "":
            clause_atom_text += " "
        elif F.g_cons.v(word) == '' and F.trailer.v(word) == "":
            clause_atom_text += " "
        else:
            clause_atom_text += " "
        
        verse_text = verse_text + clause_atom_text
    verse_text = verse_text.strip()
    bo, ch, ve = T.sectionFromNode(verse)
    final = "\t".join([bo, str(ch), str(ve), verse_text.strip()])

    if bo != bo_old:
        print(bo)
        bo_old=bo

    #if bo == 'Genesis' and str(ch) == '7' and str(ve) == '16':
    #    if i<10:
    #        print(final)
    #    i=i+1

    file_input.append(final)

    with open('../data/inputYns_MT', 'w', encoding='utf-8') as file:
        for line in file_input:
            file.write(line + '\n')

# %%
file_input=[]
bo_old=None
i=0

pattern = r"[^\s|]+"

for verse in F.otype.s('verse'):
    verse_text = ""
    cl_atoms = L.d(verse,'clause_atom')
    for clause_atom in cl_atoms:
        text = []
        for word in L.d(clause_atom, 'word'):
            if not F.trailer.v(word):
                text.append(F.g_cons.v(word))
            else:
                text.append(F.g_cons.v(word) + " ")
            
        clause_atom_text = "".join(text)
        clause_atom_text = clause_atom_text.replace("_"," ")
        clause_atom_text = clause_atom_text.strip()
            
        if F.g_cons.v(word) in ['W','H'] and F.trailer.v(word) == "":
            clause_atom_text += "| "
        elif F.g_cons.v(word) == '' and F.trailer.v(word) == "":
            clause_atom_text += "| "
        else:
            clause_atom_text += "| "

        clause_atom_text = re.sub(pattern, "X", clause_atom_text)
        clause_atom_text = clause_atom_text.replace("X|", "Y")
        clause_atom_text = clause_atom_text.replace(" ","")
        
        verse_text = verse_text + clause_atom_text
    verse_text = verse_text.strip()
    bo, ch, ve = T.sectionFromNode(verse)
    final = "\t".join([bo, str(ch), str(ve), verse_text.strip()])

    if bo != bo_old:
        print(bo)
        bo_old=bo

    #if bo == 'Genesis' and str(ch) == '7' and str(ve) == '16':
    #    if i<10:
    #        print(final)
    #    i=i+1

    file_input.append(final)

    #if bo == 'Genesis':
    #    with open('../data/outputYns_gn', 'w', encoding='utf-8') as file:
    #        for line in file_input:
    #            file.write(line + '\n')

    with open('../data/outputYns_MT', 'w', encoding='utf-8') as file:
        for line in file_input:
            file.write(line + '\n')

# %%
inputfilePath = "../data/inputYns_MT"
outputfilePath = "../data/outputYns_MT"

def count_words_in_line(file_path, line_number):
    """
    Counts the number of words in the specified line of a text file.

    Args:
        file_path (str): Path to the text file.
        line_number (int): Number of the line to count the words in.

    Returns:
        int: Number of words in the specified line.
    """

    with open(file_path, 'r') as f:
        lines = f.readlines()
    try:
        # Access the specified line
        line = lines[line_number - 1]
        # Remove leading and trailing whitespace from the line
        line = line.strip()
        # Split the line into words using whitespace as delimiters
        words = line.split()
        # Return the number of words
        return len(words)-3
    except IndexError:
        print(f"Error: Line {line_number} does not exist in the file.")
        return 0

def count_characters_in_line(file_path, line_number):
    """
    Counts the number of words in the specified line of a text file.

    Args:
        file_path (str): Path to the text file.
        line_number (int): Number of the line to count the words in.

    Returns:
        int: Number of words in the specified line.
    """

    with open(file_path, 'r') as f:
        lines = f.readlines()
        # Access the specified line
        line = lines[line_number - 1]
        # Remove leading and trailing whitespace from the line
        line = line.strip()
        # Split the line into words using whitespace as delimiters
        words = line.split()
        # Return the number of words
        return len(words[3])

with open(inputfilePath, 'r') as f:
    n_lines = len(f.readlines())

for i in range(1,n_lines):
    word_count_line_input = count_words_in_line(inputfilePath, i)
    word_count_line_output = count_characters_in_line(outputfilePath, i)

    if word_count_line_input != word_count_line_output:
            print(f"Line {i} of file {inputfilePath} contains {word_count_line_input} words.")
            print(f"Line {i} of file {outputfilePath} contains {word_count_line_output} words.\n")
    
    i=i+1

# %%
%%time
!python main.py -mo=train -i=inputYns_MT -o=outputYns_MT -ep=15 -l=10 -lr=0.0001 -et=True

# %%
file = "../evaluation_results_transformer/inputYns_MT_outputYns_MT_ONE_DATASET/results_10seq_len_0.0001lr_512embsize_8nhead_transformer_0.1dropout_128_batchsize_15epochs_3beamsize.txt"

with open(file, 'r') as f:
    line = f.readline()
    n = len(f.readlines())
    n = n-3 #correction in the number of lines since final lines does not contain data values

n = int(float(n)/2)

length=len(line.split())-1

TP = 0
FP = 0
FN = 0
TN = 0

with open(file, 'r') as f:
    lines = f.readlines()

    for i in range(0, n):
        predicted = lines[2*i].strip()
        predicted = predicted[10:].replace("X|","Y").replace(" ","")
        
        truevalue = lines[2*i+1].strip()
        truevalue = truevalue[10:].replace("X|","Y").replace(" ","")

        for j in range(0,length):
            if predicted[j] == truevalue[j] and truevalue[j] == 'X':
                TN = TN+1
            if predicted[j] == truevalue[j] and truevalue[j] == 'Y':
                TP = TP+1
            if predicted[j] != truevalue[j] and truevalue[j] == 'Y':
                FN = FN+1
            if predicted[j] != truevalue[j] and truevalue[j] == 'X':
                FP = FP+1
         
        #if i < 10:
        #    print(i, predicted)   
    
print(f"TP = {TP} --- FP = {FP}\nFN = {FN} --- TN = {TN}")

precision = TP/(TP+FP)
recall = TP/(TP+FN)
fscore = (2 * precision * recall) / (precision + recall)

print(f"Precision = {precision}\nRecall = {recall}\nFscore = {fscore}")

# %% [markdown]
# ## June 26 experiment

# %% [markdown]
# ### Spacing all words

# %%
file_input=[]
bo_old=None
i=0

pattern = r"[^\s|]+"

for verse in F.otype.s('verse'):
    verse_text = ""
    cl_atoms = L.d(verse,'clause_atom')
    for clause_atom in cl_atoms:
        text = []
        for word in L.d(clause_atom, 'word'):
            if not F.trailer.v(word) and not F.g_cons.v(word):
                text.append(F.g_cons.v(word))
            elif F.trailer.v(word) == ' ' and not F.g_cons.v(word):
                #text.append(F.g_cons.v(word))
                pass
            elif F.trailer.v(word) == '00 ' and not F.g_cons.v(word):
                pass
            else:
                text.append(F.g_cons.v(word) + " ")
            
            #text.append(F.g_cons.v(word) + " ") #space between all words
        
        clause_atom_text = "".join(text)
        clause_atom_text = clause_atom_text.replace("_"," ")
        clause_atom_text = clause_atom_text.strip()

        if F.g_cons.v(word) in ['W','H'] and F.trailer.v(word) == "":
            clause_atom_text += " "
        elif F.g_cons.v(word) == '' and F.trailer.v(word) == "":
            clause_atom_text += " "
        else:
            clause_atom_text += " "
        
        verse_text = verse_text + clause_atom_text
    verse_text = verse_text.strip()
    bo, ch, ve = T.sectionFromNode(verse)
    final = "\t".join([bo, str(ch), str(ve), verse_text.strip()])

    if bo != bo_old:
        print(bo)
        bo_old=bo

    #if bo == 'Jeremiah' and str(ch) == '31' and str(ve) == '38':
    #    print(final)

    file_input.append(final)

    with open('../data/input_space_MT', 'w', encoding='utf-8') as file:
        for line in file_input:
            file.write(line + '\n')

# %%
file_input=[]
bo_old=None
i=0

pattern = r"[^\s|]+"

for verse in F.otype.s('verse'):
    verse_text = ""
    cl_atoms = L.d(verse,'clause_atom')
    for clause_atom in cl_atoms:
        text = []
        for word in L.d(clause_atom, 'word'):
            #text.append(F.g_cons.v(word) + " ") #space between all words
            
            if not F.trailer.v(word) and not F.g_cons.v(word):
                text.append(F.g_cons.v(word))
            elif F.trailer.v(word) ==' 'and not F.g_cons.v(word):
                pass
            elif F.trailer.v(word) =='00 ' and not F.g_cons.v(word):
                pass
            else:
                text.append(F.g_cons.v(word) + " ")
            
        clause_atom_text = "".join(text)
        clause_atom_text = clause_atom_text.replace("_"," ")
        clause_atom_text = clause_atom_text.strip()
            
        if F.g_cons.v(word) in ['W','H'] and F.trailer.v(word) == "":
            clause_atom_text += "| "
        elif F.g_cons.v(word) == '' and F.trailer.v(word) == "":
            clause_atom_text += "| "
        else:
            clause_atom_text += "| "

        clause_atom_text = re.sub(pattern, "X", clause_atom_text)
        clause_atom_text = clause_atom_text.replace("X|", "Y")
        clause_atom_text = clause_atom_text.replace(" ","")
        
        verse_text = verse_text + clause_atom_text
    verse_text = verse_text.strip()
    bo, ch, ve = T.sectionFromNode(verse)
    final = "\t".join([bo, str(ch), str(ve), verse_text.strip()])

    if bo != bo_old:
        print(bo)
        bo_old=bo

    #if bo == 'Jeremiah' and str(ch) == '31' and str(ve) == '38':
    #    print(final)

    file_input.append(final)

    #if bo == 'Genesis':
    #    with open('../data/outputYns_gn', 'w', encoding='utf-8') as file:
    #        for line in file_input:
    #            file.write(line + '\n')

    with open('../data/outputYns_space_MT', 'w', encoding='utf-8') as file:
        for line in file_input:
            file.write(line + '\n')
       
#Genesis	7	16	W|HB>JM| ZKR WNQBH MKL BFR B>W| K>CR YWH >TW >LHJM| WJSGR JHWH B<DW|
#Genesis	7	16	X|X| X X X X X| X X X X| X X X|
#Genesis	7	16	YY X X X X Y X X X Y X X Y

#Genesis	7	16	Y Y X X X X Y X X X Y X X Y

#Genesis	7	16	YYXXXXYXXXYXXY

# %%
inputfilePath = "../data/input_space_MT"
outputfilePath = "../data/outputYns_space_MT"

def count_words_in_line(file_path, line_number):
    """
    Counts the number of words in the specified line of a text file.

    Args:
        file_path (str): Path to the text file.
        line_number (int): Number of the line to count the words in.

    Returns:
        int: Number of words in the specified line.
    """

    with open(file_path, 'r') as f:
        lines = f.readlines()
    try:
        # Access the specified line
        line = lines[line_number - 1]
        # Remove leading and trailing whitespace from the line
        line = line.strip()
        # Split the line into words using whitespace as delimiters
        words = line.split()
        # Return the number of words
        return len(words)-3
    except IndexError:
        print(f"Error: Line {line_number} does not exist in the file.")
        return 0

def count_characters_in_line(file_path, line_number):
    """
    Counts the number of words in the specified line of a text file.

    Args:
        file_path (str): Path to the text file.
        line_number (int): Number of the line to count the words in.

    Returns:
        int: Number of words in the specified line.
    """

    with open(file_path, 'r') as f:
        lines = f.readlines()
        # Access the specified line
        line = lines[line_number - 1]
        # Remove leading and trailing whitespace from the line
        line = line.strip()
        # Split the line into words using whitespace as delimiters
        words = line.split()
        # Return the number of words
        return len(words[3])

with open(inputfilePath, 'r') as f:
    n_lines = len(f.readlines())

for i in range(1,n_lines):
    word_count_line_input = count_words_in_line(inputfilePath, i)
    word_count_line_output = count_characters_in_line(outputfilePath, i)

    if word_count_line_input != word_count_line_output:
            print(f"Line {i} of file {inputfilePath} contains {word_count_line_input} words.")
            print(f"Line {i} of file {outputfilePath} contains {word_count_line_output} words.\n")
    
    i=i+1

# %%
%%time
!python main.py -mo=train -i=input_space_MT -o=outputYns_space_MT -ep=15 -l=10 -lr=0.0001 -et=True

# %% [markdown]
# ### Adding the part of the speech

# %%
file_input=[]
bo_old=None
i=0

pattern = r"[^\s|]+"

pdp_code = {
    'art': '1',
    'verb': '2',
    'subs': '3',
    'nmpr': '3',
    'advb': '5',
    'nega': '5',
    'prep': '6',
    'conj': '7',
    'prps': '8',
    'prde': '8',
    'prin': '8',
    'inrg': '8',
    'adjv': '9',
    'intj': '0'
}

for verse in F.otype.s('verse'):
    verse_text = ""
    cl_atoms = L.d(verse,'clause_atom')
    for clause_atom in cl_atoms:
        text = []
        for word in L.d(clause_atom, 'word'):
            #text.append(F.g_cons.v(word) + " ") #space between all words

            pdp = F.pdp.v(word)
            if not F.trailer.v(word) and not F.g_cons.v(word):
                #text.append(F.g_cons.v(word) + pdp_code[pdp])
                pass
            elif F.trailer.v(word) ==' 'and not F.g_cons.v(word):
                #text.append(F.g_cons.v(word) + pdp_code[pdp])
                pass
            elif F.trailer.v(word) =='00 ' and not F.g_cons.v(word):
                pass
            else:
                text.append(F.g_cons.v(word) + pdp_code[pdp] + " ")

            #    text.append(F.g_cons.v(word) + pdp_code[pdp] + " ")'''
        
        clause_atom_text = "".join(text)
        clause_atom_text = clause_atom_text.replace("_"," ")
        clause_atom_text = clause_atom_text.strip()

        if F.g_cons.v(word) in ['W','H'] and F.trailer.v(word) == "":
            clause_atom_text += " "
        elif F.g_cons.v(word) == '' and F.trailer.v(word) == "":
            clause_atom_text += " "
        else:
            clause_atom_text += " "
        
        verse_text = verse_text + clause_atom_text
    verse_text = verse_text.strip()
    bo, ch, ve = T.sectionFromNode(verse)
    final = "\t".join([bo, str(ch), str(ve), verse_text.strip()])

    if bo != bo_old:
        print(bo)
        bo_old=bo

    #if bo == 'Ezekiel' and str(ch) == '9' and str(ve) == '11':
    #    print(final)

    file_input.append(final)

    with open('../data/input_space_pdp_MT', 'w', encoding='utf-8') as file:
        for line in file_input:
            file.write(line + '\n')

#Jeremiah	31	38	HNH JMJM N>M JHWH W NBNTH H <JR L JHWH M MGDL XNN>L C<R H PNH
#Jeremiah	31	38	HNH0 JMJM3 N>M3 JHWH3 W7 NBNTH2 H1 <JR3 L6 JHWH3 M6 MGDL3 XNN>L3 C<R3 H1 PNH3

# %%
inputfilePath = "../data/input_space_pdp_MT"
outputfilePath = "../data/outputYns_space_MT"

def count_words_in_line(file_path, line_number):
    """
    Counts the number of words in the specified line of a text file.

    Args:
        file_path (str): Path to the text file.
        line_number (int): Number of the line to count the words in.

    Returns:
        int: Number of words in the specified line.
    """

    with open(file_path, 'r') as f:
        lines = f.readlines()
    try:
        # Access the specified line
        line = lines[line_number - 1]
        # Remove leading and trailing whitespace from the line
        line = line.strip()
        # Split the line into words using whitespace as delimiters
        words = line.split()
        # Return the number of words
        return len(words)-3
    except IndexError:
        print(f"Error: Line {line_number} does not exist in the file.")
        return 0

def count_characters_in_line(file_path, line_number):
    """
    Counts the number of words in the specified line of a text file.

    Args:
        file_path (str): Path to the text file.
        line_number (int): Number of the line to count the words in.

    Returns:
        int: Number of words in the specified line.
    """

    with open(file_path, 'r') as f:
        lines = f.readlines()
        # Access the specified line
        line = lines[line_number - 1]
        # Remove leading and trailing whitespace from the line
        line = line.strip()
        # Split the line into words using whitespace as delimiters
        words = line.split()
        # Return the number of words
        return len(words[3])

with open(inputfilePath, 'r') as f:
    n_lines = len(f.readlines())

for i in range(1,n_lines):
    word_count_line_input = count_words_in_line(inputfilePath, i)
    word_count_line_output = count_characters_in_line(outputfilePath, i)

    if word_count_line_input != word_count_line_output:
            print(f"Line {i} of file {inputfilePath} contains {word_count_line_input} words.")
            print(f"Line {i} of file {outputfilePath} contains {word_count_line_output} words.\n")
    
    i=i+1

# %%
inputfilePath = "../data/input_space_pdp_MT"
outputfilePath = "../data/input_space_MT"

def count_words_in_line(file_path, line_number):
    """
    Counts the number of words in the specified line of a text file.

    Args:
        file_path (str): Path to the text file.
        line_number (int): Number of the line to count the words in.

    Returns:
        int: Number of words in the specified line.
    """

    with open(file_path, 'r') as f:
        lines = f.readlines()
    try:
        # Access the specified line
        line = lines[line_number - 1]
        # Remove leading and trailing whitespace from the line
        line = line.strip()
        # Split the line into words using whitespace as delimiters
        words = line.split()
        # Return the number of words
        return len(words)-3
    except IndexError:
        print(f"Error: Line {line_number} does not exist in the file.")
        return 0

def count_characters_in_line(file_path, line_number):
    """
    Counts the number of words in the specified line of a text file.

    Args:
        file_path (str): Path to the text file.
        line_number (int): Number of the line to count the words in.

    Returns:
        int: Number of words in the specified line.
    """

    with open(file_path, 'r') as f:
        lines = f.readlines()
        # Access the specified line
        line = lines[line_number - 1]
        # Remove leading and trailing whitespace from the line
        line = line.strip()
        # Split the line into words using whitespace as delimiters
        words = line.split()
        # Return the number of words
        return len(words[3])

with open(inputfilePath, 'r') as f:
    n_lines = len(f.readlines())

for i in range(1,n_lines):
    word_count_line_input = count_words_in_line(inputfilePath, i)
    word_count_line_output = count_words_in_line(outputfilePath, i)

    if word_count_line_input != word_count_line_output:
            print(f"Line {i} of file {inputfilePath} contains {word_count_line_input} words.")
            print(f"Line {i} of file {outputfilePath} contains {word_count_line_output} words.\n")
    
    i=i+1

# %%
%%time
!python main.py -mo=train -i=input_space_pdp_MT -o=outputYns_space_MT -ep=15 -l=10 -lr=0.0001 -et=True

# %%
file = "../evaluation_results_transformer/input_space_pdp_MT_outputYns_space_MT_ONE_DATASET/results_10seq_len_0.0001lr_512embsize_8nhead_transformer_0.1dropout_128_batchsize_15epochs_3beamsize.txt"

with open(file, 'r') as f:
    line = f.readline()
    n = len(f.readlines())
    n = n-3 #correction in the number of lines since final lines does not contain data values

n = int(float(n)/2)

length=len(line.split())-1

TP = 0
FP = 0
FN = 0
TN = 0

with open(file, 'r') as f:
    lines = f.readlines()

    for i in range(0, n):
        predicted = lines[2*i].strip()
        predicted = predicted[10:].replace("X|","Y").replace(" ","")
        
        truevalue = lines[2*i+1].strip()
        truevalue = truevalue[10:].replace("X|","Y").replace(" ","")

        for j in range(0,length):
            if predicted[j] == truevalue[j] and truevalue[j] == 'X':
                TN = TN+1
            if predicted[j] == truevalue[j] and truevalue[j] == 'Y':
                TP = TP+1
            if predicted[j] != truevalue[j] and truevalue[j] == 'Y':
                FN = FN+1
            if predicted[j] != truevalue[j] and truevalue[j] == 'X':
                FP = FP+1
         
        #if i < 10:
        #    print(i, predicted)   
    
print(f"TP = {TP} --- FP = {FP}\nFN = {FN} --- TN = {TN}")

precision = TP/(TP+FP)
recall = TP/(TP+FN)
fscore = (2 * precision * recall) / (precision + recall)

print(f"Precision = {precision}\nRecall = {recall}\nFscore = {fscore}")

# %% [markdown]
# ### Issues with Judges 20:13, II Samuel 8:3, Jeremiah 31:38, and Ezekiel 9:11

# %%
! sed -n '7069p' ../data/input_space_pdp_MT
! sed -n '7069p' ../data/input_space_MT

# %%
results = BHS.search("""
book book=Judices
    chapter chapter=20
        verse verse=13
            word g_cons* trailer*
""")
BHS.show(results, end=4, multiFeatures=False, queryFeatures=True, condensed=True, hiddenTypes={'phrase', 'phrase_atom','subphrase', 'sentence', 'sentence_atom', 'clause_atom'})

# %%
! sed -n '8130p' ../data/input_space_pdp_MT
! sed -n '8130p' ../data/outputYns_space_MT

# %%
results = BHS.search("""
book book=Samuel_II
    chapter chapter=8
        verse verse=3
            word g_cons* trailer*
""")
BHS.show(results, end=4, multiFeatures=False, queryFeatures=True, condensed=True, hiddenTypes={'phrase', 'phrase_atom','subphrase', 'sentence', 'sentence_atom', 'clause_atom'})

# %%
! sed -n '12245p' ../data/input_space_MT
! sed -n '12245p' ../data/input_space_pdp_MT

# %%
results = BHS.search("""
book book=Jeremia
    chapter chapter=31
        verse verse=38
            word g_cons* trailer*
""")
BHS.show(results, end=4, multiFeatures=False, queryFeatures=True, condensed=True, hiddenTypes={'phrase', 'phrase_atom','subphrase', 'sentence', 'sentence_atom', 'clause_atom'})

# %%
! sed -n '12995p' ../data/input_space_MT
! sed -n '12995p' ../data/input_space_pdp_MT

# %%
results = BHS.search("""
book book=Ezechiel
    chapter chapter=9
        verse verse=11
            word g_cons* trailer*
""")
BHS.show(results, end=4, multiFeatures=False, queryFeatures=True, condensed=True, hiddenTypes={'phrase', 'phrase_atom','subphrase', 'sentence', 'sentence_atom', 'clause_atom'})

# %% [markdown]
# ## July 4, and 7 experiment - adding phrase boundaries

# %%
results = BHS.search("""
book book=Genesis
    chapter chapter=1
        verse verse=1
            word g_cons* trailer*
""")
BHS.show(results, end=4, multiFeatures=False, queryFeatures=True, extraFeatures={'number'}, condensed=True, hiddenTypes={'phrase','half_verse','subphrase', 'sentence', 'sentence_atom', 'clause'})

# %%
i=0
file_input=[]
bo_old=None

pattern = r"[^\s$]+"
prep = ['W','H','C','B','J','WL','M', 'L']

for verse in F.otype.s('verse'):
    verse_text = ""
    cl_atoms = L.d(verse,'clause_atom')
    for clause_atom in cl_atoms:
        text = []
        ph_atoms = L.d(clause_atom,'phrase_atom')

        for phrase_atom in ph_atoms:
            phrase_atom_text = ""
            for word in L.d(phrase_atom, 'word'):
                if not F.trailer.v(word):
                    text.append(F.g_cons.v(word))
                else:
                    text.append(F.g_cons.v(word) + " ")

            phrase_atom_text = phrase_atom_text + "".join(text)
            phrase_atom_text = phrase_atom_text.replace("_"," ")
            phrase_atom_text = phrase_atom_text.replace(" $","$")
            phrase_atom_text = phrase_atom_text.strip()
            
            if F.g_cons.v(word) in prep and F.trailer.v(word) == "":
                text.append("$ ")
            elif F.g_cons.v(word) == '' and F.trailer.v(word) == "":
                text.append("")
            else:
                text.append("$ ")
            
        if phrase_atom_text in prep:
            phrase_atom_text = phrase_atom_text + "$ "
        else:
            phrase_atom_text = phrase_atom_text + "$ "
        
        clause_atom_text = "".join(phrase_atom_text)
        clause_atom_text = clause_atom_text.replace(" $", "$")
        
        clause_atom_text = clause_atom_text.replace("$$", "$")
        
        #clause_atom_text = re.sub(pattern, "X", clause_atom_text)
        #clause_atom_text = clause_atom_text.replace("X$", "Y")
        #clause_atom_text = clause_atom_text.replace("$", "")
        #clause_atom_text = clause_atom_text.replace(" ", "")

        #clause_atom_text = clause_atom_text.replace("$","")
        
        verse_text = verse_text + clause_atom_text
    verse_text = verse_text.strip()
    bo, ch, ve = T.sectionFromNode(verse)
    final = "\t".join([bo, str(ch), str(ve), verse_text.strip()])

    if bo != bo_old:
        print(bo)
        bo_old=bo
	
    #if bo == '2_Samuel' and str(ch) == '16' and str(ve) == '23':
    #    print(final)

    file_input.append(final)

    with open('../data/output_phrase_MT2', 'w', encoding='utf-8') as file:
        for line in file_input:
            file.write(line + '\n')
#Genesis	1	1	BR>CJT BR> >LHJM >T HCMJM W>T H>RY
#Genesis	1	1	BR>CJT$ BR>$ >LHJM$ >T HCMJM W>T H>RY$
#Genesis	1	1	X$ X$ X$ X X X X$
#Genesis	1	1	Y Y Y X X X Y
#Genesis	1	1	YYYXXXY

# %%
inputfilePath = "../data/input_phrase_MT"
outputfilePath = "../data/output_phrase_MT"

def count_words_in_line(file_path, line_number):
    """
    Counts the number of words in the specified line of a text file.

    Args:
        file_path (str): Path to the text file.
        line_number (int): Number of the line to count the words in.

    Returns:
        int: Number of words in the specified line.
    """

    with open(file_path, 'r') as f:
        lines = f.readlines()
    try:
        # Access the specified line
        line = lines[line_number - 1]
        # Remove leading and trailing whitespace from the line
        line = line.strip()
        # Split the line into words using whitespace as delimiters
        words = line.split()
        # Return the number of words
        return len(words)-3
    except IndexError:
        print(f"Error: Line {line_number} does not exist in the file.")
        return 0

def count_characters_in_line(file_path, line_number):
    """
    Counts the number of words in the specified line of a text file.

    Args:
        file_path (str): Path to the text file.
        line_number (int): Number of the line to count the words in.

    Returns:
        int: Number of words in the specified line.
    """

    with open(file_path, 'r') as f:
        lines = f.readlines()
        # Access the specified line
        line = lines[line_number - 1]
        # Remove leading and trailing whitespace from the line
        line = line.strip()
        # Split the line into words using whitespace as delimiters
        words = line.split()
        # Return the number of words
        return len(words[3])

with open(inputfilePath, 'r') as f:
    n_lines = len(f.readlines())

for i in range(1,n_lines):
    word_count_line_input = count_words_in_line(inputfilePath, i)
    word_count_line_output = count_characters_in_line(outputfilePath, i)

    if word_count_line_input != word_count_line_output:
            print(f"Line {i} of file {inputfilePath} contains {word_count_line_input} words.")
            print(f"Line {i} of file {outputfilePath} contains {word_count_line_output} characters.\n")
    
    i=i+1

# %%
%%time
!python main.py -mo=train -i=input_phrase_MT -o=output_phrase_MT -ep=15 -l=10 -lr=0.0001 -et=True

# %%
file = "../evaluation_results_transformer/input_phrase_MT_output_phrase_MT_ONE_DATASET/results_10seq_len_0.0001lr_512embsize_8nhead_transformer_0.1dropout_128_batchsize_15epochs_3beamsize.txt"

with open(file, 'r') as f:
    line = f.readline()
    n = len(f.readlines())
    n = n-3 #correction in the number of lines since final lines does not contain data values

n = int(float(n)/2)

length=len(line.split())-1

TP = 0
FP = 0
FN = 0
TN = 0

with open(file, 'r') as f:
    lines = f.readlines()

    for i in range(0, n):
        predicted = lines[2*i].strip()
        predicted = predicted[10:].replace("X|","Y").replace(" ","")
        
        truevalue = lines[2*i+1].strip()
        truevalue = truevalue[10:].replace("X|","Y").replace(" ","")

        for j in range(0,length):
            if predicted[j] == truevalue[j] and truevalue[j] == 'X':
                TN = TN+1
            if predicted[j] == truevalue[j] and truevalue[j] == 'Y':
                TP = TP+1
            if predicted[j] != truevalue[j] and truevalue[j] == 'Y':
                FN = FN+1
            if predicted[j] != truevalue[j] and truevalue[j] == 'X':
                FP = FP+1
         
        #if i < 10:
        #    print(i, predicted)   
    
print(f"TP = {TP} --- FP = {FP}\nFN = {FN} --- TN = {TN}")

precision = TP/(TP+FP)
recall = TP/(TP+FN)
fscore = (2 * precision * recall) / (precision + recall)

print(f"Precision = {precision}\nRecall = {recall}\nFscore = {fscore}")

# %% [markdown]
# ### Issues with Genesis 6:3, II Samuel 16:2, 23; 18:20, Jeremiah 18:3; 31:38, and Ruth 3:5, 17

# %%
! sed -n '141p' ../data/input_phrase_MT
! sed -n '141p' ../data/input_MT

# %%
results = BHS.search("""
book book=Genesis
    chapter chapter=6
        verse verse=3
            word g_cons* trailer*
""")
BHS.show(results, end=4, multiFeatures=False, queryFeatures=True, condensed=True, hiddenTypes={'phrase','half_verse','subphrase', 'sentence', 'sentence_atom', 'clause'})

# %%
! sed -n '8346p' ../data/input_phrase_MT
! sed -n '8346p' ../data/input_MT

# %%
results = BHS.search("""
book book=Samuel_II
    chapter chapter=16
        verse verse=2
            word g_cons* trailer*
""")
BHS.show(results, end=4, multiFeatures=False, queryFeatures=True, condensed=True, hiddenTypes={'phrase','half_verse','subphrase', 'sentence', 'sentence_atom', 'clause'})

# %%
! sed -n '8367p' ../data/input_phrase_MT
! sed -n '8367p' ../data/input_MT

# %%
results = BHS.search("""
book book=Samuel_II
    chapter chapter=16
        verse verse=23
            word g_cons* trailer*
""")
BHS.show(results, end=4, multiFeatures=False, queryFeatures=True, condensed=True, hiddenTypes={'phrase','half_verse','subphrase', 'sentence', 'sentence_atom', 'clause'})

# %%
! sed -n '11903p' ../data/input_phrase_MT
! sed -n '11903p' ../data/input_MT

# %%
results = BHS.search("""
book book=Jeremia
    chapter chapter=18
        verse verse=3
            word g_cons* trailer*
""")
BHS.show(results, end=4, multiFeatures=False, queryFeatures=True, condensed=True, hiddenTypes={'phrase','half_verse','subphrase', 'sentence', 'sentence_atom', 'clause'})

# %%
! sed -n '12245p' ../data/input_phrase_MT
! sed -n '12245p' ../data/input_MT

! sed -n '19711p' ../data/input_phrase_MT
! sed -n '19711p' ../data/input_MT

! sed -n '19723p' ../data/input_phrase_MT
! sed -n '19723p' ../data/input_MT

# %%
results = BHS.search("""
book book=Jeremia
    chapter chapter=31
        verse verse=38
            word g_cons* trailer*
""")
BHS.show(results, end=4, multiFeatures=False, queryFeatures=True, condensed=True, hiddenTypes={'phrase','half_verse','subphrase', 'sentence', 'sentence_atom', 'clause'})

# %%
results = BHS.search("""
book book=Ruth
    chapter chapter=3
        verse verse=5|17
            word g_cons* trailer*
""")
BHS.show(results, end=2, multiFeatures=False, queryFeatures=True, condensed=True, hiddenTypes={'phrase','half_verse','subphrase', 'sentence', 'sentence_atom', 'clause'})

# %%
! sed -n '8367p' ../data/input_phrase_MT
! sed -n '8367p' ../data/output_phrase_MT
! sed -n '8367p' ../data/output_phrase_MT2

# %%
! sed -n '8367p' ../data/input_phrase_MT
! sed -n '8367p' ../data/output_phrase_MT
! sed -n '8367p' ../data/output_phrase_MT2

# %%
! sed -n '8367p' ../data/input_phrase_MT
! sed -n '8367p' ../data/output_phrase_MT
! sed -n '8367p' ../data/output_phrase_MT2

# %%
results = BHS.search("""
book book=Samuel_II
    chapter chapter=16
        verse verse=23
            word g_cons* trailer=&
""")
BHS.show(results, end=2, multiFeatures=False, queryFeatures=True, condensed=True, hiddenTypes={'phrase','half_verse','subphrase', 'sentence', 'sentence_atom', 'clause'})

# %%
! sed -n '8416p' ../data/input_phrase_MT
! sed -n '8416p' ../data/output_phrase_MT
! sed -n '8416p' ../data/output_phrase_MT2

# %%
results = BHS.search("""
book book=Samuel_II
    chapter chapter=18
        verse verse=20
            word g_cons* trailer=&
""")
BHS.show(results, end=2, multiFeatures=False, queryFeatures=True, condensed=True, hiddenTypes={'phrase','half_verse','subphrase', 'sentence', 'sentence_atom', 'clause'})

# %% [markdown]
# ## July 11 experiment

# %%
i=0
file_input=[]
bo_old=None

pattern = r"[^\s$]+"
prep = ['W','H','C','B','J','WL','M', 'L']

for verse in F.otype.s('verse'):
    verse_text = ""
    cl_atoms = L.d(verse,'clause_atom')
    for clause_atom in cl_atoms:
        text = []
        ph_atoms = L.d(clause_atom,'phrase_atom')

        for phrase_atom in ph_atoms:
            phrase_atom_text = ""
            for word in L.d(phrase_atom, 'word'):
                #if not F.trailer.v(word):
                #    text.append(F.g_cons.v(word))
                #else:
                #    text.append(F.g_cons.v(word) + " ")\
                
                text.append(F.g_cons.v(word) + " ") #space between all words

            phrase_atom_text = phrase_atom_text + "".join(text)
            phrase_atom_text = phrase_atom_text.replace("_"," ")
            phrase_atom_text = phrase_atom_text.replace(" $","$")
            phrase_atom_text = phrase_atom_text.strip()
            
            if F.g_cons.v(word) in prep and F.trailer.v(word) == "":
                text.append("$ ")
            elif F.g_cons.v(word) == '' and F.trailer.v(word) == "":
                text.append("")
            else:
                text.append("$ ")
            
        if phrase_atom_text in prep:
            phrase_atom_text = phrase_atom_text + "$ "
        else:
            phrase_atom_text = phrase_atom_text + "$ "
        
        clause_atom_text = "".join(phrase_atom_text)
        clause_atom_text = clause_atom_text.replace(" $", "$")
        
        clause_atom_text = clause_atom_text.replace("$$", "$")
        
        clause_atom_text = re.sub(pattern, "X", clause_atom_text)
        clause_atom_text = clause_atom_text.replace("X$", "Y")
        clause_atom_text = clause_atom_text.replace("$", "")
        clause_atom_text = clause_atom_text.replace(" ", "")

        #clause_atom_text = clause_atom_text.replace("$","")
        
        verse_text = verse_text + clause_atom_text
    verse_text = verse_text.strip()
    bo, ch, ve = T.sectionFromNode(verse)
    final = "\t".join([bo, str(ch), str(ve), verse_text.strip()])

    if bo != bo_old:
        print(bo)
        bo_old=bo
	
#    if bo == 'Genesis' and str(ch) == '1' and str(ve) == '1':
#        print(final)

    file_input.append(final)

    with open('../data/output_phrase_space_MT', 'w', encoding='utf-8') as file:
        for line in file_input:
            file.write(line + '\n')
#Genesis	1	1	BR>CJT BR> >LHJM >T HCMJM W>T H>RY
#Genesis	1	1	BR>CJT$ BR>$ >LHJM$ >T HCMJM W>T H>RY$
#Genesis	1	1	X$ X$ X$ X X X X$
#Genesis	1	1	Y Y Y X X X Y
#Genesis	1	1	YYYXXXY

# %%
inputfilePath = "../data/input_space_MT"
outputfilePath = "../data/output_phrase_space_MT"

def count_words_in_line(file_path, line_number):
    """
    Counts the number of words in the specified line of a text file.

    Args:
        file_path (str): Path to the text file.
        line_number (int): Number of the line to count the words in.

    Returns:
        int: Number of words in the specified line.
    """

    with open(file_path, 'r') as f:
        lines = f.readlines()
    try:
        # Access the specified line
        line = lines[line_number - 1]
        # Remove leading and trailing whitespace from the line
        line = line.strip()
        # Split the line into words using whitespace as delimiters
        words = line.split()
        # Return the number of words
        return len(words)-3
    except IndexError:
        print(f"Error: Line {line_number} does not exist in the file.")
        return 0

def count_characters_in_line(file_path, line_number):
    """
    Counts the number of words in the specified line of a text file.

    Args:
        file_path (str): Path to the text file.
        line_number (int): Number of the line to count the words in.

    Returns:
        int: Number of words in the specified line.
    """

    with open(file_path, 'r') as f:
        lines = f.readlines()
        # Access the specified line
        line = lines[line_number - 1]
        # Remove leading and trailing whitespace from the line
        line = line.strip()
        # Split the line into words using whitespace as delimiters
        words = line.split()
        # Return the number of words
        return len(words[3])

with open(inputfilePath, 'r') as f:
    n_lines = len(f.readlines())

for i in range(1,n_lines):
    word_count_line_input = count_words_in_line(inputfilePath, i)
    word_count_line_output = count_characters_in_line(outputfilePath, i)

    if word_count_line_input != word_count_line_output:
            print(f"Line {i} of file {inputfilePath} contains {word_count_line_input} words.")
            print(f"Line {i} of file {outputfilePath} contains {word_count_line_output} characters.\n")
    
    i=i+1

# %%
! head -1 ../data/input_space_MT
! head -1 ../data/output_phrase_space_MT

# %%
%%time
!python main.py -mo=train -i=input_space_MT -o=output_phrase_space_MT -ep=15 -l=10 -lr=0.0001 -et=True

# %%
file = "../evaluation_results_transformer/input_space_MT_output_phrase_space_MT_ONE_DATASET/results_10seq_len_0.0001lr_512embsize_8nhead_transformer_0.1dropout_128_batchsize_15epochs_3beamsize.txt"

with open(file, 'r') as f:
    line = f.readline()
    n = len(f.readlines())
    n = n-3 #correction in the number of lines since final lines does not contain data values

n = int(float(n)/2)

length=len(line.split())-1

TP = 0
FP = 0
FN = 0
TN = 0

with open(file, 'r') as f:
    lines = f.readlines()

    for i in range(0, n):
        predicted = lines[2*i].strip()
        predicted = predicted[10:].replace("X|","Y").replace(" ","")
        
        truevalue = lines[2*i+1].strip()
        truevalue = truevalue[10:].replace("X|","Y").replace(" ","")

        for j in range(0,length):
            if predicted[j] == truevalue[j] and truevalue[j] == 'X':
                TN = TN+1
            if predicted[j] == truevalue[j] and truevalue[j] == 'Y':
                TP = TP+1
            if predicted[j] != truevalue[j] and truevalue[j] == 'Y':
                FN = FN+1
            if predicted[j] != truevalue[j] and truevalue[j] == 'X':
                FP = FP+1
         
        #if i < 10:
        #    print(i, predicted)   
    
print(f"TP = {TP} --- FP = {FP}\nFN = {FN} --- TN = {TN}")

precision = TP/(TP+FP)
recall = TP/(TP+FN)
fscore = (2 * precision * recall) / (precision + recall)

print(f"Precision = {precision}\nRecall = {recall}\nFscore = {fscore}")

# %%
inputfilePath = "../data/input_space_pdp_MT"
outputfilePath = "../data/output_phrase_space_MT"

def count_words_in_line(file_path, line_number):
    """
    Counts the number of words in the specified line of a text file.

    Args:
        file_path (str): Path to the text file.
        line_number (int): Number of the line to count the words in.

    Returns:
        int: Number of words in the specified line.
    """

    with open(file_path, 'r') as f:
        lines = f.readlines()
    try:
        # Access the specified line
        line = lines[line_number - 1]
        # Remove leading and trailing whitespace from the line
        line = line.strip()
        # Split the line into words using whitespace as delimiters
        words = line.split()
        # Return the number of words
        return len(words)-3
    except IndexError:
        print(f"Error: Line {line_number} does not exist in the file.")
        return 0

def count_characters_in_line(file_path, line_number):
    """
    Counts the number of words in the specified line of a text file.

    Args:
        file_path (str): Path to the text file.
        line_number (int): Number of the line to count the words in.

    Returns:
        int: Number of words in the specified line.
    """

    with open(file_path, 'r') as f:
        lines = f.readlines()
        # Access the specified line
        line = lines[line_number - 1]
        # Remove leading and trailing whitespace from the line
        line = line.strip()
        # Split the line into words using whitespace as delimiters
        words = line.split()
        # Return the number of words
        return len(words[3])

with open(inputfilePath, 'r') as f:
    n_lines = len(f.readlines())

for i in range(1,n_lines):
    word_count_line_input = count_words_in_line(inputfilePath, i)
    word_count_line_output = count_characters_in_line(outputfilePath, i)

    if word_count_line_input != word_count_line_output:
            print(f"Line {i} of file {inputfilePath} contains {word_count_line_input} words.")
            print(f"Line {i} of file {outputfilePath} contains {word_count_line_output} characters.\n")
    
    i=i+1

# %%
! head -1 ../data/input_space_pdp_MT
! head -1 ../data/output_phrase_space_MT

# %%
%%time
!python main.py -mo=train -i=input_space_pdp_MT -o=output_phrase_space_MT -ep=15 -l=10 -lr=0.0001 -et=True

# %%
file = "../evaluation_results_transformer/input_space_pdp_MT_output_phrase_space_MT_ONE_DATASET/results_10seq_len_0.0001lr_512embsize_8nhead_transformer_0.1dropout_128_batchsize_15epochs_3beamsize.txt"

with open(file, 'r') as f:
    line = f.readline()
    n = len(f.readlines())
    n = n-3 #correction in the number of lines since final lines does not contain data values

n = int(float(n)/2)

length=len(line.split())-1

TP = 0
FP = 0
FN = 0
TN = 0

with open(file, 'r') as f:
    lines = f.readlines()

    for i in range(0, n):
        predicted = lines[2*i].strip()
        predicted = predicted[10:].replace("X|","Y").replace(" ","")
        
        truevalue = lines[2*i+1].strip()
        truevalue = truevalue[10:].replace("X|","Y").replace(" ","")

        for j in range(0,length):
            if predicted[j] == truevalue[j] and truevalue[j] == 'X':
                TN = TN+1
            if predicted[j] == truevalue[j] and truevalue[j] == 'Y':
                TP = TP+1
            if predicted[j] != truevalue[j] and truevalue[j] == 'Y':
                FN = FN+1
            if predicted[j] != truevalue[j] and truevalue[j] == 'X':
                FP = FP+1
         
        #if i < 10:
        #    print(i, predicted)   
    
print(f"TP = {TP} --- FP = {FP}\nFN = {FN} --- TN = {TN}")

precision = TP/(TP+FP)
recall = TP/(TP+FN)
fscore = (2 * precision * recall) / (precision + recall)

print(f"Precision = {precision}\nRecall = {recall}\nFscore = {fscore}")

# %%
i=0
file_input=[]
bo_old=None

pattern = r"[^\s$]+"
prep = ['W','H','C','B','J','WL','M', 'L']

for verse in F.otype.s('verse'):
    verse_text = ""
    cl_atoms = L.d(verse,'clause_atom')
    for clause_atom in cl_atoms:
        text = []
        ph_atoms = L.d(clause_atom,'phrase_atom')

        for phrase_atom in ph_atoms:
            phrase_atom_text = ""
            for word in L.d(phrase_atom, 'word'):
                #if not F.trailer.v(word):
                #    text.append(F.g_cons.v(word))
                #else:
                #    text.append(F.g_cons.v(word) + " ")\
                
                text.append(F.g_cons.v(word) + " ") #space between all words

            phrase_atom_text = phrase_atom_text + "".join(text)
            phrase_atom_text = phrase_atom_text.replace("_"," ")
            phrase_atom_text = phrase_atom_text.replace(" $","$")
            phrase_atom_text = phrase_atom_text.strip()
            
            if F.g_cons.v(word) in prep and F.trailer.v(word) == "":
                text.append("$ ")
            elif F.g_cons.v(word) == '' and F.trailer.v(word) == "":
                text.append("")
            else:
                text.append("$ ")
            
        if phrase_atom_text in prep:
            phrase_atom_text = phrase_atom_text + "$ "
        else:
            phrase_atom_text = phrase_atom_text + "$ "
        
        clause_atom_text = "".join(phrase_atom_text)
        clause_atom_text = clause_atom_text.replace(" $", "$")
        
        clause_atom_text = clause_atom_text.replace("$$", "$")
        
        clause_atom_text = re.sub(pattern, "X", clause_atom_text)
        clause_atom_text = clause_atom_text.replace("X$", "P")
        clause_atom_text = clause_atom_text.replace("$", "")
        
        #clause_atom_text = clause_atom_text.replace("$","")

        clause_atom_text = clause_atom_text + "| "
        clause_atom_text = clause_atom_text.replace(" | ", "| ")

        clause_atom_text = clause_atom_text.replace("P|", "Y")
        clause_atom_text = clause_atom_text.replace(" ", "")
        
        verse_text = verse_text + clause_atom_text
    verse_text = verse_text.strip()
    bo, ch, ve = T.sectionFromNode(verse)
    final = "\t".join([bo, str(ch), str(ve), verse_text.strip()])

    if bo != bo_old:
        print(bo)
        bo_old=bo
	
#    if bo == 'Genesis' and str(ch) == '1' and str(ve) == '2':
#        print(final)

    file_input.append(final)

    with open('../data/output_phrase_clause_space_MT', 'w', encoding='utf-8') as file:
        for line in file_input:
            file.write(line + '\n')
#Genesis	1	2	W$ H >RY$ HJTH$ THW W BHW$| W$ XCK$ <L PNJ THWM$| W$ RWX >LHJM$ MRXPT$ <L PNJ H MJM$|
#Genesis	1	2	X$ X X$ X$ X X X$| X$ X$ X X X$| X$ X X$ X$ X X X X$|
#Genesis	1	2	P X P P X X P| P P X X P| P X P P X X X P|
#Genesis	1	2	P X P P X X Y P P X X Y P X P P X X X Y

# %%
inputfilePath = "../data/input_space_pdp_MT"
outputfilePath = "../data/output_phrase_clause_space_MT"

def count_words_in_line(file_path, line_number):
    """
    Counts the number of words in the specified line of a text file.

    Args:
        file_path (str): Path to the text file.
        line_number (int): Number of the line to count the words in.

    Returns:
        int: Number of words in the specified line.
    """

    with open(file_path, 'r') as f:
        lines = f.readlines()
    try:
        # Access the specified line
        line = lines[line_number - 1]
        # Remove leading and trailing whitespace from the line
        line = line.strip()
        # Split the line into words using whitespace as delimiters
        words = line.split()
        # Return the number of words
        return len(words)-3
    except IndexError:
        print(f"Error: Line {line_number} does not exist in the file.")
        return 0

def count_characters_in_line(file_path, line_number):
    """
    Counts the number of words in the specified line of a text file.

    Args:
        file_path (str): Path to the text file.
        line_number (int): Number of the line to count the words in.

    Returns:
        int: Number of words in the specified line.
    """

    with open(file_path, 'r') as f:
        lines = f.readlines()
        # Access the specified line
        line = lines[line_number - 1]
        # Remove leading and trailing whitespace from the line
        line = line.strip()
        # Split the line into words using whitespace as delimiters
        words = line.split()
        # Return the number of words
        return len(words[3])

with open(inputfilePath, 'r') as f:
    n_lines = len(f.readlines())

for i in range(1,n_lines):
    word_count_line_input = count_words_in_line(inputfilePath, i)
    word_count_line_output = count_characters_in_line(outputfilePath, i)

    if word_count_line_input != word_count_line_output:
            print(f"Line {i} of file {inputfilePath} contains {word_count_line_input} words.")
            print(f"Line {i} of file {outputfilePath} contains {word_count_line_output} characters.\n")
    
    i=i+1

# %%
! head -1 ../data/input_space_pdp_MT
! head -1 ../data/output_phrase_clause_space_MT2
! head -1 ../data/output_phrase_clause_space_MT

# %%
%%time
!python main.py -mo=train -i=input_space_pdp_MT -o=output_phrase_clause_space_MT -ep=15 -l=10 -lr=0.0001 -et=True

# %%
file = "../evaluation_results_transformer/input_space_pdp_MT_output_phrase_clause_space_MT_ONE_DATASET/results_10seq_len_0.0001lr_512embsize_8nhead_transformer_0.1dropout_128_batchsize_15epochs_3beamsize.txt"

with open(file, 'r') as f:
    line = f.readline()
    n = len(f.readlines())
    n = n-3 #correction in the number of lines since final lines does not contain data values

n = int(float(n)/2)

length=len(line.split())-1

TPx = 0
Exp = 0
Exy = 0
Epx = 0
TPp = 0
Epy = 0
Eyx = 0
Eyp = 0
TPy = 0

with open(file, 'r') as f:
    lines = f.readlines()

    for i in range(0, n):
        predicted = lines[2*i].strip()
        predicted = predicted[10:].replace(" ","")
        
        truevalue = lines[2*i+1].strip()
        truevalue = truevalue[10:].replace(" ","")

        for j in range(0,length):
            if predicted[j] == truevalue[j] and truevalue[j] == 'X':
                TPx = TPx+1
            if predicted[j] == truevalue[j] and truevalue[j] == 'P':
                TPp = TPp+1
            if predicted[j] == truevalue[j] and truevalue[j] == 'Y':
                TPy = TPy+1
            if predicted[j] == 'X' and truevalue[j] == 'P':
                Epx = Epx+1
            if predicted[j] == 'X' and truevalue[j] == 'Y':
                Eyx = Eyx+1
            if predicted[j] == 'P' and truevalue[j] == 'X':
                Exp = Exp+1
            if predicted[j] == 'P' and truevalue[j] == 'Y':
                Eyp = Eyp+1
            if predicted[j] == 'Y' and truevalue[j] == 'X':
                Exy = Exy+1
            if predicted[j] == 'Y' and truevalue[j] == 'P':
                Epy = Epy+1
    
print(f"TPy = {TPy} --- Epy = {Epy} --- Exy = {Exy}\nEyp = {Eyp} --- TPp = {TPp} --- Exp = {Exp}\nEyx = {Eyx} --- Epx = {Epx} --- TPx = {TPx}\n")

precision_y = TPy/(TPy+Exy+Epy)
recall_y = TPy/(TPy+Eyx+Eyp)
fscore_y = (2 * precision_y * recall_y) / (precision_y + recall_y)

print(f"Precision_y = {precision_y}\nRecall_y = {recall_y}\nFscore_y = {fscore_y}\n")

precision_p = TPp/(TPp+Exp+Eyp)
recall_p = TPp/(TPp+Epx+Epy)
fscore_p = (2 * precision_p * recall_p) / (precision_p + recall_p)

print(f"Precision_p = {precision_p}\nRecall_p = {recall_p}\nFscore_p = {fscore_p}")

# %% [markdown]
# Therefore, the models which are trained on **only one thing** are better than the models with the phrases/clauses combined.

# %% [markdown]
# ## July 19 experiment - Samaritan Pentateuch

# %%
results = SP.search("""
book book=Genesis
    chapter chapter=1
        verse verse=1
            word g_cons* trailer* sp*
""")
SP.show(results, end=4, multiFeatures=False, queryFeatures=True, condensed=True)

# %%
file_input=[]
#bo_old=None
i=0

pattern = r"[^\s|]+"

for verse in F.otype.s('verse'):
    verse_text = ""
    text = []    
    for word in L.d(verse, 'word'):
        #if not F.trailer.v(word):
        #    text.append(F.g_cons.v(word))
        #else:
        #    text.append(F.g_cons.v(word) + " ")

        text.append(F.g_cons.v(word) + " ") #space between all words
            
    clause_atom_text = "".join(text)
    clause_atom_text = clause_atom_text.replace("_"," ")
    clause_atom_text = clause_atom_text.strip()

    if F.g_cons.v(word) in ['W','H'] and F.trailer.v(word) == "":
        clause_atom_text += " "
    elif F.g_cons.v(word) == '' and F.trailer.v(word) == "":
        clause_atom_text += " "
    else:
        clause_atom_text += " "
        
    verse_text = verse_text + clause_atom_text
    verse_text = verse_text.strip()
    bo, ch, ve = T.sectionFromNode(verse)
    final = "\t".join([bo, str(ch), str(ve), verse_text.strip()])

    if bo != bo_old:
        print(bo)
        bo_old=bo

#    if bo == 'Genesis' and str(ch) == '1' and str(ve) == '1':
#        if i<10:
#            print(final)
#        i=i+1

    file_input.append(final)

    with open('../data/input_space_SP', 'w', encoding='utf-8') as file:
        for line in file_input:
            file.write(line + '\n')

# %%
%%time
!python main.py -mo=predict -pcf=data.yaml

# %% [markdown]
# ## August 18 experiment - adding `pdp` to the Samaritan Pentateuch data

# %%
F.sp.freqList(nodeTypes={"word"})

# %%
results = SP.search("""
word sp=absent g_cons* lex*
""")
SP.show(results, end=1, multiFeatures=False, queryFeatures=True, condensed=True)

# %%
results = BHS.search("""
book book=Genesis
    chapter chapter=3
        verse verse=7
            word g_cons=JTPRW
""")
BHS.show(results, end=1, multiFeatures=False, queryFeatures=True, condensed=True)

# %% [markdown]
# Just ignoring the _absent_ value without attributing a new code, we have the following:

# %%
file_input=[]
bo_old=None
i=0

pattern = r"[^\s|]+"

pdp_code = {
    'art': '1',
    'verb': '2',
    'subs': '3',
    'nmpr': '3',
    'advb': '5',
    'nega': '5',
    'prep': '6',
    'conj': '7',
    'prps': '8',
    'prde': '8',
    'prin': '8',
    'inrg': '8',
    'adjv': '9',
    'intj': '0'
}

for verse in F.otype.s('verse'):
    verse_text = ""
    text = []    
    for word in L.d(verse, 'word'):
        pdp = F.sp.v(word)

        if F.lex.v(word) == 'absent':
            text.append(F.g_cons.v(word) + " ")
        else:
            text.append(F.g_cons.v(word) + pdp_code[pdp] + " ")
                   
    clause_atom_text = "".join(text)
    clause_atom_text = clause_atom_text.replace("_"," ")
    clause_atom_text = clause_atom_text.strip()

    if F.g_cons.v(word) in ['W','H'] and F.trailer.v(word) == "":
        clause_atom_text += " "
    elif F.g_cons.v(word) == '' and F.trailer.v(word) == "":
        clause_atom_text += " "
    else:
        clause_atom_text += " "
        
    verse_text = verse_text + clause_atom_text
    verse_text = verse_text.strip()
    bo, ch, ve = T.sectionFromNode(verse)
    final = "\t".join([bo, str(ch), str(ve), verse_text.strip()])

    if bo != bo_old:
        print(bo)
        bo_old=bo

    if bo == 'Genesis' and str(ch) == '3' and str(ve) == '7':
#        if i<10:
        print("SP: ", final)
        print("BHS: Genesis	3	7	W7 TPQXNH2 <JNJ3 CNJHM3 W7 JD<W2 KJ7 <JRMM9 HM8 W7 JTPRW2 <LH3 T>NH3 W7 J<FW2 LHM6 XGRT3")
#        i=i+1

    file_input.append(final)

    with open('../data/input_space_pdp_SP1', 'w', encoding='utf-8') as file:
        for line in file_input:
            file.write(line + '\n')

# %%
inputfilePath = "../data/input_space_SP"
outputfilePath = "../data/input_space_pdp_SP"

def count_words_in_line(file_path, line_number):
    """
    Counts the number of words in the specified line of a text file.

    Args:
        file_path (str): Path to the text file.
        line_number (int): Number of the line to count the words in.

    Returns:
        int: Number of words in the specified line.
    """

    with open(file_path, 'r') as f:
        lines = f.readlines()
    try:
        # Access the specified line
        line = lines[line_number - 1]
        # Remove leading and trailing whitespace from the line
        line = line.strip()
        # Split the line into words using whitespace as delimiters
        words = line.split()
        # Return the number of words
        return len(words)-3
    except IndexError:
        print(f"Error: Line {line_number} does not exist in the file.")
        return 0

def count_characters_in_line(file_path, line_number):
    """
    Counts the number of words in the specified line of a text file.

    Args:
        file_path (str): Path to the text file.
        line_number (int): Number of the line to count the words in.

    Returns:
        int: Number of words in the specified line.
    """

    with open(file_path, 'r') as f:
        lines = f.readlines()
        # Access the specified line
        line = lines[line_number - 1]
        # Remove leading and trailing whitespace from the line
        line = line.strip()
        # Split the line into words using whitespace as delimiters
        words = line.split()
        # Return the number of words
        return len(words[3])

with open(inputfilePath, 'r') as f:
    n_lines = len(f.readlines())

for i in range(1,n_lines):
    word_count_line_input = count_words_in_line(inputfilePath, i)
    word_count_line_output = count_words_in_line(outputfilePath, i)

    if word_count_line_input != word_count_line_output:
            print(f"Line {i} of file {inputfilePath} contains {word_count_line_input} words.")
            print(f"Line {i} of file {outputfilePath} contains {word_count_line_output} words.\n")
    
    i=i+1

# %%
%%time
!python main.py -mo=predict -pcf=data.yaml

# %%
! head ../data/output_phrase_space_MT

# %%
! head -n 11 ../new_data/output_phrase_space_pdp_SP

# %%
inputfilePath = "../new_data/output_phrase_space_pdp_SP"
outputfilePath = "../new_data/output_phrase_space_pdp_SP_test"

verse_old = None
result_old = ""
book_old = None

with open(inputfilePath, 'r') as f:
    lines = f.readlines()

with open(outputfilePath, 'w') as output:
    for i in range(1, len(lines)+1):
        line = lines[i - 1].strip()
        words = line.split()

        book = words[1]
        chapter = words[2]
        verse = words[3]

        result = words[5]

        if verse == verse_old:
            result = result_old + result
        else:
            if verse_old is not None:
                print(book, chapter, verse_old, result_old, sep='\t', file=output)
        
        result_old = result
        verse_old = verse

        if book != book_old:
            print(book)
        book_old=book

# %%
! head ../data/output_phrase_space_MT

# %%
! head ../new_data/output_phrase_space_pdp_SP_test

# %% [markdown]
# Being more rigorous and attributing a new code to the _absent_ value, we have the following:

# %%
file_input=[]
bo_old=None
i=0

pattern = r"[^\s|]+"

pdp_code = {
    'art': '1',
    'verb': '2',
    'subs': '3',
    'nmpr': '3',
    'advb': '5',
    'nega': '5',
    'prep': '6',
    'conj': '7',
    'prps': '8',
    'prde': '8',
    'prin': '8',
    'inrg': '8',
    'adjv': '9',
    'intj': '0',
    'absent': '2'
}

for verse in F.otype.s('verse'):
    verse_text = ""
    text = []    
    for word in L.d(verse, 'word'):
        pdp = F.sp.v(word)

        text.append(F.g_cons.v(word) + pdp_code[pdp] + " ")
                   
    clause_atom_text = "".join(text)
    clause_atom_text = clause_atom_text.replace("_"," ")
    clause_atom_text = clause_atom_text.strip()

    if F.g_cons.v(word) in ['W','H'] and F.trailer.v(word) == "":
        clause_atom_text += " "
    elif F.g_cons.v(word) == '' and F.trailer.v(word) == "":
        clause_atom_text += " "
    else:
        clause_atom_text += " "
        
    verse_text = verse_text + clause_atom_text
    verse_text = verse_text.strip()
    bo, ch, ve = T.sectionFromNode(verse)
    final = "\t".join([bo, str(ch), str(ve), verse_text.strip()])

    if bo != bo_old:
        print(bo)
        bo_old=bo

    if bo == 'Genesis' and str(ch) == '3' and str(ve) == '7':
#        if i<10:
        print("SP: ", final)
        print("BHS: Genesis	3	7	W7 TPQXNH2 <JNJ3 CNJHM3 W7 JD<W2 KJ7 <JRMM9 HM8 W7 JTPRW2 <LH3 T>NH3 W7 J<FW2 LHM6 XGRT3")
#        i=i+1

    file_input.append(final)

    with open('../data/input_space_pdp_SP', 'w', encoding='utf-8') as file:
        for line in file_input:
            file.write(line + '\n')

# %%
%%time
!python main.py -mo=predict -pcf=data.yaml

# %%
inputfilePath = "../new_data/output_phrase_space_pdp_SP"
outputfilePath = "../new_data/output_phrase_space_pdp_SP_test"

verse_old = None
result_old = ""
book_old = None

with open(inputfilePath, 'r') as f:
    lines = f.readlines()

with open(outputfilePath, 'w') as output:
    for i in range(1, len(lines)+1):
        line = lines[i - 1].strip()
        words = line.split()

        book = words[1]
        chapter = words[2]
        verse = words[3]

        result = words[5]

        if verse == verse_old:
            result = result_old + result
        else:
            if verse_old is not None:
                print(book, chapter, verse_old, result_old, sep='\t', file=output)
        
        result_old = result
        verse_old = verse

        if book != book_old:
            print(book)
        book_old=book

# %%
! head -n 20 ../new_data/output_phrase_space_pdp_SP

# %%
! head ../data/output_phrase_space_MT

# %%
! head ../new_data/output_phrase_space_pdp_SP_test

# %% [markdown]
# ## August 20 experiment - MT morphologically analyzed text

# %% [markdown]
# `g_cons` - word -consonantal-transliterated
# 
# `g_pfm` - preformative -pointed-transliterated\
# `g_vbs`- root formation -pointed-transliterated\
# `g_lex` - lexeme -pointed-transliterated\
# `g_vbe` - verbal ending -pointed-transliterated\
# `g_nme` - nominal ending -pointed-transliterated\
# `g_prs` - pronominal suffix -pointed-transliterated\
# `g_uvf` - univalent final consonant -pointed-transliterated

# %%
results = SP.search("""
word g_lex* g_nme* g_pfm* g_prs* g_uvf* g_vbe* g_vbs* trailer*
""")
SP.show(results, end=1, multiFeatures=False, queryFeatures=True, condensed=True)

# %%
results = BHS.search("""
book book=Genesis
    chapter chapter=2
        verse verse=24
            word g_lex* g_nme* g_pfm* g_prs* g_uvf* g_vbe* g_vbs* g_cons*
""")
BHS.show(results, end=1, multiFeatures=False, queryFeatures=True, condensed=True)

# %% [markdown]
# Examples of values (showing up the 10th most frequent value) for the above features on the Samaritan Pentateuch:

# %%
F.g_pfm.freqList(nodeTypes={"word"})[:10]

# %%
F.g_prs.freqList(nodeTypes={"word"})[:10]

# %%
F.g_uvf.freqList(nodeTypes={"word"})[:10]

# %%
F.g_vbe.freqList(nodeTypes={"word"})[:10]

# %%
F.g_vbs.freqList(nodeTypes={"word"})[:10]

# %%
F.g_nme.freqList(nodeTypes={"word"})[:10]

# %% [markdown]
# Generating the MT morphologically analyzed text

# %%
file_input=[]
bo_old=None
i=0

characters_to_remove = "A@E;IOU:."
translation_table = str.maketrans('', '', characters_to_remove)

for verse in F.otype.s('verse'):
    verse_text = ""
    text = []    
    for word in L.d(verse, 'word'):
        if not F.trailer.v(word):
            text.append(F.g_pfm.v(word) + F.g_vbs.v(word) + F.g_lex.v(word) + F.g_vbe.v(word) + F.g_nme.v(word) + F.g_uvf.v(word) + F.g_prs.v(word))
        else:
            text.append(F.g_pfm.v(word) + F.g_vbs.v(word) + F.g_lex.v(word) + F.g_vbe.v(word) + F.g_nme.v(word) + F.g_uvf.v(word) + F.g_prs.v(word) + " ")

    clause_atom_text = "".join(text)

    clause_atom_text = clause_atom_text.translate(translation_table)

    clause_atom_text = clause_atom_text.replace("--","-")
    clause_atom_text = clause_atom_text.replace("-"," ")
    
    clause_atom_text = clause_atom_text.replace("_"," ")
    clause_atom_text = clause_atom_text.strip()

    if F.g_cons.v(word) in ['W','H'] and F.trailer.v(word) == "":
        clause_atom_text += " "
    elif F.g_cons.v(word) == '' and F.trailer.v(word) == "":
        clause_atom_text += " "
    else:
        clause_atom_text += " "
        
    verse_text = verse_text + clause_atom_text
    verse_text = verse_text.strip()
    bo, ch, ve = T.sectionFromNode(verse)
    final = "\t".join([bo, str(ch), str(ve), verse_text.strip()])
    
    #if bo != bo_old:
    #    print(bo)
    #    bo_old=bo

    if bo == 'Genesis' and str(ch) == '3' and str(ve) == '7':
    #if i<10:
        print(final)
    #i=i+1

    #file_input.append(final)

    #with open('../data/input_morph_MT', 'w', encoding='utf-8') as file:
    #    for line in file_input:
    #        file.write(line + '\n')

# %% [markdown]
# Generating the MT morphologically analyzed text + clause boundaries

# %%
file_input=[]
bo_old=None
i=0

pattern = r"[^\s|]+"

characters_to_remove = "A@E;IOU:."
translation_table = str.maketrans('', '', characters_to_remove)

for verse in F.otype.s('verse'):
    verse_text = ""
    cl_atoms = L.d(verse,'clause_atom')
    for clause_atom in cl_atoms:
        text = []
        for word in L.d(clause_atom, 'word'):
            if not F.trailer.v(word):
                text.append(F.g_pfm.v(word) + F.g_vbs.v(word) + F.g_lex.v(word) + F.g_vbe.v(word) + F.g_nme.v(word) + F.g_uvf.v(word) + F.g_prs.v(word))
            else:
                text.append(F.g_pfm.v(word) + F.g_vbs.v(word) + F.g_lex.v(word) + F.g_vbe.v(word) + F.g_nme.v(word) + F.g_uvf.v(word) + F.g_prs.v(word) + " ")
            
        clause_atom_text = "".join(text)

        clause_atom_text = clause_atom_text.translate(translation_table)
        clause_atom_text = clause_atom_text.replace("--","-")
        clause_atom_text = clause_atom_text.replace("-"," ")

        clause_atom_text = clause_atom_text.replace("_"," ")
        clause_atom_text = clause_atom_text.strip()

        if F.g_cons.v(word) in ['W','H'] and F.trailer.v(word) == "":
            clause_atom_text += "| "
        elif F.g_cons.v(word) == '' and F.trailer.v(word) == "":
            clause_atom_text += "| "
        else:
            clause_atom_text += "| "

        clause_atom_text = re.sub(pattern, "X", clause_atom_text)
        clause_atom_text = clause_atom_text.replace("X|", "Y")
        clause_atom_text = clause_atom_text.replace(" ","")
        
        verse_text = verse_text + clause_atom_text
    verse_text = verse_text.strip()
    bo, ch, ve = T.sectionFromNode(verse)
    final = "\t".join([bo, str(ch), str(ve), verse_text.strip()])

    if bo != bo_old:
        print(bo)
        bo_old=bo

    if bo == 'Genesis' and str(ch) == '2' and str(ve) == '24':
    #    if i<10:
        print(final)
    #    i=i+1

    file_input.append(final)

    with open('../data/output_morph_clause_MT', 'w', encoding='utf-8') as file:
        for line in file_input:
            file.write(line + '\n')

# %%
inputfilePath = "../data/input_morph_MT"
outputfilePath = "../data/output_morph_clause_MT"

def count_words_in_line(file_path, line_number):
    """
    Counts the number of words in the specified line of a text file.

    Args:
        file_path (str): Path to the text file.
        line_number (int): Number of the line to count the words in.

    Returns:
        int: Number of words in the specified line.
    """

    with open(file_path, 'r') as f:
        lines = f.readlines()
    try:
        # Access the specified line
        line = lines[line_number - 1]
        # Remove leading and trailing whitespace from the line
        line = line.strip()
        # Split the line into words using whitespace as delimiters
        words = line.split()
        # Return the number of words
        return len(words)-3
    except IndexError:
        print(f"Error: Line {line_number} does not exist in the file.")
        return 0

def count_characters_in_line(file_path, line_number):
    """
    Counts the number of words in the specified line of a text file.

    Args:
        file_path (str): Path to the text file.
        line_number (int): Number of the line to count the words in.

    Returns:
        int: Number of words in the specified line.
    """

    with open(file_path, 'r') as f:
        lines = f.readlines()
        # Access the specified line
        line = lines[line_number - 1]
        # Remove leading and trailing whitespace from the line
        line = line.strip()
        # Split the line into words using whitespace as delimiters
        words = line.split()
        # Return the number of words
        return len(words[3])

with open(inputfilePath, 'r') as f:
    n_lines = len(f.readlines())

for i in range(1,n_lines):
    word_count_line_input = count_words_in_line(inputfilePath, i)
    word_count_line_output = count_characters_in_line(outputfilePath, i)

    if word_count_line_input != word_count_line_output:
            print(f"Line {i} of file {inputfilePath} contains {word_count_line_input} words.")
            print(f"Line {i} of file {outputfilePath} contains {word_count_line_output} words.\n")
    
    i=i+1

# %%
! head -1 ../data/input_morph_MT
! head -1 ../data/output_phrase_clause_space_MT2
! head -1 ../data/output_morph_clause_MT

# %%
%%time
!python main.py -mo=train -i=input_morph_MT -o=output_morph_clause_MT -ep=15 -l=10 -lr=0.0001 -et=True

# %% [markdown]
# Updated version (September 18 and 19)

# %%
file = "../evaluation_results_transformer/input_morph_MT_output_morph_clause_MT_ONE_DATASET/results_10seq_len_0.0001lr_512embsize_8nhead_transformer_0.1dropout_128_batchsize_15epochs_3beamsize.txt"

with open(file, 'r') as f:
    line = f.readline()
    n = len(f.readlines())
    n = n-3 #correction in the number of lines since final lines does not contain data values

n = int(float(n)/2)

length=len(line.split())-1

TP = 0
FP = 0
FN = 0
TN = 0

with open(file, 'r') as f:
    lines = f.readlines()

    for i in range(0, n):
        predicted = lines[2*i].strip()
        predicted = predicted[10:].replace("X|","Y").replace(" ","")
        
        truevalue = lines[2*i+1].strip()
        truevalue = truevalue[10:].replace("X|","Y").replace(" ","")

        for j in range(0,length):
            if predicted[j] == truevalue[j] and truevalue[j] == 'X':
                TN = TN+1
            if predicted[j] == truevalue[j] and truevalue[j] == 'Y':
                TP = TP+1
            if predicted[j] != truevalue[j] and truevalue[j] == 'Y':
                FN = FN+1
            if predicted[j] != truevalue[j] and truevalue[j] == 'X':
                FP = FP+1
         
        #if i < 10:
        #    print(i, predicted)   
    
print(f"TP = {TP} --- FP = {FP}\nFN = {FN} --- TN = {TN}")

precision = TP/(TP+FP)
recall = TP/(TP+FN)
fscore = (2 * precision * recall) / (precision + recall)

print(f"Precision = {precision}\nRecall = {recall}\nFscore = {fscore}")

# %% [markdown]
# Initial version

# %%
file = "../evaluation_results_transformer/input_morph_MT_output_morph_clause_MT_ONE_DATASET/results_10seq_len_0.0001lr_512embsize_8nhead_transformer_0.1dropout_128_batchsize_15epochs_3beamsize.txt"

with open(file, 'r') as f:
    line = f.readline()
    n = len(f.readlines())
    n = n-3 #correction in the number of lines since final lines does not contain data values

n = int(float(n)/2)

length=len(line.split())-1

TP = 0
FP = 0
FN = 0
TN = 0

with open(file, 'r') as f:
    lines = f.readlines()

    for i in range(0, n):
        predicted = lines[2*i].strip()
        predicted = predicted[10:].replace("X|","Y").replace(" ","")
        
        truevalue = lines[2*i+1].strip()
        truevalue = truevalue[10:].replace("X|","Y").replace(" ","")

        for j in range(0,length):
            if predicted[j] == truevalue[j] and truevalue[j] == 'X':
                TN = TN+1
            if predicted[j] == truevalue[j] and truevalue[j] == 'Y':
                TP = TP+1
            if predicted[j] != truevalue[j] and truevalue[j] == 'Y':
                FN = FN+1
            if predicted[j] != truevalue[j] and truevalue[j] == 'X':
                FP = FP+1
         
        #if i < 10:
        #    print(i, predicted)   
    
print(f"TP = {TP} --- FP = {FP}\nFN = {FN} --- TN = {TN}")

precision = TP/(TP+FP)
recall = TP/(TP+FN)
fscore = (2 * precision * recall) / (precision + recall)

print(f"Precision = {precision}\nRecall = {recall}\nFscore = {fscore}")

# %% [markdown]
# Generating the MT morphologically analyzed text + phrase boundaries

# %%
i=0
file_input=[]
bo_old=None

pattern = r"[^\s$]+"
prep = ['W','H','C','B','J','WL','M', 'L']

characters_to_remove = "A@E;IOU:."
translation_table = str.maketrans('', '', characters_to_remove)

for verse in F.otype.s('verse'):
    verse_text = ""
    cl_atoms = L.d(verse,'clause_atom')
    for clause_atom in cl_atoms:
        text = []
        ph_atoms = L.d(clause_atom,'phrase_atom')

        for phrase_atom in ph_atoms:
            phrase_atom_text = ""
            for word in L.d(phrase_atom, 'word'):
                if not F.trailer.v(word):
                    text.append(F.g_pfm.v(word) + F.g_vbs.v(word) + F.g_lex.v(word) + F.g_vbe.v(word) + F.g_nme.v(word) + F.g_uvf.v(word) + F.g_prs.v(word))
                else:
                    text.append(F.g_pfm.v(word) + F.g_vbs.v(word) + F.g_lex.v(word) + F.g_vbe.v(word) + F.g_nme.v(word) + F.g_uvf.v(word) + F.g_prs.v(word) + " ")

            phrase_atom_text = phrase_atom_text + "".join(text)
            
            phrase_atom_text = phrase_atom_text.translate(translation_table)
            phrase_atom_text = phrase_atom_text.replace("--","-")
            phrase_atom_text = phrase_atom_text.replace("-"," ")
            
            phrase_atom_text = phrase_atom_text.replace("_"," ")
            phrase_atom_text = phrase_atom_text.replace(" $","$")
            phrase_atom_text = phrase_atom_text.strip()
            
            if F.g_cons.v(word) in prep and F.trailer.v(word) == "":
                text.append("$ ")
            elif F.g_cons.v(word) == '' and F.trailer.v(word) == "":
                text.append("")
            else:
                text.append("$ ")
            
        if phrase_atom_text in prep:
            phrase_atom_text = phrase_atom_text + "$ "
        else:
            phrase_atom_text = phrase_atom_text + "$ "
        
        clause_atom_text = "".join(phrase_atom_text)
        clause_atom_text = clause_atom_text.replace(" $", "$")
        
        clause_atom_text = clause_atom_text.replace("$$", "$")
        
        clause_atom_text = re.sub(pattern, "X", clause_atom_text)
        clause_atom_text = clause_atom_text.replace("X$", "Y")
        clause_atom_text = clause_atom_text.replace("$", "")
        clause_atom_text = clause_atom_text.replace(" ", "")

        #clause_atom_text = clause_atom_text.replace("$","")
        
        verse_text = verse_text + clause_atom_text
    verse_text = verse_text.strip()
    bo, ch, ve = T.sectionFromNode(verse)
    final = "\t".join([bo, str(ch), str(ve), verse_text.strip()])

    if bo != bo_old:
        print(bo)
        bo_old=bo
	
    if bo == 'Genesis' and str(ch) == '1' and str(ve) == '1':
        print(final)

    file_input.append(final)

    with open('../data/output_morph_phrase_MT', 'w', encoding='utf-8') as file:
        for line in file_input:
            file.write(line + '\n')

# %%
inputfilePath = "../data/input_morph_MT"
outputfilePath = "../data/output_morph_phrase_MT"

def count_words_in_line(file_path, line_number):
    """
    Counts the number of words in the specified line of a text file.

    Args:
        file_path (str): Path to the text file.
        line_number (int): Number of the line to count the words in.

    Returns:
        int: Number of words in the specified line.
    """

    with open(file_path, 'r') as f:
        lines = f.readlines()
    try:
        # Access the specified line
        line = lines[line_number - 1]
        # Remove leading and trailing whitespace from the line
        line = line.strip()
        # Split the line into words using whitespace as delimiters
        words = line.split()
        # Return the number of words
        return len(words)-3
    except IndexError:
        print(f"Error: Line {line_number} does not exist in the file.")
        return 0

def count_characters_in_line(file_path, line_number):
    """
    Counts the number of words in the specified line of a text file.

    Args:
        file_path (str): Path to the text file.
        line_number (int): Number of the line to count the words in.

    Returns:
        int: Number of words in the specified line.
    """

    with open(file_path, 'r') as f:
        lines = f.readlines()
        # Access the specified line
        line = lines[line_number - 1]
        # Remove leading and trailing whitespace from the line
        line = line.strip()
        # Split the line into words using whitespace as delimiters
        words = line.split()
        # Return the number of words
        return len(words[3])

with open(inputfilePath, 'r') as f:
    n_lines = len(f.readlines())

for i in range(1,n_lines):
    word_count_line_input = count_words_in_line(inputfilePath, i)
    word_count_line_output = count_characters_in_line(outputfilePath, i)

    if word_count_line_input != word_count_line_output:
            print(f"Line {i} of file {inputfilePath} contains {word_count_line_input} words.")
            print(f"Line {i} of file {outputfilePath} contains {word_count_line_output} words.\n")
    
    i=i+1

# %%
! head -1 ../data/input_morph_MT
! head -1 ../data/output_phrase_clause_space_MT2
! head -1 ../data/output_morph_phrase_MT

# %%
%%time
!python main.py -mo=train -i=input_morph_MT -o=output_morph_phrase_MT -ep=15 -l=10 -lr=0.0001 -et=True

# %% [markdown]
# Updated version (September 22 and 23)

# %%
file = "../evaluation_results_transformer/input_morph_MT_output_morph_phrase_MT_ONE_DATASET/results_10seq_len_0.0001lr_512embsize_8nhead_transformer_0.1dropout_128_batchsize_15epochs_3beamsize.txt"

with open(file, 'r') as f:
    line = f.readline()
    n = len(f.readlines())
    n = n-3 #correction in the number of lines since final lines does not contain data values

n = int(float(n)/2)

length=len(line.split())-1

TP = 0
FP = 0
FN = 0
TN = 0

with open(file, 'r') as f:
    lines = f.readlines()

    for i in range(0, n):
        predicted = lines[2*i].strip()
        predicted = predicted[10:].replace("X|","Y").replace(" ","")
        
        truevalue = lines[2*i+1].strip()
        truevalue = truevalue[10:].replace("X|","Y").replace(" ","")

        for j in range(0,length):
            if predicted[j] == truevalue[j] and truevalue[j] == 'X':
                TN = TN+1
            if predicted[j] == truevalue[j] and truevalue[j] == 'Y':
                TP = TP+1
            if predicted[j] != truevalue[j] and truevalue[j] == 'Y':
                FN = FN+1
            if predicted[j] != truevalue[j] and truevalue[j] == 'X':
                FP = FP+1
         
        #if i < 10:
        #    print(i, predicted)   
    
print(f"TP = {TP} --- FP = {FP}\nFN = {FN} --- TN = {TN}")

precision = TP/(TP+FP)
recall = TP/(TP+FN)
fscore = (2 * precision * recall) / (precision + recall)

print(f"Precision = {precision}\nRecall = {recall}\nFscore = {fscore}")

# %% [markdown]
# Initial version

# %%
file = "../evaluation_results_transformer/input_morph_MT_output_morph_phrase_MT_ONE_DATASET/results_10seq_len_0.0001lr_512embsize_8nhead_transformer_0.1dropout_128_batchsize_15epochs_3beamsize.txt"

with open(file, 'r') as f:
    line = f.readline()
    n = len(f.readlines())
    n = n-3 #correction in the number of lines since final lines does not contain data values

n = int(float(n)/2)

length=len(line.split())-1

TP = 0
FP = 0
FN = 0
TN = 0

with open(file, 'r') as f:
    lines = f.readlines()

    for i in range(0, n):
        predicted = lines[2*i].strip()
        predicted = predicted[10:].replace("X|","Y").replace(" ","")
        
        truevalue = lines[2*i+1].strip()
        truevalue = truevalue[10:].replace("X|","Y").replace(" ","")

        for j in range(0,length):
            if predicted[j] == truevalue[j] and truevalue[j] == 'X':
                TN = TN+1
            if predicted[j] == truevalue[j] and truevalue[j] == 'Y':
                TP = TP+1
            if predicted[j] != truevalue[j] and truevalue[j] == 'Y':
                FN = FN+1
            if predicted[j] != truevalue[j] and truevalue[j] == 'X':
                FP = FP+1
         
        #if i < 10:
        #    print(i, predicted)   
    
print(f"TP = {TP} --- FP = {FP}\nFN = {FN} --- TN = {TN}")

precision = TP/(TP+FP)
recall = TP/(TP+FN)
fscore = (2 * precision * recall) / (precision + recall)

print(f"Precision = {precision}\nRecall = {recall}\nFscore = {fscore}")

# %% [markdown]
# ## September 2, 3 experiment - SP morphologically analyzed text

# %%
results = SP.search("""
book book=Genesis
    chapter chapter=3
        verse verse=7
            word g_lex=absent g_pfm* g_vbs* g_vbe* g_nme* g_prs* g_uvf* trailer* sp* g_cons* prediction*
""")
SP.show(results, end=1, multiFeatures=False, queryFeatures=True, condensed=True)

# %%
results = BHS.search("""
book book=Genesis
    chapter chapter=3
        verse verse=7
            word g_pfm* g_vbs* g_lex* g_vbe* g_nme* g_prs* g_uvf* trailer* sp=verb g_cons*
""")
BHS.show(results, end=1, multiFeatures=False, queryFeatures=True, condensed=True)

# %%
results = SP.search("""
book book=Genesis
    chapter chapter=3
        verse verse=7
            word g_pfm* g_vbs* g_lex* g_vbe* g_nme* g_prs* g_uvf* trailer* sp=verb g_cons* prediction*
""")
SP.show(results, end=1, multiFeatures=False, queryFeatures=True, condensed=True)

# %%
file_input=[]
bo_old=None
i=0

characters_to_remove = "A@E;IOU:."
translation_table = str.maketrans('', '', characters_to_remove)

for verse in F.otype.s('verse'):
    verse_text = ""
    text = []    
    for word in L.d(verse, 'word'):
        if not F.trailer.v(word):
            if F.g_lex.v(word) == 'absent': #if g_lex is absent, we are using g_cons
                text.append(F.g_pfm.v(word) + F.g_vbs.v(word) + F.g_cons.v(word) + F.g_vbe.v(word) + F.g_nme.v(word) + F.g_uvf.v(word) + F.g_prs.v(word) + " ")
            else:
                text.append(F.g_pfm.v(word) + F.g_vbs.v(word) + F.g_lex.v(word) + F.g_vbe.v(word) + F.g_nme.v(word) + F.g_uvf.v(word) + F.g_prs.v(word) + " ")
        else:
            if F.g_lex.v(word) == 'absent': #if g_lex is absent, we are using g_cons
                text.append(F.g_pfm.v(word) + F.g_vbs.v(word) + F.g_cons.v(word) + F.g_vbe.v(word) + F.g_nme.v(word) + F.g_uvf.v(word) + F.g_prs.v(word) + " ")
            else:
                text.append(F.g_pfm.v(word) + F.g_vbs.v(word) + F.g_lex.v(word) + F.g_vbe.v(word) + F.g_nme.v(word) + F.g_uvf.v(word) + F.g_prs.v(word) + " ")

    clause_atom_text = "".join(text)

    clause_atom_text = clause_atom_text.translate(translation_table)

    clause_atom_text = clause_atom_text.replace("--","-")
    clause_atom_text = clause_atom_text.replace("-"," ")
    
    clause_atom_text = clause_atom_text.replace("_"," ")
    clause_atom_text = clause_atom_text.strip()

    if F.g_cons.v(word) in ['W','H'] and F.trailer.v(word) == "":
        clause_atom_text += " "
    elif F.g_cons.v(word) == '' and F.trailer.v(word) == "":
        clause_atom_text += " "
    else:
        clause_atom_text += " "
        
    verse_text = verse_text + clause_atom_text
    verse_text = verse_text.strip()
    bo, ch, ve = T.sectionFromNode(verse)
    final = "\t".join([bo, str(ch), str(ve), verse_text.strip()])
    
    if bo != bo_old:
        print(bo)
        bo_old=bo

    if bo == 'Exodus' and str(ch) == '30' and str(ve) == '1':
    #if i<10:
        print(final)
    #i=i+1

    file_input.append(final)

#    with open('../data/input_morph_SP', 'w', encoding='utf-8') as file:
#        for line in file_input:
#            file.write(line + '\n')
#MT - Genesis	3	7	W !T!]]PQX[NH <JN/J CN/J+HM W !J!D<[W KJ <JRM/M HM W !J!TPR[W <LH/ T>N/H W !J!<F[W L+HM XGR/T
#SP - Genesis	3	7	W !T!]]PQX[NH <JN/J CN/J+HM W !J!D<[W KJ <RM/JM HM W JTP>RW <LJ/ T>N/H W !J!<F[W L+HM XGR/WT

# %%
%%time
!python main.py -mo=predict -pcf=data.yaml

# %%
inputfilePath = "../new_data/output_morph_phrase_SP"
outputfilePath = "../new_data/output_morph_phrase_SP_test"

verse_old = None
result_old = ""
book_old = None
chapter_old = None

with open(inputfilePath, 'r') as f:
    lines = f.readlines()

with open(outputfilePath, 'w') as output:
    for i in range(1, len(lines)+1):
        line = lines[i - 1].strip()
        words = line.split()

        book = words[1]
        chapter = words[2]
        verse = words[3]

        result = words[5]

        if verse == verse_old:
            result = result_old + result
        else:
            if verse_old is not None:
                print(book_old, chapter_old, verse_old, result_old, sep='\t', file=output)
        
        result_old = result
        verse_old = verse
        chapter_old = chapter

        if book != book_old:
            print(book)
        book_old=book
    print(book, chapter, verse_old, result_old, sep='\t', file=output)

# %%
! head -n 20 ../new_data/output_morph_phrase_SP

# %%
! head ../data/output_phrase_space_MT

# %%
! head ../new_data/output_phrase_space_pdp_SP_test

# %%
! head ../new_data/output_morph_phrase_SP_test

# %% [markdown]
# ## September 23 - analysis (first attempts)

# %% [markdown]
# Comparing the results of the models `output_phrase_space_pdp_SP` and `output_morph_phrase_SP`

# %%
model1 = "../new_data/output_morph_phrase_SP"
model2 = "../new_data/output_phrase_space_pdp_SP"
#model1 = "../new_data/test_morph"
#model2 = "../new_data/test_phrase"

name1 = os.path.basename(model1)
name2 = os.path.basename(model2)

outputfile1 = f"../new_data/comparison_negative_{name1}_{name2}"
output1 = open(outputfile1, "w")

outputfile2 = f"../new_data/comparison_positive_{name1}_{name2}"
output2 = open(outputfile2, "w")

print('num', 'bo', 'ch', 've', 'w+morph', 'result-w+morph', 'w+pdp', 'result-word+pdp', 'w-clean', sep='\t', file=output1)
print('num', 'bo', 'ch', 've', 'w+morph', 'result-w+morph', 'w+pdp', 'result-word+pdp', 'w-clean', sep='\t', file=output2)

k = 0
l = 0
book_old = None
chap_old = None

morphsign = "~!>+~[]/<>"
pdpsign = "0123456789<>"

translation_table1 = str.maketrans('', '', morphsign)
translation_table2 = str.maketrans('', '', pdpsign)

def process_line(line, translation_table):
    words = line.strip().split()
    try:
        word = words[4].translate(translation_table)
        word_original = words[4]
        result = words[5]
        book = words[1]
        chapter = words[2]
        verse = words[3]
        return word, word_original, result, book, chapter, verse
    except IndexError:
        return None

with open(model1, 'r') as f:
    lines_model1 = f.readlines()

with open(model2, 'r') as f:
    lines_model2 = f.readlines()

n_lines1 = len(lines_model1)
n_lines2 = len(lines_model2)

if n_lines1 != n_lines2:
    for i in range(n_lines1):
        result1 = process_line(lines_model1[i], translation_table1)
        if not result1:
            continue
        
        word1, word_original1, result1_value, book1, chapter1, verse1 = result1
        
        for j in range(i, min(i + abs(n_lines1 - n_lines2), n_lines2)): 
            result2 = process_line(lines_model2[j], translation_table2)
            if not result2:
                continue
    
            word2, word_original2, result2_value, book2, chapter2, verse2 = result2
            
            if word1 == word2 and book1 == book2 and chapter1 == chapter2 and verse1 == verse2:
                if result1_value == result2_value:
                    k += 1
                else:
                    l += 1
                    print(l, book1, chapter1, verse1, word_original1, result1_value, word_original2, result2_value, word1, sep='\t', file=output1)
                break
        
        if book1 != book_old:
            print(book1)
        book_old = book1
        
    if l == 0:
        print("\nNo word mismatch")
else:
    for i in range(n_lines1):
        result1 = process_line(lines_model1[i], translation_table1)
        word1, word_original1, result1_value, book1, chapter1, verse1 = result1

        result2 = process_line(lines_model2[i], translation_table2)
        word2, word_original2, result2_value, book2, chapter2, verse2 = result2

        if word1 == word2 and book1 == book2 and chapter1 == chapter2 and verse1 == verse2:
            if result1_value == result2_value:
                k += 1
                #print(i, book1, chapter1, verse1, word_original1, result1_value, word_original2, result2_value, word1)
                print(i, book1, chapter1, verse1, word_original1, result1_value, word_original2, result2_value, word1, sep='\t', file=output2)
            else:
                l += 1
                print(i, book1, chapter1, verse1, word_original1, result1_value, word_original2, result2_value, word1, sep='\t', file=output1)
                #print(i, book1, chapter1, verse1, word_original1, result1_value, word_original2, result2_value, word1)
        
        if book1 != book_old:
            print(book1)
        book_old = book1

output1.close()
output2.close()

# %%
import matplotlib.pyplot as plt
from collections import Counter

# %%
def carregar_ultima_coluna(arquivo):
    with open(arquivo, 'r') as f:
        linhas = f.readlines()
    
    ultima_coluna = [linha.split()[-1] for linha in linhas]  #Using the last column of each line
    return ultima_coluna

arquivo = '../new_data/comparison_negative_output_morph_phrase_SP_output_phrase_space_pdp_SP'
lower_limit = 20
higher_limit = 900

ultima_coluna = carregar_ultima_coluna(arquivo)

contagem = Counter(ultima_coluna)

contagem_filtrada = {chave: valor for chave, valor in contagem.items() if (valor > lower_limit and valor < higher_limit)}

chaves_ordenadas = sorted(contagem_filtrada, key=contagem_filtrada.get, reverse=True)
valores_ordenados = [contagem_filtrada[chave] for chave in chaves_ordenadas]

plt.bar(chaves_ordenadas, valores_ordenados, color='#c1121f')

plt.xlabel(f'Word - between {lower_limit} and {higher_limit}')
plt.ylabel('Frequency')
plt.xticks(rotation=30)

plt.show()

# %%
def carregar_ultima_coluna(arquivo):
    with open(arquivo, 'r') as f:
        linhas = f.readlines()
    
    ultima_coluna = [linha.split()[-1] for linha in linhas]  #Using the last column of each line
    return ultima_coluna

arquivo = '../new_data/comparison_positive_output_morph_phrase_SP_output_phrase_space_pdp_SP'
lower_limit = 500
higher_limit = 14000

ultima_coluna = carregar_ultima_coluna(arquivo)

contagem = Counter(ultima_coluna)

contagem_filtrada = {chave: valor for chave, valor in contagem.items() if (valor > lower_limit and valor < higher_limit)}

chaves_ordenadas = sorted(contagem_filtrada, key=contagem_filtrada.get, reverse=True)
valores_ordenados = [contagem_filtrada[chave] for chave in chaves_ordenadas]

plt.bar(chaves_ordenadas, valores_ordenados)

plt.xlabel(f'Word - between {lower_limit} and {higher_limit}')
plt.ylabel('Frequency')
plt.xticks(rotation=30)

plt.show()

# %% [markdown]
# ## September 26, October 10 - analysis

# %%
# Load the SP data, and rename the node features class F,
# the locality class L and the text class T, 
# then they cannot be overwritten while loading the MT.
SP = use('DT-UCPH/sp', version='3.4.1')
Fsp, Lsp, Tsp = SP.api.F, SP.api.L, SP.api.T

# Do the same for the MT dataset.
MT = use('etcbc/bhsa', version='2021')
Fmt, Lmt, Tmt = MT.api.F, MT.api.L, MT.api.T

# %%
import json

# %%
verse_texts = {}

PENTATEUCH = ['Genesis' , 'Exodus', 'Leviticus', 'Numbers', 'Deuteronomy']

outputfilePath = "../new_data/data_SP"
output = open(outputfilePath, "w")

data = '../data/mt_to_sp.json'

with open(data, 'r') as f:
    dictionary = json.load(f)

print('node', 'bo', 'ch', 've', 'SP', sep='\t', file=output)

for verse_node in Fsp.otype.s('verse'):
    bo, ch, ve = Tsp.sectionFromNode(verse_node)
    if bo in PENTATEUCH:
        word_nodes = Lsp.d(verse_node, 'word')
        for word_node in word_nodes:
            word_text_mt = Fsp.g_cons.v(word_node)
            print(word_node, bo, ch, ve, word_text_mt, sep='\t', file=output)

output.close()

# %%
verse_texts = {}

PENTATEUCH = ['Genesis' , 'Exodus', 'Leviticus', 'Numbers', 'Deuteronomy']

outputfilePath = "../new_data/data_MT"
output = open(outputfilePath, "w")

data = '../data/mt_to_sp.json'

with open(data, 'r') as f:
    dictionary = json.load(f)

print('node', 'bo', 'ch', 've', 'MT', sep='\t', file=output)

for verse_node in Fmt.otype.s('verse'):
    bo, ch, ve = Tmt.sectionFromNode(verse_node)
    if bo in PENTATEUCH:
        word_nodes = Lmt.d(verse_node, 'word')
        for word_node in word_nodes:
            word_text_mt = Fmt.g_cons.v(word_node)
            print(word_node, bo, ch, ve, word_text_mt, sep='\t', file=output)

output.close()

# %% [markdown]
# #### Issue with `g_cons` empty in the MT

# %%
results = SP.search("""
book book=Genesis
    chapter chapter=1
        verse verse=5
            word g_pfm* g_vbs* g_lex* g_vbe* g_nme* g_prs* g_uvf* trailer* sp* g_cons=L prediction*
""")
SP.show(results, end=1, multiFeatures=False, queryFeatures=True, condensed=True, withNodes=True)

# %%
results = MT.search("""
book book=Genesis
    chapter chapter=1
        verse verse=5
            word g_pfm* g_vbs* g_lex=- g_vbe* g_nme* g_prs* g_uvf* trailer* sp* g_cons*
""")
MT.show(results, end=1, multiFeatures=False, queryFeatures=True, condensed=True, withNodes=True)

# %% [markdown]
# #### Issue with compound `g_cons` in the MT

# %%
results = SP.search("""
book book=Genesis
    chapter chapter=4
        verse verse=22
            word g_pfm* g_vbs* g_lex* g_vbe* g_nme* g_prs* g_uvf* trailer* sp* g_cons=TWBL_QJN prediction*
""")
SP.show(results, end=1, multiFeatures=False, queryFeatures=True, condensed=True, withNodes=True)

# %%
results = MT.search("""
book book=Genesis
    chapter chapter=4
        verse verse=22
            word g_pfm* g_vbs* g_lex* g_vbe* g_nme* g_prs* g_uvf* trailer* sp* g_cons=TWBL_QJN
""")
MT.show(results, end=1, multiFeatures=False, queryFeatures=True, condensed=True, withNodes=True)

# %%
verse_texts = {}

PENTATEUCH = ['Genesis' , 'Exodus', 'Leviticus', 'Numbers', 'Deuteronomy']

data = '../data/mt_to_sp.json'

i=0

outputfilePath = "../new_data/data_MT_SP"
output = open(outputfilePath, "w")

with open(data, 'r') as f:
    dictionary = json.load(f)

print('MT_node', 'SP_node', 'bo', 'ch', 've', 'MT', 'SP', sep='\t', file=output)

for verse_node in Fmt.otype.s('verse'):
    bo, ch, ve = Tmt.sectionFromNode(verse_node)
    if bo in PENTATEUCH:
        word_nodes = Lmt.d(verse_node, 'word')
        for word_node in word_nodes:
            word_text_mt = Fmt.g_cons.v(word_node)
            if str(word_node) in dictionary:
                word_text_sp = Fsp.g_cons.v(dictionary[str(word_node)])
                print(word_node, dictionary[str(word_node)], bo, ch, ve, word_text_mt, word_text_sp, sep='\t', file=output)
            else:
                print(word_node, ' ', bo, ch, ve, word_text_mt, ' ', sep='\t', file=output)

output.close()

# %%
model1 = "../new_data/output_morph_phrase_SP"
model2 = "../new_data/output_phrase_space_pdp_SP"

name1 = os.path.basename(model1)
name2 = os.path.basename(model2)

outputfilePath = f"../new_data/comparison_{name1}_{name2}"
output = open(outputfilePath, "w")

print('num', 'bo', 'ch', 've', 'w+morph', 'result-w+morph', 'w+pdp', 'result-word+pdp', 'word_clean', sep='\t', file=output)

k = 0
l = 0
book_old = None
chap_old = None

morphsign = "~!>+~[]/<>"
pdpsign = "0123456789<>"

translation_table1 = str.maketrans('', '', morphsign)
translation_table2 = str.maketrans('', '', pdpsign)

def process_line(line, translation_table):
    words = line.strip().split()
    try:
        word = words[4].translate(translation_table)
        word_original = words[4]
        result = words[5]
        book = words[1]
        chapter = words[2]
        verse = words[3]
        return word, word_original, result, book, chapter, verse
    except IndexError:
        return None

with open(model1, 'r') as f:
    lines_model1 = f.readlines()

with open(model2, 'r') as f:
    lines_model2 = f.readlines()

n_lines1 = len(lines_model1)
n_lines2 = len(lines_model2)

if n_lines1 != n_lines2:
    for i in range(n_lines1):
        result1 = process_line(lines_model1[i], translation_table1)
        if not result1:
            continue
        
        word1, word_original1, result1_value, book1, chapter1, verse1 = result1
        
        for j in range(i, min(i + abs(n_lines1 - n_lines2), n_lines2)): 
            result2 = process_line(lines_model2[j], translation_table2)
            if not result2:
                continue
    
            word2, word_original2, result2_value, book2, chapter2, verse2 = result2
            
            if word1 == word2 and book1 == book2 and chapter1 == chapter2 and verse1 == verse2:
                if result1_value == result2_value:
                    k += 1
                else:
                    l += 1
                    print(l, book1, chapter1, verse1, word_original1, result1_value, word_original2, result2_value, word1, sep='\t', file=output)
                break
        
        if book1 != book_old:
            print(book1)
        book_old = book1
        
    if l == 0:
        print("\nNo word mismatch")
else:
    for i in range(n_lines1):
        result1 = process_line(lines_model1[i], translation_table1)
        word1, word_original1, result1_value, book1, chapter1, verse1 = result1

        result2 = process_line(lines_model2[i], translation_table2)
        word2, word_original2, result2_value, book2, chapter2, verse2 = result2

        if word1 == word2 and book1 == book2 and chapter1 == chapter2 and verse1 == verse2:
            if result1_value == result2_value:
                k += 1
                #print(i, book1, chapter1, verse1, word_original1, result1_value, word_original2, result2_value, word1)
                print(i, book1, chapter1, verse1, word_original1, result1_value, word_original2, result2_value, word1, sep='\t', file=output)
            else:
                l += 1
                print(i, book1, chapter1, verse1, word_original1, result1_value, word_original2, result2_value, word1, sep='\t', file=output)
                #print(i, book1, chapter1, verse1, word_original1, result1_value, word_original2, result2_value, word1)
        
        if book1 != book_old:
            print(book1)
        book_old = book1

output.close()

# %%
model = "../new_data/comparison_output_morph_phrase_SP_output_phrase_space_pdp_SP"
data_MT = "../new_data/data_MT_SP"

name = os.path.basename(model)

outputfilePath = f"../new_data/{name}_t"
output = open(outputfilePath, "w")

print('num', 'bo', 'ch', 've', 'SPw+morph', 'SPresult-w+morph', 'SPw+pdp', 'SPresult-w+pdp', 'MT', sep='\t', file=output)

k = 0
l = 0
book_old = None
chap_old = None
j_old = None

morphsign = "~!>+~[]/<>"

translation_table = str.maketrans('', '', morphsign)

def process_model(line):
    words = line.strip().split()
    try:
        num = words[0]
        book = words[1]
        chapter = words[2]
        verse = words[3]
        SPwordmorph = words[4]
        SPwordmorphresult = words[5]
        SPwordpdp = words[6]
        SPwordpdpresult = words[7]
        word_original = words[8]
        return num, book, chapter, verse, SPwordmorph, SPwordmorphresult, SPwordpdp, SPwordpdpresult, word_original
    except IndexError:
        return None

def process_data(line, table):
    words = line.strip().split()
    try:
        book = words[2]
        chapter = words[3]
        verse = words[4]
        MTword = words[5]
        SPword_original = words[6].translate(table)
        return book, chapter, verse, MTword, SPword_original
    except IndexError:
        return None

with open(model, 'r') as f:
    lines_model = f.readlines()

with open(data_MT, 'r') as f:
    lines_data_MT = f.readlines()

n_lines1 = len(lines_model)
n_lines2 = len(lines_data_MT)

for i in range(n_lines1):
    result1 = process_model(lines_model[i])
    if not result1:
        continue
    
    num1, book1, chapter1, verse1, SPwordmorph1, SPwordmorphresult1, SPwordpdp1, SPwordpdpresult1, word_original1 = result1
    
    for j in range(i, min(i + abs(n_lines1 - n_lines2), n_lines2)): 
        result2 = process_data(lines_data_MT[j], translation_table)
        if not result2:
            continue

        book2, chapter2, verse2, MTword2, SPword_original2 = result2
        
        if word_original1 == SPword_original2 and book1 == book2 and chapter1 == chapter2 and verse1 == verse2:
            k += 1
            print(num1, book1, chapter1, verse1, SPwordmorph1, SPwordmorphresult1, SPwordpdp1, SPwordpdpresult1, MTword2, sep='\t', file=output)
        else:
            l += 1
            #print(num1, book1, chapter1, verse1, SPwordmorph1, SPwordmorphresult1, SPwordpdp1, SPwordpdpresult1, MTword2, sep='\t')
            print(num1, book1, chapter1, verse1, SPwordmorph1, SPwordmorphresult1, SPwordpdp1, SPwordpdpresult1, MTword2, sep='\t', file=output)   
        break
    
    if book1 != book_old:
        print(book1)
    book_old = book1
    
if l == 0:
    print("\nNo word mismatch")

output.close()

# %%
# Abrir os arquivos
with open('../new_data/comparison_output_morph_phrase_SP_output_phrase_space_pdp_SP', 'r', encoding='utf-8') as file1, \
     open('../new_data/data_MT_SP', 'r', encoding='utf-8') as file2, \
     open('../new_data/comparison_output_morph_phrase_SP_output_phrase_space_pdp_SP_MT', 'w', encoding='utf-8') as output_file:

    # Ler o conteúdo do arquivo 1 e arquivo 2
    arquivo1_linhas = file1.readlines()
    arquivo2_linhas = file2.readlines()
    
    # Escrever o cabeçalho no arquivo de saída (sem a coluna 'word_clean')
    header_colunas1 = arquivo1_linhas[0].strip().split('\t')
    header_sem_word_clean = header_colunas1[:8]  # Excluir 'word_clean'
    output_file.write('\t'.join(header_sem_word_clean) + '\tMT\n')  # Adicionar 'MT' no cabeçalho
    
    # Criar um dicionário para armazenar os dados do arquivo 2 com a SP modificada
    arquivo2_data = {}
    for linha2 in arquivo2_linhas[1:]:  # Ignorar o cabeçalho
        colunas2 = linha2.strip().split('\t')
        
        # Preencher valores vazios caso a linha tenha menos colunas que o esperado
        while len(colunas2) < 7:
            colunas2.append('')  # Adicionar string vazia para as colunas faltantes

        bo2, ch2, ve2, sp2, mt2 = colunas2[2], colunas2[3], colunas2[4], colunas2[6], colunas2[5]
        sp_modified = sp2.replace('<', '').replace('>', '')  # Remover os símbolos < e >
        
        # Usar (bo, ch, ve, SP modificada) como chave e MT como valor
        arquivo2_data[(bo2, ch2, ve2, sp_modified)] = mt2

    # Processar as linhas do arquivo 1
    for linha1 in arquivo1_linhas[1:]:  # Ignorar o cabeçalho
        colunas1 = linha1.strip().split('\t')
        bo1, ch1, ve1, word_clean1 = colunas1[1], colunas1[2], colunas1[3], colunas1[8]
        
        # Montar a chave para procurar no dicionário
        key = (bo1, ch1, ve1, word_clean1)
        
        # Obter o valor correspondente da coluna MT ou vazio se não houver correspondência
        mt_value = arquivo2_data.get(key, '')
        
        # Escrever a linha no arquivo de saída sem a coluna 'word_clean'
        linha_sem_word_clean = colunas1[:8]  # Excluir a coluna 'word_clean'
        output_file.write('\t'.join(linha_sem_word_clean) + '\t' + mt_value + '\n')

# %% [markdown]
# #### October 14 experiment

# %%
with open('../data/output_phrase_space_MT2', 'r') as f1:
    lines = f1.readlines()

with open('../new_data/output_morph_phrase_MT_word', 'w') as f2:
    f2.write('bo\tch\tve\tresult-MT\tMT\n')
    for line in lines:
        # Dividir a linha em partes: Livro, Capítulo, Versículo e Sequência de caracteres
        parts = line.strip().split()
        book, chapter, verse = parts[0], parts[1], parts[2]
        sequence = parts[3:]

        for word in sequence:
            if '$' in word:
                result = 'Y'
            else:
                result = 'X'
            word_updated = word.replace('$', '')
            if book in ['Genesis', 'Exodus', 'Leviticus', 'Numbers', 'Deuteronomy']:
                f2.write(f'{book}\t{chapter}\t{verse}\t{result}\t{word_updated}\n')

# %%
with open('../new_data/comparison_output_morph_phrase_SP_output_phrase_space_pdp_SP_MT', 'r', encoding='utf-8') as file1, \
     open('../new_data/output_morph_phrase_MT_word', 'r', encoding='utf-8') as file3, \
     open('../new_data/comparison_output_morph_phrase_SP_output_phrase_space_pdp_SP_MT_word', 'w', encoding='utf-8') as output_file:

    # Ler o conteúdo do arquivo 1 modificado e arquivo 3
    arquivo1_linhas = file1.readlines()
    arquivo3_linhas = file3.readlines()
    
    # Escrever o cabeçalho no arquivo de saída (adicionando a coluna result-MT)
    header_colunas1 = arquivo1_linhas[0].strip().split('\t')
    output_file.write('\t'.join(header_colunas1) + '\tresult-MT\n')  # Adicionar 'result-MT' no cabeçalho
    
    # Criar um dicionário para armazenar os dados do arquivo 3 com base em MT
    arquivo3_data = {}
    for linha3 in arquivo3_linhas[1:]:  # Ignorar o cabeçalho
        colunas3 = linha3.strip().split('\t')
        
        # Verificar se a linha tem o número esperado de colunas (mínimo 5)
        if len(colunas3) >= 5:
            bo3, ch3, ve3, result_mt3, word_clean3 = colunas3[0], colunas3[1], colunas3[2], colunas3[3], colunas3[4]
            # Usar (bo, ch, ve, word_clean) como chave e result-MT como valor
            arquivo3_data[(bo3, ch3, ve3, word_clean3)] = result_mt3

    # Processar as linhas do arquivo 1 modificado
    for linha1 in arquivo1_linhas[1:]:  # Ignorar o cabeçalho
        colunas1 = linha1.strip().split('\t')
        
        while len(colunas1) < 9:
            colunas1.append(' ') 
        
        bo1, ch1, ve1, word_clean1 = colunas1[1], colunas1[2], colunas1[3], colunas1[8]  # 'MT' está na posição 8 agora
        
        # Montar a chave para procurar no dicionário
        key = (bo1, ch1, ve1, word_clean1)

        # Verificar se a chave existe no arquivo 3
        if key in arquivo3_data:
            result_mt_value = arquivo3_data[key]
        else:
            result_mt_value = ''  # Se não houver correspondência, manter vazio
        
        # Debug: imprimir a chave e o valor correspondente para verificar correspondências
        #print(f"Chave: {key}, result-MT: {result_mt_value}")

        # Escrever a linha no arquivo de saída com a nova coluna result-MT
        output_file.write('\t'.join(colunas1) + '\t' + result_mt_value + '\n')

output_file.close()

# %% [markdown]
# ## October 17 experiment - creating a new dataset

# %%
SP = use('DT-UCPH/sp:clone', checkout='clone', version='3.4.1', hoist=globals())

# %%
results = SP.search("""
word
""")

outputfilePath = f"../new_data/node_words_SP"
output = open(outputfilePath, "w")

for i in range(0,len(results)):
    print(results[i][0], F.g_cons.v(results[i][0]), file=output, sep='\t')

# %% [markdown]
# Words are defined by the node 405426 to 520316

# %%
# Leitura dos arquivos
with open('../new_data/output_morph_phrase_SP', 'r') as f1, open('../new_data/node_words_SP', 'r') as f2:
    linhas_arquivo1 = [linha.strip().split('\t') for linha in f1.readlines()]
    linhas_arquivo2 = [linha.strip().split('\t') for linha in f2.readlines()]

# Caracteres a serem removidos
morphsign = "~!>+~[]/<>"
translation_table = str.maketrans('', '', morphsign)

# Dicionário para correspondência entre a coluna 5 do arquivo 1 e a coluna 2 do arquivo 2
dict_arq2 = {}
index = 405426
maqaf = False

# Preenchendo o dicionário com valores de test_node_words_SP
for coluna1, coluna2 in linhas_arquivo2:
    coluna2 = coluna2.translate(translation_table)
    
    # Separar palavras compostas e armazenar no dicionário
    if '_' in coluna2:
        coluna2, coluna2b = map(str, coluna2.split('_'))
        maqaf = True
    else:
        maqaf = False
    dict_arq2[index] = [coluna1, coluna2, maqaf]
    index = index+1

index = 405426
arquivo1 = {}

for linha in linhas_arquivo1:
    arquivo1[index] = [str(index), linha[2].translate(translation_table), linha[3]]
    index = index+1

index = 405426
newfile = {}
j=1

for i in range(index, index+len(dict_arq2)):
    if dict_arq2[i][2] == True:
        newfile[i] = [str(i), dict_arq2[i][1], arquivo1[i+j][2]]
        j=j+1
    else:
        newfile[i] = [i, dict_arq2[i][1], arquivo1[i][2]]

index = 405426
old_value = 'X'
old_node = index

caminho_arquivo3 = '../new_data/phrase'
f3 = open(caminho_arquivo3, 'w')

for i in range(index, index+len(newfile)):
    current_value = newfile[i][2]
    current_node = newfile[i][0]
    
    if current_value == 'X' and old_value == 'X':
        pass
    elif current_value == 'X' and old_value == 'Y':
        old_value = current_value
        old_node = current_node
    elif current_value == 'Y' and old_value == 'X':
        print(f"{old_node}-{current_node}", file=f3)
        old_node = current_node
        old_value = current_value
    elif current_value == 'Y' and old_value == 'Y':
        print(f"{current_node}", file=f3)
        old_node = current_node

f3.close()

# %% [markdown]
# Testing if the sequence in `phrase` file is complete and consistent.

# %%
# File path for the nodes file
nodes_file_path = '../new_data/phrase'

# Function to expand the sequence into a full list of nodes
def expand_sequence(sequence):
    expanded_sequence = []
    for item in sequence:
        item = item.strip()  # Remove any extra whitespace
        if '-' in item:  # Expand ranges
            start, end = map(int, item.split('-'))
            expanded_sequence.extend(range(start, end + 1))
        else:  # Add single nodes
            expanded_sequence.append(int(item))
    return expanded_sequence

# Read the file and collect the sequence
with open(nodes_file_path, "r") as file:
    sequence = file.readlines()

# Expand the sequence read from the file
expanded_sequence = expand_sequence(sequence)

# Function to check for integrity by verifying consecutive nodes
def check_integrity(expanded_sequence):
    for i in range(len(expanded_sequence) - 1):
        if expanded_sequence[i] + 1 != expanded_sequence[i + 1]:
            print(f"Missing or inconsistent node between {expanded_sequence[i]} and {expanded_sequence[i + 1]}")
            return False
    return True

# Run the integrity check
if check_integrity(expanded_sequence):
    print("The sequence in the file is complete and consistent.")
else:
    print("The sequence in the file has missing or inconsistent nodes.")

# %% [markdown]
# #### November 4, 6, 7, 10 experiment - seeking gaps in the `phrase` file

# %% [markdown]
#  Example of gaps in the `phrase` file due the Hebrew maqaf (־) between the words (represented by the underscore in the text):

# %% [markdown]
# <div>
# <img src="attachment:e79e7521-3c9b-4161-96ff-9d93f3dac1b0.png" width="800"/>
# </div>

# %% [markdown]
# <div>
# <img src="attachment:98e2e261-d8fb-4ac7-94f4-172f91e19117.png" width="800"/>
# </div>

# %%
results = SP.search("""
book book=Genesis
    chapter chapter=4
        verse verse=22
            word
""")
SP.show(results, end=1, multiFeatures=False, queryFeatures=True, condensed=True, withNodes=True)

# %% [markdown]
# ## November 14 experiment - comparing results using `comparison_output_morph_phrase_SP_output_phrase_space_pdp_SP_MT_word` file

# %%
numSPmorph = 0
numSPpdp = 0
threeagree = 0
twodisagree = 0
wordsmaqaf = 0

ref_twodisagree = set()
ref_numSPmorph = set()
ref_numSPpdp = set()

with open('../new_data/comparison_output_morph_phrase_SP_output_phrase_space_pdp_SP_MT_word', 'r', encoding='utf-8') as file:
    lines = file.readlines()
    
    for line in lines[1:]:  # Ignorar o cabeçalho
        text = line.strip().split('\t')
        SPmorph = text[5]
        SPpdp = text[7]
        if len(text) == 10:
            MT = text[9]
        else:
            MT = ''
        bo = text[1]
        ch = text[2]
        ve = text[3]
        ref = text[0]
        wordSPmorph = text[4]
        wordpdp = text[6][-1]

        if MT != '':
            if SPmorph == SPpdp:
                if SPmorph == MT:
                    threeagree = threeagree + 1
                else:
                    twodisagree = twodisagree + 1
                    ref_twodisagree.add((ref,(bo,ch,ve),wordSPmorph, wordpdp))
            else:
                if SPmorph == MT:
                    numSPmorph = numSPmorph + 1
                    ref_numSPmorph.add((ref,(bo,ch,ve),wordSPmorph, wordpdp))
                else:
                    numSPpdp = numSPpdp + 1
                    ref_numSPpdp.add((ref,(bo,ch,ve),wordSPmorph, wordpdp))
        else:
            wordsmaqaf = wordsmaqaf + 1

ref_twodisagree = sorted(ref_twodisagree, key=lambda x: int(x[0]))
ref_numSPmorph = sorted(ref_numSPmorph, key=lambda x: int(x[0]))
ref_numSPpdp = sorted(ref_numSPpdp, key=lambda x: int(x[0]))

total = len(lines) - wordsmaqaf

print("All agree:\t\t\t\t", threeagree, "{:.3f}".format(threeagree/total), sep="\t")
print("Two agree, but disagree with MT:\t", twodisagree, "{:.3f}".format(twodisagree/total), sep="\t")
print("SP_morph agrees with MT (disagree SP_pdp):", numSPmorph, "{:.3f}".format(numSPmorph/total), sep="\t")
print("SP_pdp agrees with MT (disagree SP_morph):", numSPpdp, "{:.3f}".format(numSPpdp/total), sep="\t")

# %% [markdown]
# There is a great agreement between both models and MT (89.5%).\
# 6.7% of the results in common between the two models disagree with MT.\
# About 2.2% of the results of the model that uses morphological information (SP_morph) agree with MT and disagree with the the model that uses part-of-speach (SP_pdp).\
# In the other hand, 1.7% of the results of the model that uses part-of-speach (SP_pdp) agree with MT and disagree with model that uses morphological information (SP_morph).\
# Therefore, **SP_morph have a higher alignment with MT than SP_pdp**, given the slightly higher percentage of agreement with MT.

# %%
ref_twodisagree

# %%
ref_numSPpdp

# %%
ref_numSPmorph

# %% [markdown]
# ### Analysis of dictionaries of differences

# %%
from collections import Counter

# %%
words = [item[2] for item in ref_twodisagree]
word_counts = Counter(words)
sorted_counts = word_counts.most_common()

i=1
for word, count in sorted_counts:
    if i<10:
        print(f"{word}: {count}")
        i=i+1

# %%
words = [item[2] for item in ref_numSPpdp]
word_counts = Counter(words)
sorted_counts = word_counts.most_common()

i=1
for word, count in sorted_counts:
    if i<10:
        print(f"{word}: {count}")
        i=i+1

# %%
words = [item[2] for item in ref_numSPmorph]
word_counts = Counter(words)
sorted_counts = word_counts.most_common()

i=1
for word, count in sorted_counts:
    if i<10:
        print(f"{word}: {count}")
        i=i+1

# %% [markdown]
# ## November 17 experiment - comparing results with MT

# %%
twoagree = 0
disagree = 0
wordsmaqaf = 0

ref_disagree = set()

with open('../new_data/comparison_output_morph_phrase_SP_output_phrase_space_pdp_SP_MT_word', 'r', encoding='utf-8') as file:
    lines = file.readlines()
    
    for line in lines[1:]:  # Ignorar o cabeçalho
        text = line.strip().split('\t')
        SPmorph = text[5]
        if len(text) == 10:
            MT = text[9]
        else:
            MT = ''
        bo = text[1]
        ch = text[2]
        ve = text[3]
        ref = text[0]
        wordSPmorph = text[4]
        wordpdp = text[6][-1]

        if MT != '':
            if SPmorph == MT:
                twoagree = twoagree + 1
            else:
                disagree = disagree + 1
                ref_disagree.add((ref,(bo,ch,ve),wordSPmorph, wordpdp))
        else:
            wordsmaqaf = wordsmaqaf + 1

ref_disagree = sorted(ref_disagree, key=lambda x: int(x[0]))

total = len(lines) - wordsmaqaf

print("SP_morph agrees with MT:", twoagree, "{:.3f}".format(twoagree/total), sep="\t")
print("SP_morph disagrees with MT:", disagree, "{:.3f}".format(disagree/total), sep="\t")

# %%
ref_disagree

# %%
from collections import Counter

# %%
words = [item[2] for item in ref_disagree]
word_counts = Counter(words)
sorted_counts = word_counts.most_common()

i=1
for word, count in sorted_counts:
    if i<15:
        print(f"{word}: {count}")
        i=i+1

# %% [markdown]
# ## November 20, 21 experiment - Updating `output_morph_phrase_SP` file

# %% [markdown]
# November 20 experiment - Looking for breaking sequence of verses

# %%
ch_old = None
bo_old = None
ve_expected = 1

discrepancies = []

for verse in F.otype.s('verse'):
    bo, ch, ve = T.sectionFromNode(verse)
    
    if bo != bo_old:
        bo_old = bo
        ch_old = None
        ve_expected = 1

    if ch != ch_old:
        ch_old = ch
        ve_expected = 1

    if ve != ve_expected:
        discrepancies.append((bo, ch, ve_expected, ve))
        print(f"Discrepância encontrada: {bo} {ch}:{ve_expected} -> Encontrado {ve}")
    
    ve_expected += 1

#if discrepancies:
#    print("\nResumo das discrepâncias:")
#    for bo, ch, ve_exp, ve_found in discrepancies:
#        print(f"Livro: {bo}, Capítulo: {ch}, Esperado: {ve_exp}, Encontrado: {ve_found}")
#else:
#    print("Nenhuma discrepância encontrada. Sequência completa e correta.")

# %% [markdown]
# Gaps found in the text:
# - Exodus 29:21 - moved to the end of Exo 29:28
# - Exodus 30:1-10 - removed
# - Deuteronomy 34:3 - removed

# %% [markdown]
# <div>
# <img src="attachment:9ecc60f1-16cc-417b-ab26-e3346aaf5036.png" width="800"/>
# </div>

# %% [markdown]
# <div>
# <img src="attachment:e49f59e9-86cd-494c-8a72-1cf75f64a1bc.png" width="800"/>
# </div>

# %% [markdown]
# Exodus 30:1-10

# %% [markdown]
# <div>
# <img src="attachment:fd3abc25-3dea-4e23-a167-00451a54141f.png" width="800"/>
# </div>

# %% [markdown]
# <div>
# <img src="attachment:67145031-f534-4471-b61d-8b8f257ff282.png" width="800"/>
# </div>

# %%
def load_file_lines(file_path):
    """Load lines from a file and split by tab."""
    with open(file_path, 'r') as f:
        return [line.strip().split('\t') for line in f]

# Load all files
linhas_arquivo1 = load_file_lines('../data/input_morph_MT')
linhas_arquivo2 = load_file_lines('../data/input_morph_SP')
linhas_arquivo4 = load_file_lines('../data/output_morph_phrase_MT')
linhas_arquivo5 = load_file_lines('../new_data/output_morph_phrase_SP_test')
output = open('../new_data/output_morph_phrase_SP_test_up', 'w')

# Create a lookup dictionary for the MT phrase result data
file_dict = {
    (line[0], line[1], line[2]): line[3]
    for line in linhas_arquivo4
}

# Initialize variables
different_classification = set()
skip = 1

# Iterate through the lines of the second file
for index2, linha2 in enumerate(linhas_arquivo2):
    for index1 in range(index2, index2 + skip):
        if index1 >= len(linhas_arquivo4):
            break  # Avoid out-of-range errors

        linha1 = linhas_arquivo1[index1]

        # Check if book, chapter, and verse match
        if linha1[:3] == linha2[:3]:
            words1 = linha1[3].strip().split(' ')
            words2 = linha2[3].strip().split(' ')

            # Check if the words match
            if words1 == words2:
                # Check classification differences
                if linhas_arquivo4[index1][3] != linhas_arquivo5[index2][3]:
                    key = tuple(linhas_arquivo4[index1][:3])
                    different_classification.add((key, file_dict.get(key, None)))
                    print(linhas_arquivo4[index1][0], linhas_arquivo4[index1][1], linhas_arquivo4[index1][2],
                          file_dict[linhas_arquivo4[index1][0], linhas_arquivo4[index1][1], linhas_arquivo4[index1][2]], sep='\t', file=output)
                else:
                    print(linhas_arquivo4[index1][0], linhas_arquivo4[index1][1], linhas_arquivo4[index1][2], linhas_arquivo5[index2][3], sep='\t', file=output)
            else:
                skip += 1
                print(linhas_arquivo4[index1][0], linhas_arquivo4[index1][1], linhas_arquivo4[index1][2], linhas_arquivo5[index2][3], sep='\t', file=output)

output.close()

# %% [markdown]
# Warning! Gap in Exodus 29:21 was manually fixed in the file `output_morph_phrase_SP_test_up` copying results from `output_morph_phrase_SP_test`

# %% [markdown]
# Comparison between first version of the file and improved version (using MT results)
# <div>
# <img src="attachment:11ff06a4-5bc1-41e2-9f85-3681bbef6a81.png" width="1000"/>
# </div>

# %%
inputfilePath = "../new_data/output_morph_phrase_SP"
resultfile = "../new_data/output_morph_phrase_SP_test_up"
outputfilePath = "../new_data/output_morph_phrase_SP_up"

verse_old = None
result_old = ""
book_old = None
chapter_old = None

output = open(outputfilePath, 'w')

linhas_arquivo = []
with open(resultfile, 'r') as f:
    for line in f:
        line = line.strip().split('\t')
        linhas_arquivo.append(line)

file_dict = {}
for line in linhas_arquivo:
    key = (line[0], line[1], line[2])
    value = line[3]
    file_dict[key] = value

with open(inputfilePath, 'r') as input:
    
    inputlines = input.readlines()
    for i in range (0, len(inputlines)):
        word = inputlines[i].strip().split()
        data = word[:5]
        key = tuple(word[1:4])
        result = file_dict[key]
        verse = word[3]
        if verse != verse_old:
            j=0
            verse_old = verse
        else:
            j=j+1
        print(data[0],data[1],data[2],data[3],data[4],result[j],sep='\t',file=output)

output.close()

# %% [markdown]
# New conversion to the dataset - Updated in December 23 (considering mismatch because maqaf between words)

# %%
# Leitura dos arquivos
with open('../new_data/output_morph_phrase_SP_up', 'r') as f1, open('../new_data/node_words_SP', 'r') as f2:
    linhas_arquivo1 = [linha.strip().split('\t') for linha in f1.readlines()]
    linhas_arquivo2 = [linha.strip().split('\t') for linha in f2.readlines()]

# Caracteres a serem removidos
morphsign = "~!>+~[]/<>"
translation_table = str.maketrans('', '', morphsign)

# Dicionário para correspondência entre a coluna 5 do arquivo 1 e a coluna 2 do arquivo 2
dict_arq2 = {}
index = 405426
maqaf = False

# Preenchendo o dicionário com valores de node_words_SP
for coluna1, coluna2 in linhas_arquivo2:
    coluna2 = coluna2.translate(translation_table)
    
    # Separar palavras compostas e armazenar no dicionário
    if '_' in coluna2:
        coluna2, coluna2b = map(str, coluna2.split('_'))
        maqaf = True
    else:
        maqaf = False
    dict_arq2[index] = [coluna1, coluna2, maqaf]
    index = index+1

index = 405426
arquivo1 = {}

for linha in linhas_arquivo1:
    arquivo1[index] = [str(index), linha[4].translate(translation_table), linha[5]]
    index = index+1

index = 405426
newfile = {}
j=0

for i in range(index, index+len(dict_arq2)):
    if dict_arq2[i][2] == True:
        j=j+1
        newfile[i] = [str(i), dict_arq2[i][1], arquivo1[i+j][2]]
    else:
        newfile[i] = [i, dict_arq2[i][1], arquivo1[i+j][2]]

index = 405426
old_value = 'X'
old_node = index

caminho_arquivo3 = '../new_data/phrase_atom_up'
f3 = open(caminho_arquivo3, 'w')

for i in range(index, index+len(newfile)):
    current_value = newfile[i][2]
    current_node = newfile[i][0]
    
    if current_value == 'X' and old_value == 'X':
        pass
    elif current_value == 'X' and old_value == 'Y':
        old_value = current_value
        old_node = current_node
    elif current_value == 'Y' and old_value == 'X':
        print(f"{old_node}-{current_node}", file=f3)
        old_node = current_node
        old_value = current_value
    elif current_value == 'Y' and old_value == 'Y':
        print(f"{current_node}", file=f3)
        old_node = current_node

f3.close()

# %%
# File path for the nodes file
nodes_file_path = '../new_data/phrase_atom_up'

# Function to expand the sequence into a full list of nodes
def expand_sequence(sequence):
    expanded_sequence = []
    for item in sequence:
        item = item.strip()  # Remove any extra whitespace
        if '-' in item:  # Expand ranges
            start, end = map(int, item.split('-'))
            expanded_sequence.extend(range(start, end + 1))
        else:  # Add single nodes
            expanded_sequence.append(int(item))
    return expanded_sequence

# Read the file and collect the sequence
with open(nodes_file_path, "r") as file:
    sequence = file.readlines()

# Expand the sequence read from the file
expanded_sequence = expand_sequence(sequence)

# Function to check for integrity by verifying consecutive nodes
def check_integrity(expanded_sequence):
    for i in range(len(expanded_sequence) - 1):
        if expanded_sequence[i] + 1 != expanded_sequence[i + 1]:
            print(f"Missing or inconsistent node between {expanded_sequence[i]} and {expanded_sequence[i + 1]}")
            return False
    return True

# Run the integrity check
if check_integrity(expanded_sequence):
    print("The sequence in the file is complete and consistent.")
else:
    print("The sequence in the file has missing or inconsistent nodes.")

# %% [markdown]
# ## December 16 - comparing results with MT (updated version)

# %% [markdown]
# Adding the updated results as a new column at the end of the `comparison_output_morph_phrase_SP_output_phrase_space_pdp_SP_MT_word` file

# %%
def load_file_lines(file_path):
    """Load lines from a file and split by tab."""
    with open(file_path, 'r') as f:
        return [line.strip().split('\t') for line in f]

input_file1 = load_file_lines('../new_data/output_morph_phrase_SP_test_up')
input_file2 = load_file_lines('../new_data/comparison_output_morph_phrase_SP_output_phrase_space_pdp_SP_MT_word')
output_file = '../new_data/comparison_output_morph_phrase_SP_output_phrase_space_pdp_SP_MT_word_up'
output = open(output_file, 'w')

num=0

input = {}

for i in input_file1:
    for j in i[3]:
        input[num] = j
        num = num + 1

print(input_file2[0][0],input_file2[0][1],input_file2[0][2],input_file2[0][3],input_file2[0][4],
      input_file2[0][5],input_file2[0][6],input_file2[0][7],input_file2[0][8],input_file2[0][9],'result-w+morph-up', sep='\t', file=output)

num=0
for i in input_file2[1:]:
    if len(i) == 10:
        print(i[0],i[1],i[2],i[3],i[4],i[5],
              i[6],i[7],i[8],i[9],input[num], sep='\t', file=output)
    else:
        print(i[0],i[1],i[2],i[3],i[4],i[5],
              i[6],i[7],'','',input[num], sep='\t', file=output)
    num = num + 1

output.close()

# %%
twoagree = 0
disagree = 0
wordsmaqaf = 0

ref_disagree = set()

with open('../new_data/comparison_output_morph_phrase_SP_output_phrase_space_pdp_SP_MT_word_up', 'r', encoding='utf-8') as file:
    lines = file.readlines()
    
    for line in lines[1:]:  # Ignorar o cabeçalho
        text = line.strip().split('\t')
        SPmorph = text[10] #using updated results in the last column of the file
        if len(text) == 11:
            MT = text[9]
        else:
            MT = ''
        bo = text[1]
        ch = text[2]
        ve = text[3]
        ref = text[0]
        wordSPmorph = text[4]
        wordpdp = text[6][-1]

        if MT != '':
            if SPmorph == MT:
                twoagree = twoagree + 1
            else:
                disagree = disagree + 1
                ref_disagree.add((ref,(bo,ch,ve),wordSPmorph, wordpdp))
        else:
            wordsmaqaf = wordsmaqaf + 1

ref_disagree = sorted(ref_disagree, key=lambda x: int(x[0]))

total = len(lines) - wordsmaqaf

print("SP_morph agrees with MT:", twoagree, "{:.3f}".format(twoagree/total), sep="\t")
print("SP_morph disagrees with MT:", disagree, "{:.3f}".format(disagree/total), sep="\t")

# %% [markdown]
# Previous result:\
# SP_morph agrees with MT: 102653 (0.916)\
# SP_morph disagrees with MT: 9360 (0.084)

# %%
ref_disagree

# %% [markdown]
# ## December 16, 21, 23 - Manual check

# %% [markdown]
# Choosing 100 verses randomly that were not previously updated, and checking its `phrase_atom` structure, by comparison with the BHSA

# %%
import random

# %%
def load_file_lines(file_path):
    """Load lines from a file and split by tab."""
    with open(file_path, 'r') as f:
        return [line.strip().split('\t') for line in f]

initial_file = load_file_lines("../new_data/output_morph_phrase_SP_test")
final_file = load_file_lines("../new_data/output_morph_phrase_SP_test_up")

ref_nonupdated = {}

initial = {}

for num, line in enumerate(initial_file):
    book, chapter, verse = line[0], int(line[1]), int(line[2])
    initial[(book, chapter, verse)] = line[3:]

for num, line in enumerate(final_file):
    book, chapter, verse = line[0], int(line[1]), int(line[2])
    final[(book, chapter, verse)] = line[3:]

while len(ref_nonupdated) < 100:
    ref = random.choice(list(initial.keys()))
    if ref in final and initial[ref] == final[ref]:
        ref_nonupdated[ref] = initial[ref]

# %%
ref_nonupdated

# %%
ref_nonupdated = {('Genesis', 19, 28): ['YYXXXXYYXXXXXXYYYYYYXXYXXXY'],
 ('Leviticus', 1, 7): ['YYXXXYYXXYYYYXXY'],
 ('Numbers', 31, 53): ['XXYYYY'],
 ('Genesis', 14, 14): ['YYYYYYYYXYXYXXXXYYYXY'],
 ('Exodus', 2, 11): ['YYXXXYYYYYYXYYYXYYYXYYXYXY'],
 ('Genesis', 42, 29): ['YYXYYXYYYYXYYYYXY'],
 ('Leviticus', 22, 4): ['XYXXYYYYYYXYYYXYYYYYXXXYYYYYYXY'],
 ('Deuteronomy', 25, 5): ['YYYYYYYYYYYYYYYYYXYXYXYYYYYYXYYY'],
 ('Numbers', 7, 30): ['XXXYYXXYYXY'],
 ('Exodus', 14, 19): ['YYXXYYYXXXYYYXYYYXXYXYYYXY'],
 ('Numbers', 21, 15): ['YYYYYXYYYYXXY'],
 ('Genesis', 4, 4): ['YYYXYXXXXXYYYYXXXXX'],
 ('Exodus', 21, 19): ['YYYYXYXYYYYYXYYYYY'],
 ('Numbers', 26, 50): ['YXYXXYYXXXYXXY'],
 ('Numbers', 23, 17): ['YYYYYYXYYXYYYYYYYYY'],
 ('Numbers', 20, 4): ['YYYXXYXXXXYXYYXXY'],
 ('Deuteronomy', 4, 29): ['YYXYXYYYYYYXXXYXXY'],
 ('Exodus', 37, 20): ['YXYXXYXXY'],
 ('Leviticus', 24, 18): ['YYXYYYXY'],
 ('Deuteronomy', 27, 19): ['YYXXXXYYYXXYY'],
 ('Genesis', 43, 21): ['YYYYXXYYYXYYYXYXXYYXYYYYXY'],
 ('Exodus', 16, 32): ['YYYYXYYYYYXYYXYXYYYXXYYYYXYXYYXXY'],
 ('Deuteronomy', 4, 3): ['YYYXYYYXXYYXXYYYXXYYYYXY'],
 ('Genesis', 24, 52): ['YYXYYXYXYYYYXY'],
 ('Numbers', 3, 21): ['XYXXYYXXYYYXXY'],
 ('Deuteronomy', 4, 14): ['YYYYXXXYXYYXXYXYYXYYYYYXY'],
 ('Numbers', 22, 16): ['YYXYYYYYYYXYYYYXYY'],
 ('Leviticus', 3, 10): ['YXXXXXXXYYYYXXXYXXYXXYXXYY'],
 ('Leviticus', 10, 12): ['YYYXXXXYYXYYXYYXXYYYXXYYYXXXYYXYY'],
 ('Genesis', 47, 6): ['XYXYYXXXYYXYYXYYXXYYYYYYYXYYYXYXYY'],
 ('Leviticus', 14, 22): ['YXYYXXYYYYYYXYYYY'],
 ('Genesis', 27, 45): ['XYXYYYYXYYYYYYYXYYYXYXY'],
 ('Deuteronomy', 17, 7): ['XXYYYXYXYYXXXYXYYYXYXY'],
 ('Genesis', 24, 17): ['YYXYXYYYYYXYXY'],
 ('Genesis', 8, 4): ['YYXYXXXYXXXYXYXXY'],
 ('Exodus', 14, 12): ['YYYXYYYYXYXYYYYYYXYYYYYXYXYXY'],
 ('Genesis', 19, 1): ['YYXXYYXYYYYXXYYYYYYXYYYYY'],
 ('Genesis', 30, 3): ['YYYYYYYYYXYYYXYY'],
 ('Genesis', 33, 12): ['YYYYYYYXY'],
 ('Genesis', 40, 13): ['XYXYYYXYYYXYYYXYXYXXXYYYY'],
 ('Exodus', 28, 12): ['YYXXXYXXXYXYYXXYYYYXYXXYXXYXY'],
 ('Leviticus', 24, 7): ['YYXXXYYYYXYXYYXY'],
 ('Genesis', 21, 8): ['YYXYYYYYYXYXYYXYY'],
 ('Exodus', 34, 1): ['YYYXYYYXXYXYYYXXYXXYYYXXXXYYY'],
 ('Numbers', 26, 31): ['YYXXYYYXXY'],
 ('Exodus', 38, 9): ['YYXXYXXYYXXYYYYXY'],
 ('Numbers', 10, 16): ['YXXXXYYXY'],
 ('Deuteronomy', 1, 17): ['YYYXYXYXYYYYXXYYXYXYYYXYYYYYYYY'],
 ('Numbers', 35, 3): ['YYXYYXYYYYXXXXXXXXY'],
 ('Deuteronomy', 2, 28): ['YXYYYYYYXYYYYYYYXY'],
 ('Exodus', 8, 12): ['YYYXYYXYYXYXYYYXXXYYYXYXXXY'],
 ('Deuteronomy',
  10,
  7): ['XYYYYXYXYYYYXYYXYXYYYYXYXYYYYXYYXYYYYXXYYYXYYYYXXXYYYYYYYYYYYYY'],
 ('Leviticus', 24, 15): ['YXXYYXYXYYYYYYY'],
 ('Exodus', 38, 25): ['YYXXYXXXXXXXXXYXX'],
 ('Numbers', 6, 1): ['YYYXYXY'],
 ('Leviticus', 8, 24): ['YYXXYYYYXXYXYXXYYXXXXYYXXXXYYYYXXYXXXY'],
 ('Exodus', 9, 24): ['YYXYYYYXXXXXYYYYYXYXYYXY'],
 ('Deuteronomy', 23, 5): ['XYYYYYXYYXYXYXYXYYYYYXYXYYXYXY'],
 ('Leviticus', 14, 7): ['YYXYYXXYXYYYYYXXXXYXXXY'],
 ('Numbers', 3, 5): ['YYYXYXY'],
 ('Numbers', 34, 5): ['YYXYXYXYYYYXY'],
 ('Numbers', 14, 29): ['XXXYYXYXYXXYXXXYYYYYY'],
 ('Leviticus', 8, 30): ['YYYXXXYYXXYYXXYYYXXXXXXXYYXXYYYYXXXXXXXXYXXYY'],
 ('Genesis', 31, 35): ['YYXYYYXXYYYYXYXYYXXYYYYYYYXXY'],
 ('Leviticus', 7, 10): ['YXYYXYYYXXXYYYXY'],
 ('Numbers', 33, 1): ['YXXYYYXXYXYXXXXY'],
 ('Numbers', 26, 30): ['YXYYXXYYYXXY'],
 ('Leviticus', 11, 28): ['YYYXYYYYYXXXYYY'],
 ('Genesis', 35, 16): ['YYXYYYYYXXYXYYYYYYYXY'],
 ('Exodus', 22, 11): ['YYYXYYYXY'],
 ('Numbers', 28, 13): ['YYYYYYXYXXXYYXYYXY'],
 ('Genesis', 35, 22): ['YYXYYXXXYYYYYYXYYYYYYYYXYXX'],
 ('Leviticus', 23, 44): ['YYYXXYXXY'],
 ('Deuteronomy', 2, 24): ['YYYYXXYYYXYXYXYXYYXYYYYYYY'],
 ('Numbers', 14, 24): ['YYXYYXYYYYYYYXXYYYYYYY'],
 ('Leviticus', 27, 33): ['YYXYXYYYYYYYYYYYYYYYYY'],
 ('Genesis', 25, 17): ['YYXXYXXXXXXXXYYYYYYYYY'],
 ('Numbers', 14, 30): ['YYYXXYYYXYXYYYXYYXYYYXY'],
 ('Deuteronomy', 29, 25): ['YYYYXYYYYYYYYYYYY'],
 ('Exodus', 40, 34): ['YYXYXXYYXYYXXY'],
 ('Leviticus', 3, 7): ['YYYYXYYYYXXY'],
 ('Exodus', 36, 22): ['XYXXXYYYXYYYXXXXY'],
 ('Numbers', 34, 7): ['YYYYYYXXXXYYYXXY'],
 ('Numbers', 6, 13): ['YYXXYXYYXYYYXXXY'],
 ('Deuteronomy', 8, 4): ['YYYXYYYYYYXY'],
 ('Deuteronomy', 27, 4): ['YYXYXXYYXXXXYYYYYXYXYYYYXY'],
 ('Deuteronomy', 26, 5): ['YYYYXXYYYYYYYYYYYXXYYYYXXXXXXY'],
 ('Numbers', 33, 31): ['YYXYYYXXY'],
 ('Leviticus', 26, 23): ['YYXYYYYYYYY'],
 ('Leviticus', 11, 11): ['YYYYXYYYYXYY'],
 ('Numbers', 4, 43): ['XXXYYYYXXXYYYYXYXYXXY'],
 ('Exodus', 26, 19): ['YYXYYXXXYXYXXXXYXXYYXYXXXXYXXY'],
 ('Leviticus', 25, 31): ['YXXYYYYYYXXXYYYYYYXYY'],
 ('Leviticus', 20, 11): ['YYYYXXYXYYYYYYY'],
 ('Exodus', 22, 24): ['YYYXYXYYYYYXYYYYY'],
 ('Exodus', 22, 29): ['YYXYYXYXYYXYYXXXYYY'],
 ('Leviticus', 7, 8): ['YXYYYXXXXXYYYXYYY'],
 ('Genesis', 47, 26): ['YYYYXYXXXXYXXYXYXYXXXYXYYYXY'],
 ('Numbers', 4, 12): ['YYXXXXYYYYXYYYXXYYYYXXXYYYXXY'],
 ('Exodus', 10, 24): ['YYYXXXXYYYYYXYXYYYYXYYY']}

# %%
ref_nonupdated_sort = dict(sorted(ref_nonupdated.items(), key=lambda item: item[0]))

# %% [markdown]
# Looking for nonupdated texts in SP which are different to MT

# %%
def load_file_lines(file_path):
    """Load lines from a file and split by tab."""
    with open(file_path, 'r') as f:
        return [line.strip().split('\t') for line in f]

MT_file = load_file_lines("../data/output_phrase_space_MT")

MT = {}

for num, line in enumerate(MT_file):
    book, chapter, verse = line[0], int(line[1]), int(line[2])
    MT[(book, chapter, verse)] = line[3:]

i=1

print('SP and MT texts are different')
for ref in ref_nonupdated_sort:
    if ref_nonupdated_sort[ref] != MT[ref]:
        print(i,ref,ref_nonupdated_sort[ref], MT[ref], sep='\t')
        i = i+1

# %%
def load_file_lines(file_path):
    """Load lines from a file and split by tab."""
    with open(file_path, 'r') as f:
        return [line.strip().split('\t') for line in f]

MT_file = load_file_lines("../data/output_phrase_space_MT")

MT = {}

for num, line in enumerate(MT_file):
    book, chapter, verse = line[0], int(line[1]), int(line[2])
    MT[(book, chapter, verse)] = line[3:]

i=1

print('SP and MT texts are equal')
for ref in ref_nonupdated_sort:
    if ref_nonupdated_sort[ref] == MT[ref]:
        print(i,ref,ref_nonupdated_sort[ref], MT[ref], sep='\t')
        i = i+1

# %%
SP = use('saulocantanhede/sp', version=3.5)
Fsp, Lsp, Tsp = SP.api.F, SP.api.L, SP.api.T

MT = use('etcbc/bhsa')
MT.load(['g_prs', 'g_nme', 'g_pfm', 'g_vbs', 'g_vbe'])
F, L, T = MT.api.F, MT.api.L, MT.api.T

SP_old = use("app:~/github/saulocantanhede/sp/app", locations="/Users/saulo/github/saulocantanhede/sp/tf", version="3.5", hoist=globals())

# %% [markdown]
# #### SP text is different of MT text

# %% [markdown]
# ##### Genesis 19:28

# %%
results = MT.search("""
book book=Genesis
    chapter chapter=19
        verse verse=28
            word
""")
MT.show(results, end=4, multiFeatures=False, queryFeatures=False, condensed=True, hiddenTypes={'phrase','half_verse','subphrase', 'sentence', 'sentence_atom', 'clause', 'clause_atom'})

# %%
results = SP.search("""
book book=Genesis
    chapter chapter=19
        verse verse=28
            word
""")
SP.show(results, end=4, multiFeatures=False, queryFeatures=False, condensed=True, withNodes=True)

# %%
results = SP_old.search("""
book book=Genesis
    chapter chapter=19
        verse verse=28
            word
""")
SP_old.show(results, end=4, multiFeatures=False, queryFeatures=False, condensed=True, withNodes=False)

# %% [markdown]
# <div>
# <img src="attachment:42f0822f-51b0-47e4-a79d-d50e587f215e.png" width="1000"/>
# </div>

# %% [markdown]
# <div>
# <img src="attachment:39c2d5b8-4a71-410f-a33b-ba15f8bb37b2.png" width="350"/>
# </div>

# %% [markdown]
# ##### Leviticus 1:7

# %%
results = MT.search("""
book book=Leviticus
    chapter chapter=1
        verse verse=7
            word
""")
MT.show(results, end=4, multiFeatures=False, queryFeatures=False, condensed=True, hiddenTypes={'phrase','half_verse','subphrase', 'sentence', 'sentence_atom', 'clause', 'clause_atom'})

# %%
results = SP.search("""
book book=Leviticus
    chapter chapter=1
        verse verse=7
            word
""")
SP.show(results, end=4, multiFeatures=False, queryFeatures=False, condensed=True, withNodes=False)

# %% [markdown]
# <div>
# <img src="attachment:3afa0a93-ea57-4a1b-9bb8-bbc2f2de6b1d.png" width="800"/>
# </div>

# %% [markdown]
# <div>
# <img src="attachment:360a8431-1b2f-43e3-ae07-4a424c2df6c9.png" width="350"/>
# </div>

# %% [markdown]
# ##### Numbers 21:15

# %%
results = MT.search("""
book book=Numeri
    chapter chapter=21
        verse verse=15
            word
""")
MT.show(results, end=4, multiFeatures=False, queryFeatures=False, condensed=True, hiddenTypes={'phrase','half_verse','subphrase', 'sentence', 'sentence_atom', 'clause', 'clause_atom'})

# %%
results = SP.search("""
book book=Numbers
    chapter chapter=21
        verse verse=15
            word
""")
SP.show(results, end=4, multiFeatures=False, queryFeatures=False, condensed=True, withNodes=False)

# %% [markdown]
# <div>
# <img src="attachment:20f534bf-86be-4c70-85e3-c741d143b7e0.png" width="800"/>
# </div>

# %% [markdown]
# <div>
# <img src="attachment:db4c5e5e-1b61-404c-886b-41f9c415117d.png" width="350"/>
# </div>

# %% [markdown]
# #### SP text is equal to MT text

# %% [markdown]
# ##### Numbers 31:53

# %%
results = MT.search("""
book book=Numeri
    chapter chapter=31
        verse verse=53
            word
""")
MT.show(results, end=4, multiFeatures=False, queryFeatures=False, condensed=True, hiddenTypes={'phrase','half_verse','subphrase', 'sentence', 'sentence_atom', 'clause', 'clause_atom'})

# %%
results = SP.search("""
book book=Numbers
    chapter chapter=31
        verse verse=53
            word
""")
SP.show(results, end=4, multiFeatures=False, queryFeatures=False, condensed=True, withNodes=False)

# %% [markdown]
# <div>
# <img src="attachment:5d237366-79ae-4305-80a3-8806983fa408.png" width="600"/>
# </div>

# %% [markdown]
# <div>
# <img src="attachment:c099e28e-2ba8-4744-b598-7b7642af3e61.png" width="400"/>
# </div>

# %% [markdown]
# ##### Numbers 7:30

# %%
results = MT.search("""
book book=Numeri
    chapter chapter=7
        verse verse=30
            word
""")
MT.show(results, end=4, multiFeatures=False, queryFeatures=False, condensed=True, hiddenTypes={'phrase','half_verse','subphrase', 'sentence', 'sentence_atom', 'clause', 'clause_atom'})

# %%
results = SP.search("""
book book=Numbers
    chapter chapter=7
        verse verse=30
            word
""")
SP.show(results, end=4, multiFeatures=False, queryFeatures=False, condensed=True, withNodes=False)

# %% [markdown]
# <div>
# <img src="attachment:01c20745-e9b9-4a41-9a28-9ec09361a764.png" width="800"/>
# </div>

# %% [markdown]
# <div>
# <img src="attachment:6961ca76-c5a4-441b-a029-e84e5ed93a1e.png" width="400"/>
# </div>

# %% [markdown]
# ##### Exodus 21:19

# %%
results = MT.search("""
book book=Exodus
    chapter chapter=21
        verse verse=19
            word
""")
MT.show(results, end=4, multiFeatures=False, queryFeatures=False, condensed=True, hiddenTypes={'phrase','half_verse','subphrase', 'sentence', 'sentence_atom', 'clause', 'clause_atom'})

# %%
results = SP.search("""
book book=Exodus
    chapter chapter=21
        verse verse=19
            word
""")
SP.show(results, end=4, multiFeatures=False, queryFeatures=False, condensed=True, withNodes=False)

# %% [markdown]
# <div>
# <img src="attachment:89519e05-2152-4bc7-9015-5620bcc387bb.png" width="900"/>
# </div>

# %% [markdown]
# <div>
# <img src="attachment:3c96965c-9c2f-4bee-86d8-583d1e6ee9cd.png" width="400"/>
# </div>

# %% [markdown]
# #### Identifying the issue with maqaf (again)

# %% [markdown]
# The issue of the mismatch in the classification of the words happens first in Genesis 4:22

# %%
results = MT.search("""
book book=Genesis
    chapter chapter=4
        verse verse=22
            word
""")
MT.show(results, end=4, multiFeatures=False, queryFeatures=False, condensed=True, hiddenTypes={'phrase','half_verse','subphrase', 'sentence', 'sentence_atom', 'clause', 'clause_atom'})

# %%
results = SP.search("""
book book=Genesis
    chapter chapter=4
        verse verse=22
            word
""")
SP.show(results, end=4, multiFeatures=False, queryFeatures=False, condensed=True, withNodes=True)

# %%
for i in range(407450, 407467):
    print(dict_arq2[i], arquivo1[i], newfile[i], sep='\t\t\t')

# %%
for i in range(407450, 407467):
    print(dict_arq2[i], arquivo1[i], newfile[i], sep='\t\t\t')

# %% [markdown]
# #### Conclusion

# %% [markdown]
# About 50% (55/100) of the randomly selected SP verses are equal to MT and have the same classification as MT.\
# The rest of the verses (45/100) differ from MT and present a different classification compared to MT.\
# Considering only different verses, we have verses (12/45) with similar structures (e.g., differing only by a few words or switching the position of words compared to MT), which present the same classification compared to MT.

# %% [markdown]
# ## December 26 - Manual check in Exodus 20

# %% [markdown]
# Looking for SP verses with a different classification than the MT:

# %%
def load_file_lines(file_path):
    """Load lines from a file and split by tab."""
    with open(file_path, 'r') as f:
        return [line.strip().split('\t') for line in f]

SP_file = load_file_lines("../new_data/output_morph_phrase_SP_test_up")
MT_file = load_file_lines("../data/output_phrase_space_MT")

SP_result = {}

for num, line in enumerate(SP_file):
    book, chapter, verse = line[0], int(line[1]), int(line[2])
    SP_result[(book, chapter, verse)] = line[3:]

MT_result = {}

for num, line in enumerate(MT_file):
    book, chapter, verse = line[0], int(line[1]), int(line[2])
    MT_result[(book, chapter, verse)] = line[3:]

i = 1

for (book, chapter, verse), value in SP_result.items():
    if book == 'Exodus' and chapter == 20:
        ref = (book, chapter, verse)
        if MT_result[ref] != SP_result[ref]:
            print(i, ref, sep='\t')
            i = i+1

# %% [markdown]
# ## January 3 - Updating verses ending with open phrase atoms boundaries

# %%
with open('../new_data/output_morph_phrase_SP_test_up', 'r') as f1:
    linhas_arquivo1 = [linha.strip().split('\t') for linha in f1.readlines()]

output_file = '../new_data/output_morph_phrase_SP_test_up2'
output = open(output_file, 'w')

for linha in linhas_arquivo1:
    if linha[3][-1] != 'Y':
        newvalue = list(linha[3])
        newvalue[-1] = 'Y'
        print(linha[0], linha[1], linha[2], "".join(newvalue), sep='\t', file=output)
    else:
        print(linha[0], linha[1], linha[2], linha[3], sep='\t', file=output)

output.close()

# %%
inputfilePath = "../new_data/output_morph_phrase_SP"
resultfile = "../new_data/output_morph_phrase_SP_test_up2"
outputfilePath = "../new_data/output_morph_phrase_SP_up2"

verse_old = None
result_old = ""
book_old = None
chapter_old = None

output = open(outputfilePath, 'w')

linhas_arquivo = []
with open(resultfile, 'r') as f:
    for line in f:
        line = line.strip().split('\t')
        linhas_arquivo.append(line)

file_dict = {}
for line in linhas_arquivo:
    key = (line[0], line[1], line[2])
    value = line[3]
    file_dict[key] = value

with open(inputfilePath, 'r') as input:
    
    inputlines = input.readlines()
    for i in range (0, len(inputlines)):
        word = inputlines[i].strip().split()
        data = word[:5]
        key = tuple(word[1:4])
        result = file_dict[key]
        verse = word[3]
        if verse != verse_old:
            j=0
            verse_old = verse
        else:
            j=j+1
        print(data[0],data[1],data[2],data[3],data[4],result[j],sep='\t',file=output)

output.close()

# %%
# Leitura dos arquivos
with open('../new_data/output_morph_phrase_SP_up2', 'r') as f1, open('../new_data/node_words_SP', 'r') as f2:
    linhas_arquivo1 = [linha.strip().split('\t') for linha in f1.readlines()]
    linhas_arquivo2 = [linha.strip().split('\t') for linha in f2.readlines()]

# Caracteres a serem removidos
morphsign = "~!>+~[]/<>"
translation_table = str.maketrans('', '', morphsign)

# Dicionário para correspondência entre a coluna 5 do arquivo 1 e a coluna 2 do arquivo 2
dict_arq2 = {}
index = 405426
maqaf = False

# Preenchendo o dicionário com valores de node_words_SP
for coluna1, coluna2 in linhas_arquivo2:
    coluna2 = coluna2.translate(translation_table)
    
    # Separar palavras compostas e armazenar no dicionário
    if '_' in coluna2:
        coluna2, coluna2b = map(str, coluna2.split('_'))
        maqaf = True
    else:
        maqaf = False
    dict_arq2[index] = [coluna1, coluna2, maqaf]
    index = index+1

index = 405426
arquivo1 = {}

for linha in linhas_arquivo1:
    arquivo1[index] = [str(index), linha[4].translate(translation_table), linha[5]]
    index = index+1

index = 405426
newfile = {}
j=0

for i in range(index, index+len(dict_arq2)):
    if dict_arq2[i][2] == True:
        j=j+1
        newfile[i] = [str(i), dict_arq2[i][1], arquivo1[i+j][2]]
    else:
        newfile[i] = [i, dict_arq2[i][1], arquivo1[i+j][2]]

index = 405426
old_value = 'X'
old_node = index

caminho_arquivo3 = '../new_data/phrase_atom_up2'
f3 = open(caminho_arquivo3, 'w')

for i in range(index, index+len(newfile)):
    current_value = newfile[i][2]
    current_node = newfile[i][0]
    
    if current_value == 'X' and old_value == 'X':
        pass
    elif current_value == 'X' and old_value == 'Y':
        old_value = current_value
        old_node = current_node
    elif current_value == 'Y' and old_value == 'X':
        print(f"{old_node}-{current_node}", file=f3)
        old_node = current_node
        old_value = current_value
    elif current_value == 'Y' and old_value == 'Y':
        print(f"{current_node}", file=f3)
        old_node = current_node

f3.close()

# %%
# File path for the nodes file
nodes_file_path = '../new_data/phrase_atom_up2'

# Function to expand the sequence into a full list of nodes
def expand_sequence(sequence):
    expanded_sequence = []
    for item in sequence:
        item = item.strip()  # Remove any extra whitespace
        if '-' in item:  # Expand ranges
            start, end = map(int, item.split('-'))
            expanded_sequence.extend(range(start, end + 1))
        else:  # Add single nodes
            expanded_sequence.append(int(item))
    return expanded_sequence

# Read the file and collect the sequence
with open(nodes_file_path, "r") as file:
    sequence = file.readlines()

# Expand the sequence read from the file
expanded_sequence = expand_sequence(sequence)

# Function to check for integrity by verifying consecutive nodes
def check_integrity(expanded_sequence):
    for i in range(len(expanded_sequence) - 1):
        if expanded_sequence[i] + 1 != expanded_sequence[i + 1]:
            print(f"Missing or inconsistent node between {expanded_sequence[i]} and {expanded_sequence[i + 1]}")
            return False
    return True

# Run the integrity check
if check_integrity(expanded_sequence):
    print("The sequence in the file is complete and consistent.")
else:
    print("The sequence in the file has missing or inconsistent nodes.")

# %% [markdown]
# Adding the updated results as a new column at the end of the `comparison_output_morph_phrase_SP_output_phrase_space_pdp_SP_MT_word_up` file

# %%
def load_file_lines(file_path):
    """Load lines from a file and split by tab."""
    with open(file_path, 'r') as f:
        return [line.strip().split('\t') for line in f]

input_file1 = load_file_lines('../new_data/output_morph_phrase_SP_test_up2')
input_file2 = load_file_lines('../new_data/comparison_output_morph_phrase_SP_output_phrase_space_pdp_SP_MT_word')
output_file = '../new_data/comparison_output_morph_phrase_SP_output_phrase_space_pdp_SP_MT_word_up2'
output = open(output_file, 'w')

num=0

input = {}

for i in input_file1:
    for j in i[3]:
        input[num] = j
        num = num + 1

print(input_file2[0][0],input_file2[0][1],input_file2[0][2],input_file2[0][3],input_file2[0][4],
      input_file2[0][5],input_file2[0][6],input_file2[0][7],input_file2[0][8],input_file2[0][9],'result-w+morph-up', sep='\t', file=output)

num=0
for i in input_file2[1:]:
    if len(i) == 10:
        print(i[0],i[1],i[2],i[3],i[4],i[5],
              i[6],i[7],i[8],i[9],input[num], sep='\t', file=output)
    else:
        print(i[0],i[1],i[2],i[3],i[4],i[5],
              i[6],i[7],'','',input[num], sep='\t', file=output)
    num = num + 1

output.close()

# %%
twoagree = 0
disagree = 0
wordsmaqaf = 0

ref_disagree = set()

with open('../new_data/comparison_output_morph_phrase_SP_output_phrase_space_pdp_SP_MT_word_up2', 'r', encoding='utf-8') as file:
    lines = file.readlines()
    
    for line in lines[1:]:  # Ignorar o cabeçalho
        text = line.strip().split('\t')
        SPmorph = text[10] #using updated results in the last column of the file
        if len(text) == 11:
            MT = text[9]
        else:
            MT = ''
        bo = text[1]
        ch = text[2]
        ve = text[3]
        ref = text[0]
        wordSPmorph = text[4]
        wordpdp = text[6][-1]

        if MT != '':
            if SPmorph == MT:
                twoagree = twoagree + 1
            else:
                disagree = disagree + 1
                ref_disagree.add((ref,(bo,ch,ve),wordSPmorph, wordpdp))
        else:
            wordsmaqaf = wordsmaqaf + 1

ref_disagree = sorted(ref_disagree, key=lambda x: int(x[0]))

total = len(lines) - wordsmaqaf

print("SP_morph agrees with MT:", twoagree, "{:.3f}".format(twoagree/total), sep="\t")
print("SP_morph disagrees with MT:", disagree, "{:.3f}".format(disagree/total), sep="\t")

# %%
ref_disagree

# %% [markdown]
# ## January 6, 16, 20, 21 - studying the differences between MT and SP phrase atom classification and length of verses

# %%
def load_file_lines(file_path):
    """Load lines from a file and split by tab."""
    with open(file_path, 'r') as f:
        return [line.strip().split('\t') for line in f]

# Load all files
linhas_arquivo1 = load_file_lines('../data/input_morph_MT')
linhas_arquivo2 = load_file_lines('../data/input_morph_SP')
linhas_arquivo4 = load_file_lines('../data/output_morph_phrase_MT')
linhas_arquivo5 = load_file_lines('../new_data/output_morph_phrase_SP_test_up2')
output = open('../new_data/output_morph_phrase_SP_test_up3', 'w')

# Create a lookup dictionary for the MT phrase result data
file_dict = {
    (line[0], line[1], line[2]): line[3]
    for line in linhas_arquivo4
}

# Initialize variables
different_classification = set()
skip = 1
different_number_words = set()
samelength_different_number_words = set()
diff_letters_words = set()

# Iterate through the lines of the second file
for index2, linha2 in enumerate(linhas_arquivo2):
    for index1 in range(index2, index2 + skip):
        if index1 >= len(linhas_arquivo4):
            break  # Avoid out-of-range errors

        linha1 = linhas_arquivo1[index1]

        # Check if book, chapter, and verse match
        if linha1[:3] == linha2[:3]:
            words1 = linha1[3].strip().split(' ') #morphological MT text
            words2 = linha2[3].strip().split(' ') #morphological SP text

            # Check if the words match (matching of words in a verse imply that the number of words also match)
            if words1 == words2:
                # Check classification differences
                if linhas_arquivo4[index1][3] != linhas_arquivo5[index2][3]:
                    key = tuple(linhas_arquivo4[index1][:3])
                    different_classification.add((key, file_dict.get(key, None)))
                    print(linhas_arquivo4[index1][0], linhas_arquivo4[index1][1], linhas_arquivo4[index1][2],
                          file_dict[linhas_arquivo4[index1][0], linhas_arquivo4[index1][1], linhas_arquivo4[index1][2]], sep='\t', file=output)
                else:
                    #pass
                    print(linhas_arquivo4[index1][0], linhas_arquivo4[index1][1], linhas_arquivo4[index1][2], linhas_arquivo5[index2][3], sep='\t', file=output)
            else:
                skip += 1
                #print(linhas_arquivo4[index1][0], linhas_arquivo4[index1][1], linhas_arquivo4[index1][2], linhas_arquivo5[index2][3], sep='\t')

            if len(words1) == len(words2) and words1 != words2:
                if linhas_arquivo4[index1][3] != linhas_arquivo5[index2][3]:
                    #pass
                    #print((linhas_arquivo4[index1][0], linhas_arquivo4[index1][1], linhas_arquivo4[index1][2]), sep='\t')
                    samelength_different_number_words.add((linhas_arquivo4[index1][0], linhas_arquivo4[index1][1], linhas_arquivo4[index1][2]))
                    print(linhas_arquivo4[index1][0], linhas_arquivo4[index1][1], linhas_arquivo4[index1][2],
                          file_dict[linhas_arquivo4[index1][0], linhas_arquivo4[index1][1], linhas_arquivo4[index1][2]], sep='\t', file=output)
                    
                    diff_letters = 0
                    for w1, w2 in zip(words1, words2):
                        diff_letters = len(set(w1) ^ set(w2)) + diff_letters
                    print(diff_letters,(linhas_arquivo4[index1][0], linhas_arquivo4[index1][1], linhas_arquivo4[index1][2]))
                    diff_letters_words.add((diff_letters,(linhas_arquivo4[index1][0], linhas_arquivo4[index1][1], linhas_arquivo4[index1][2])))
                    
                else:
                    #pass
                    print(linhas_arquivo4[index1][0], linhas_arquivo4[index1][1], linhas_arquivo4[index1][2], linhas_arquivo5[index2][3], sep='\t', file=output)
                    
            if len(words1) != len(words2) and words1 != words2:
                different_number_words.add((len(words2)-len(words1), (linhas_arquivo4[index1][0], linhas_arquivo4[index1][1], linhas_arquivo4[index1][2])))
                #print(len(words2)-len(words1), (linhas_arquivo4[index1][0], linhas_arquivo4[index1][1], linhas_arquivo4[index1][2]), sep='\t')
                print(linhas_arquivo4[index1][0], linhas_arquivo4[index1][1], linhas_arquivo4[index1][2], linhas_arquivo5[index2][3], sep='\t', file=output)
            
#different_number_words = sorted(different_number_words, key=lambda x: (x[0], x[1]))

output.close()

# %%
diff_letters_words = sorted(diff_letters_words, key=lambda x: (x[0], x[1]))

# %%
filtered_elements = [element for element in diff_letters_words if element[0] == 4]
for element in filtered_elements:
    print(element)

# %%
from collections import Counter

# %%
counts = Counter(element[0] for element in diff_letters_words)
for number, count in counts.items():
    print(f"Difference between SP and MT: {number} letters - {count} occurrences")
    filtered_elements = [element for element in diff_letters_words if element[0] == number]
    for element in filtered_elements:
        print(f"\t{element[1][0]} {element[1][1]}:{element[1][2]}")

# %%
samelength_different_number_words

# %%
len(samelength_different_number_words)

# %%
len(different_number_words)

# %%
import random

# %%
ref_samelength_different_number_words = set()

while len(ref_samelength_different_number_words) < 100:
    ref = random.choice(list(samelength_different_number_words))
    ref_samelength_different_number_words.add(ref)

# %%
ref_samelength_different_number_words

# %%
different_number_words

# %%
from collections import Counter

# %%
primeiros_elementos = [item[0] for item in different_number_words]
frequencia = Counter(primeiros_elementos)
for numero, contagem in frequencia.items():
    print(f"Difference between SP and MT: {numero} - {contagem} ocorrences")

# %%
primeiros_elementos = [item[0] for item in different_number_words]
frequencia = Counter(primeiros_elementos)
for numero, contagem in frequencia.items():
    print(f"Difference between SP and MT: {numero} - {contagem} ocorrences")
    values = [item[1] for item in different_number_words if item[0] == numero]
    for valor in values:
        print(f"\t{valor[0]} {valor[1]}:{valor[2]}")

# %% [markdown]
# ## January 16 - first attempt of manual correction
# 
# This is an update in the dataset (considering verses with the same length in MT and SP but with slight difference in the words of the verse)

# %%
inputfilePath = "../new_data/output_morph_phrase_SP"
resultfile = "../new_data/output_morph_phrase_SP_test_up3"
outputfilePath = "../new_data/output_morph_phrase_SP_up3"

verse_old = None
result_old = ""
book_old = None
chapter_old = None

output = open(outputfilePath, 'w')

linhas_arquivo = []
with open(resultfile, 'r') as f:
    for line in f:
        line = line.strip().split('\t')
        linhas_arquivo.append(line)

file_dict = {}
for line in linhas_arquivo:
    key = (line[0], line[1], line[2])
    value = line[3]
    file_dict[key] = value

with open(inputfilePath, 'r') as input:
    
    inputlines = input.readlines()
    for i in range (0, len(inputlines)):
        word = inputlines[i].strip().split()
        data = word[:5]
        key = tuple(word[1:4])
        result = file_dict[key]
        verse = word[3]
        if verse != verse_old:
            j=0
            verse_old = verse
        else:
            j=j+1
        print(data[0],data[1],data[2],data[3],data[4],result[j],sep='\t',file=output)

output.close()

# %%
# Leitura dos arquivos
with open('../new_data/output_morph_phrase_SP_up3', 'r') as f1, open('../new_data/node_words_SP', 'r') as f2:
    linhas_arquivo1 = [linha.strip().split('\t') for linha in f1.readlines()]
    linhas_arquivo2 = [linha.strip().split('\t') for linha in f2.readlines()]

# Caracteres a serem removidos
morphsign = "~!>+~[]/<>"
translation_table = str.maketrans('', '', morphsign)

# Dicionário para correspondência entre a coluna 5 do arquivo 1 e a coluna 2 do arquivo 2
dict_arq2 = {}
index = 405426
maqaf = False

# Preenchendo o dicionário com valores de node_words_SP
for coluna1, coluna2 in linhas_arquivo2:
    coluna2 = coluna2.translate(translation_table)
    
    # Separar palavras compostas e armazenar no dicionário
    if '_' in coluna2:
        coluna2, coluna2b = map(str, coluna2.split('_'))
        maqaf = True
    else:
        maqaf = False
    dict_arq2[index] = [coluna1, coluna2, maqaf]
    index = index+1

index = 405426
arquivo1 = {}

for linha in linhas_arquivo1:
    arquivo1[index] = [str(index), linha[4].translate(translation_table), linha[5]]
    index = index+1

index = 405426
newfile = {}
j=0

for i in range(index, index+len(dict_arq2)):
    if dict_arq2[i][2] == True:
        j=j+1
        newfile[i] = [str(i), dict_arq2[i][1], arquivo1[i+j][2]]
    else:
        newfile[i] = [i, dict_arq2[i][1], arquivo1[i+j][2]]

index = 405426
old_value = 'X'
old_node = index

caminho_arquivo3 = '../new_data/phrase_atom_up3'
f3 = open(caminho_arquivo3, 'w')

for i in range(index, index+len(newfile)):
    current_value = newfile[i][2]
    current_node = newfile[i][0]
    
    if current_value == 'X' and old_value == 'X':
        pass
    elif current_value == 'X' and old_value == 'Y':
        old_value = current_value
        old_node = current_node
    elif current_value == 'Y' and old_value == 'X':
        print(f"{old_node}-{current_node}", file=f3)
        old_node = current_node
        old_value = current_value
    elif current_value == 'Y' and old_value == 'Y':
        print(f"{current_node}", file=f3)
        old_node = current_node

f3.close()

# %%
def load_file_lines(file_path):
    """Load lines from a file and split by tab."""
    with open(file_path, 'r') as f:
        return [line.strip().split('\t') for line in f]

input_file1 = load_file_lines('../new_data/output_morph_phrase_SP_test_up3')
input_file2 = load_file_lines('../new_data/comparison_output_morph_phrase_SP_output_phrase_space_pdp_SP_MT_word')
output_file = '../new_data/comparison_output_morph_phrase_SP_output_phrase_space_pdp_SP_MT_word_up3'
output = open(output_file, 'w')

num=0

input = {}

for i in input_file1:
    for j in i[3]:
        input[num] = j
        num = num + 1

print(input_file2[0][0],input_file2[0][1],input_file2[0][2],input_file2[0][3],input_file2[0][4],
      input_file2[0][5],input_file2[0][6],input_file2[0][7],input_file2[0][8],input_file2[0][9],'result-w+morph-up', sep='\t', file=output)

num=0
for i in input_file2[1:]:
    if len(i) == 10:
        print(i[0],i[1],i[2],i[3],i[4],i[5],
              i[6],i[7],i[8],i[9],input[num], sep='\t', file=output)
    else:
        print(i[0],i[1],i[2],i[3],i[4],i[5],
              i[6],i[7],'','',input[num], sep='\t', file=output)
    num = num + 1

output.close()

# %% [markdown]
# ## February 6 - second attempt of manual correction
# 
# This is the secong update in the dataset (considering verses with the same length in MT and SP but with slight difference of the verse)

# %%
UPDATE = '4'

# %%
inputfilePath = "../new_data/output_morph_phrase_SP"
resultfile = f"../new_data/output_morph_phrase_SP_test_up{UPDATE}"
outputfilePath = f"../new_data/output_morph_phrase_SP_up{UPDATE}"

verse_old = None
result_old = ""
book_old = None
chapter_old = None

output = open(outputfilePath, 'w')

linhas_arquivo = []
with open(resultfile, 'r') as f:
    for line in f:
        line = line.strip().split('\t')
        linhas_arquivo.append(line)

file_dict = {}
for line in linhas_arquivo:
    key = (line[0], line[1], line[2])
    value = line[3]
    file_dict[key] = value

with open(inputfilePath, 'r') as input:
    
    inputlines = input.readlines()
    for i in range (0, len(inputlines)):
        word = inputlines[i].strip().split()
        data = word[:5]
        key = tuple(word[1:4])
        result = file_dict[key]
        verse = word[3]
        if verse != verse_old:
            j=0
            verse_old = verse
        else:
            j=j+1
        print(data[0],data[1],data[2],data[3],data[4],result[j],sep='\t',file=output)

output.close()

# %%
# Leitura dos arquivos
with open(f'../new_data/output_morph_phrase_SP_up{UPDATE}', 'r') as f1, open('../new_data/node_words_SP', 'r') as f2:
    linhas_arquivo1 = [linha.strip().split('\t') for linha in f1.readlines()]
    linhas_arquivo2 = [linha.strip().split('\t') for linha in f2.readlines()]

# Caracteres a serem removidos
morphsign = "~!>+~[]/<>"
translation_table = str.maketrans('', '', morphsign)

# Dicionário para correspondência entre a coluna 5 do arquivo 1 e a coluna 2 do arquivo 2
dict_arq2 = {}
index = 405426
maqaf = False

# Preenchendo o dicionário com valores de node_words_SP
for coluna1, coluna2 in linhas_arquivo2:
    coluna2 = coluna2.translate(translation_table)
    
    # Separar palavras compostas e armazenar no dicionário
    if '_' in coluna2:
        coluna2, coluna2b = map(str, coluna2.split('_'))
        maqaf = True
    else:
        maqaf = False
    dict_arq2[index] = [coluna1, coluna2, maqaf]
    index = index+1

index = 405426
arquivo1 = {}

for linha in linhas_arquivo1:
    arquivo1[index] = [str(index), linha[4].translate(translation_table), linha[5]]
    index = index+1

index = 405426
newfile = {}
j=0

for i in range(index, index+len(dict_arq2)):
    if dict_arq2[i][2] == True:
        j=j+1
        newfile[i] = [str(i), dict_arq2[i][1], arquivo1[i+j][2]]
    else:
        newfile[i] = [i, dict_arq2[i][1], arquivo1[i+j][2]]

index = 405426
old_value = 'X'
old_node = index

caminho_arquivo3 = f'../new_data/phrase_atom_up{UPDATE}'
f3 = open(caminho_arquivo3, 'w')

for i in range(index, index+len(newfile)):
    current_value = newfile[i][2]
    current_node = newfile[i][0]
    
    if current_value == 'X' and old_value == 'X':
        pass
    elif current_value == 'X' and old_value == 'Y':
        old_value = current_value
        old_node = current_node
    elif current_value == 'Y' and old_value == 'X':
        print(f"{old_node}-{current_node}", file=f3)
        old_node = current_node
        old_value = current_value
    elif current_value == 'Y' and old_value == 'Y':
        print(f"{current_node}", file=f3)
        old_node = current_node

f3.close()

# %% [markdown]
# ## April 24 - third attempt of manual correction
# 
# This is the first update in the dataset addressing verses with different number of words of a verse in MT and SP

# %%
!diff -y --suppress-common-lines ../new_data/output_morph_phrase_SP_test_up5 ../new_data/output_morph_phrase_SP_test_up4 | wc -l

# %% [markdown]
# There were 741 updated verses out of 1413 verses to be updated (~ 52.4\%) out of a total of 1458 verses; 45 verses were separated to be updated by the students.

# %%
UPDATE = '5'

# %%
inputfilePath = "../new_data/output_morph_phrase_SP"
resultfile = f"../new_data/output_morph_phrase_SP_test_up{UPDATE}"
outputfilePath = f"../new_data/output_morph_phrase_SP_up{UPDATE}"

verse_old = None
result_old = ""
book_old = None
chapter_old = None

output = open(outputfilePath, 'w')

linhas_arquivo = []
with open(resultfile, 'r') as f:
    for line in f:
        line = line.strip().split('\t')
        linhas_arquivo.append(line)

file_dict = {}
for line in linhas_arquivo:
    key = (line[0], line[1], line[2])
    value = line[3]
    file_dict[key] = value

with open(inputfilePath, 'r') as input:
    
    inputlines = input.readlines()
    for i in range (0, len(inputlines)):
        word = inputlines[i].strip().split()
        data = word[:5]
        key = tuple(word[1:4])
        result = file_dict[key]
        verse = word[3]
        if verse != verse_old:
            j=0
            verse_old = verse
        else:
            j=j+1
        print(data[0],data[1],data[2],data[3],data[4],result[j],sep='\t',file=output)

output.close()

# %%
# Leitura dos arquivos
with open(f'../new_data/output_morph_phrase_SP_up{UPDATE}', 'r') as f1, open('../new_data/node_words_SP', 'r') as f2:
    linhas_arquivo1 = [linha.strip().split('\t') for linha in f1.readlines()]
    linhas_arquivo2 = [linha.strip().split('\t') for linha in f2.readlines()]

# Caracteres a serem removidos
morphsign = "~!>+~[]/<>"
translation_table = str.maketrans('', '', morphsign)

# Dicionário para correspondência entre a coluna 5 do arquivo 1 e a coluna 2 do arquivo 2
dict_arq2 = {}
index = 405426
maqaf = False

# Preenchendo o dicionário com valores de node_words_SP
for coluna1, coluna2 in linhas_arquivo2:
    coluna2 = coluna2.translate(translation_table)
    
    # Separar palavras compostas e armazenar no dicionário
    if '_' in coluna2:
        coluna2, coluna2b = map(str, coluna2.split('_'))
        maqaf = True
    else:
        maqaf = False
    dict_arq2[index] = [coluna1, coluna2, maqaf]
    index = index+1

index = 405426
arquivo1 = {}

for linha in linhas_arquivo1:
    arquivo1[index] = [str(index), linha[4].translate(translation_table), linha[5]]
    index = index+1

index = 405426
newfile = {}
j=0

for i in range(index, index+len(dict_arq2)):
    if dict_arq2[i][2] == True:
        j=j+1
        newfile[i] = [str(i), dict_arq2[i][1], arquivo1[i+j][2]]
    else:
        newfile[i] = [i, dict_arq2[i][1], arquivo1[i+j][2]]

index = 405426
old_value = 'X'
old_node = index

caminho_arquivo3 = f'../new_data/phrase_atom_up{UPDATE}'
f3 = open(caminho_arquivo3, 'w')

for i in range(index, index+len(newfile)):
    current_value = newfile[i][2]
    current_node = newfile[i][0]
    
    if current_value == 'X' and old_value == 'X':
        pass
    elif current_value == 'X' and old_value == 'Y':
        old_value = current_value
        old_node = current_node
    elif current_value == 'Y' and old_value == 'X':
        print(f"{old_node}-{current_node}", file=f3)
        old_node = current_node
        old_value = current_value
    elif current_value == 'Y' and old_value == 'Y':
        print(f"{current_node}", file=f3)
        old_node = current_node

f3.close()

# %% [markdown]
# ## April 27, 28, May 1, 2 - preparing students corrections

# %%
import re

# %%
Sergei = '''
Numbers	12	16	[W] [>XR] [NS<W] [H <M] [M XYRWT] [W] [JXNW] [B MDBR PR>N] [W] [J>MR] [MCH] [L BNJ JFR>L] [B>TM] [<D HR H >MRJ] [>CR] [JHWH] [>LHJNW] [NTN] [LNW] [R>H] [NTN] [JHWH] [>LHJK] [L PNJK] [>T H >RY] [<LH] [RC] [K >CR] [DBR] [JHWH] [>LHJ >BTJK] [LK] [>L] [TJR>] [W] [>L] [TXT] [W] [JQRBW] [>L] [MCH] [W] [J>MRW] [NCLXH] [>NCJM] [L PNJNW] [W] [JXPDW] [LNW] [>T H >RY] [W] [JCJBW] [>TNW] [DBR] [>T H DRK] [>CR] [N<LH] [BH] [W] [>T H <RJM] [>CR] [NBW>] [<LJHN] [W] [JJVB] [H DBR] [B <JNJ MCH]
Numbers	13	33	[W] [CM] [R>JNW] [>T H NPLJM] [BNJ <NQ] [MN H NPLJM] [W] [NHJH] [B <JNJNW] [K XGBJM] [W] [KN] [HJJNW] [B <JNJHM] [W] [JRGNW] [BNJ JFR>L] [B >HLJHM] [W] [J>MRW] [B FN>T JHWH] [>TNW] [HWYJ>NW] [M >RY MYRJM] [L TT] [>TNW] [B JD H >MRJ] [L HCMJDNW] [>NH] [>NXNW] [<LJM] [W] [>XJNW] [HMJSW] [>T LBBNW] [L >MR] [<M] [GDWL W RB] [MMNW] [<RJM] [GDLWT W BYRWT] [B CMJM] [W] [GM BNJ <NQJM] [R>JNW] [CM] [W] [J>MR] [MCH] [L BNJ JFR>L] [L>] [T<RYWN] [W] [L>] [TJR>WN] [MHM] [JHWH] [>LHJKM] [H] [HLK] [L PNJKM] [HW>] [JLXM] [LKM] [K KL] [>CR] [<FH] [>TKM] [B MYRJM] [L <JNJKM] [W] [B MDBR] [>CR] [R>JT] [>CR] [NF>K] [JHWH] [>LHJK] [K >CR] [JF>] [>JC] [>T BNW] [B KL H DRK] [>CR] [HLKTM] [<D B>KM] [<D H MQWM H ZH] [W] [B DBR H ZH] [>JNKM] [M>MNJM] [B JHWH] [>LHJKM] [H] [HLK] [L PNJKM] [B DRK] [L TWR] [LKM] [MQWM] [L HXNWTKM] [B >C] [LJLH] [L HR>WTKM] [B DRK] [>CR] [TLKW] [BH] [W] [B <NN] [JWMM]
Numbers	14	40	[W] [JCKMW] [B BQR] [W] [J<LW] [>L R>C H HR] [L >MR] [HNNW] [W] [<LJNW] [>L H MQWM] [>CR] [>MR] [JHWH] [KJ] [XV>NW] [W] [J>MR] [JHWH] [>L MCH] [>MR] [LHM] [L>] [T<LW] [W] [L>] [TLXMW] [KJ] [>JNNJ] [B QRBKM] [W] [L>] [TNGPW] [L PNJ >JBJKM]
Numbers	14	45	[W] [JRD] [H <MLQJ W H KN<NJ] [H] [JCB] [B HR H HW>] [L QR>TM] [W] [JRDPW] [>TM] [K >CR] [T<FJNH] [H DBRJM] [W] [JKWM] [W] [JKTWM] [<D XRMH] [W] [JCBW] [>L H MXNH]
Numbers	20	13	[HM] [MJ MRJBH] [>CR] [RBW] [BNJ JFR>L] [>T JHWH] [W] [JQDC] [BM] [W] [J>MR] [MCH] [>DNJ] [JHWH] [>TH] [HXLT] [L HR>WT] [>T <BDK] [>T GDLK W >T JDK H XZQH] [>CR] [MJ] [>L] [B CMJM W B >RY] [>CR] [J<FH] [K M<FJK W K GBWRTJK] [><BRH] [N>] [W] [>R>H] [>T H >RY H VWBH] [>CR] [B <BR H JRDN] [H HR H VWB H ZH W H LBNWN] [W] [J>MR] [JHWH] [>L MCH] [RB] [LK] [>L] [TWSP] [DBR] [>LJ] [<WD] [B DBR H ZH] [<LH] [>L R>C H PSGH] [W] [F>] [<JNJK] [JMH W YPWNH W TJMNH W MZRXH] [W] [R>H] [B <JNJK] [KJ] [L>] [T<BR] [>T H JRDN H ZH] [W] [YWJ] [>T JHWC<] [BN NWN] [W] [XZQHW] [W] [>MYHW] [KJ] [HW>] [J<BR] [L PNJ H <M H ZH] [W] [HW>] [JNXL] [>TM] [>T H >RY] [>CR] [TR>H] [W] [JDBR] [JHWH] [>L MCH] [L >MR] [RB] [LKM] [SWB] [>T H HR H ZH] [PNW] [LKM] [YPWNH] [W] [>T H <M] [YWJ] [L >MR] [>TM] [<BRJM] [B GBWL >XJKM] [BNJ <FW] [H] [JCBJM] [B F<JR] [W] [JJR>W] [MKM] [W] [NCMRTM] [M>D] [>L] [TTGRW] [BM] [KJ] [L>] [>TN] [LKM] [M >RYM] [JRCH] [<D M DRK KP RGL] [KJ] [JRCH] [L <FW] [NTTJ] [>T HR F<JR] [>KL] [TCBRW] [M >TM] [B KSP] [W] [>KLTM] [W] [GM MJM] [TKJRW] [M >TM] [B KSP] [W] [CTJTM]
Numbers	21	11	[W] [JS<W] [M >BWT] [W] [JXNW] [B <JJ H<BRJM] [B MDBR] [>CR] [<L PNJ MW>B] [MZRX H CMC] [W] [J>MR] [JHWH] [>L MCH] [>L] [TYWR] [>T MW>B] [W] [>L] [TTGR] [BM] [KJ] [L>] [>TN] [LK] [M >RYW] [JRCH] [KJ] [L BNJ LWV] [NTTJ] [>T <R] [JRCH]
Numbers	21	12	[M CM] [NS<W] [W] [JXNW] [B NXL ZRD] [W] [JDBR] [JHWH] [>L MCH] [L >MR] [>TH] [<BR] [H JWM] [>T GBWL MW>B] [>T <R] [W] [QRBT] [MWL BNJ <MWN] [>L] [TYWRM] [W] [>L] [TTGR] [BM] [KJ] [L>] [>TN] [M >RY BNJ <MWN] [LK] [JRCH] [KJ] [L BNJ LWV] [NTTJH] [JRCH]
Numbers	21	20	[W] [M BMWT] [H GJ>] [>CR] [B FDH MW>B] [R>C H PSGH] [H] [NCQP] [<L PNJ H JCMWN] [W] [J>MR] [JHWH] [>L MCH] [QWMW] [S<W] [W] [<BRW] [>T NXL >RNN] [R>H] [NTTJ] [B JDK] [>T SJXWN] [MLK XCBWN] [H >MRJ] [W] [>T >RYW] [HXL] [RC] [W] [HTGR] [BW] [MLXMH] [H JWM H ZH] [HXL] [TT] [PXDK W JR>TK] [<L PNJ H <MJM] [TXT KL H CMJM] [>CR] [JCM<W] [>T CM<K] [W] [RGZW] [W] [XLW] [M PNJK]
Numbers	21	22	[><BRH] [B >RYK] [B DRK H MLK] [>LK] [L>] [>SWR] [JMJN W FM>L] [L>] [>VH] [B FDH W B KRM] [>KL] [B KSP] [TCBJRNJ] [W] [>KLTJ] [W] [MJM] [B KSP] [TTN] [LJ] [W] [CTJTJ] [RQ] [><BRH] [B RGLJ] [K >CR] [<FW] [LJ] [BNJ <FW] [H] [JCBJM] [B F<JR] [W] [H MW>BJM] [H] [JCBJM] [B <R]
Numbers	21	23	[W] [L>] [NTN] [SJXWN] [>T JFR>L] [<BR] [B GBWLW] [W] [J>MR] [JHWH] [>L MCH] [R>H] [HXLTJ] [TT] [L PNJK] [>T SJXWN W >T >RYW] [HXL] [RC] [L RCT] [>T >RYW] [W] [J>SP] [SJXWN] [>T KL <MW] [W] [JY>] [L QR>T] [JFR>L] [H MDBRH] [W] [JB>] [JXYH] [W] [JLXM] [B JFR>L]
Numbers	27	23	[W] [JSMK] [>T JDW] [<LJW] [W] [JYWHW] [K >CR] [DBR] [JHWH] [B JD MCH] [W] [J>MR] [>LJW] [<JNJK] [H] [R>WT] [>T >CR] [<FH] [JHWH] [L CNJ H MLKJM H >LH] [KN] [J<FH] [JHWH] [L KL H MMLKWT] [>CR] [>TH] [<BR] [CMH] [L>] [TJR>M] [KJ] [JHWH] [>LHJKM] [HW>] [H] [NLXM] [LKM]
Numbers	31	20	[W] [KL BGD W KL KLJ <WR W KL M<FH <ZJM W KL KLJ <Y] [TTXV>W] [W] [J>MR] [MCH] [>L >L<ZR] [H KHN] [>MR] [>L >NCJ H YB>] [H] [B>JM] [L MLXMH] [Z>T] [XQT H TWRH] [>CR] [YWH] [JHWH] [>K] [>T H ZHB W >T H KSP W >T H NXCT W >T H BRZL W >T H BDJL W >T H <WPRT] [KL DBR] [>CR] [JBW>] [B >C] [T<BJRW] [B >C] [W] [VHR] [>K B MJ NDH] [JTXV>] [W] [KL >CR] [L>] [JBW>] [B >C] [T<BJRW] [B MJM] [W] [KBSTM] [BGDJKM] [B JWM H CBJ<J] [W] [VHRTM] [W] [>XR] [TB>W] [>L H MXNH]
Deuteronomy	2	7	[KJ] [JHWH] [>LHJK] [BRKK] [B KL M<FH JDJK] [JD<] [LKTK] [>T H MDBR H GDWL H ZH] [ZH] [>RB<JM CNH] [JHWH] [>LHJK] [<MK] [L>] [XSRT] [DBR] [W] [>CLXH] [ML>KJM] [>L MLK >DWM] [L >MR] [><BRH] [B >RYK] [L>] [>VH] [B FDH W B KRM] [W] [L>] [NCTH] [MJ BWR] [DRK H MLK] [NLK] [L>] [NSWR] [JMJN W FM>L] [<D >CR] [N<BR] [GBWLK] [W] [J>MR] [L>] [T<BR] [BJ] [PN] [B XRB] [>Y>] [L QR>TK]
Deuteronomy	5	21	[L>] [TXMD] [BJT R<K] [W] [L>] [TXMD] [>CT R<K] [FDHW] [<BDW W >MTW CWRW W XMRW W KL] [>CR] [L R<K] [W] [HJH] [KJ] [JBJ>K] [JHWH] [>LHJK] [>L >RY H KN<NJ] [>CR] [>TH] [B>] [CMH] [L RCTH] [W] [HQMT] [LK] [>BNJM GDLWT] [W] [FDT] [>TM] [B FJD] [W] [KTBT] [<L H >BNJM] [>T KL DBRJ H TWRH H Z>T] [W] [HJH] [B <BRKM] [>T H JRDN] [TQJMW] [>T H >BNJM H >LH] [>CR] [>NKJ] [MYWH] [>TKM] [H JWM] [B HRGRJZJM] [W] [BNJT] [CM] [MZBX] [L JHWH] [>LHJK] [MZBX >BNJM] [L>] [TNJP] [<LJHM] [BRZL] [>BNJM CLMWT] [TBNH] [>T MZBX JHWH] [>LHJK] [W] [H<LJT] [<LJW] [<LWT] [L JHWH] [>LHJK] [W] [ZBXT] [CLMJM] [W] [>KLT] [CM] [W] [FMXT] [L PNJ JHWH] [>LHJK] [H HR H HW>] [B <BR H JRDN] [>XRJ DRK MBW> H CMC] [B >RY H KN<NJ] [H] [JCB] [B <RBH] [MWL H GLGL] [>YL >LWN MWR>] [MWL CKM]
Deuteronomy	10	7	[M CM] [NS<W] [W] [JXNW] [H GDGDH] [M CM] [NS<W] [W] [JXNW] [B JVBTH] [>RY NXLJ MJM] [M CM] [NS<W] [W] [JXNW] [B <BRNH] [M CM] [NS<W] [W] [JXNW] [B <YJWN GBR] [M CM] [NS<W] [W] [JXNW] [B MDBR YN] [HJ>] [QDC] [M CM] [NS<W] [W] [JXNW] [B HR H HR] [W] [JMT] [CM] [>HRN] [W] [JQBR] [CM] [W] [JKHN] [>L<ZR] [BNW] [TXTJW]
'''

# %%
word_num = 0
for i in Sergei.split('\n'):
    if len(i)>0:
        result = ''
        start_phrase_atom = False
        book = i.split('\t')[0]
        chapter = i.split('\t')[1]
        verse = i.split('\t')[2]
        verse_text = i.split('\t')[-1]
        phrase_atom = verse_text.split(' ')
        for word in phrase_atom:
            word_num = word_num+1
            if word[0] == '[':
                start_phrase_atom = True
            if word[-1] == ']' and start_phrase_atom == True:
                result = result + 'Y'
                start_phrase_atom = False
            else:
                result = result + 'X'
        
        print(book, chapter, verse, result, sep='\t')
print(word_num)

# %%
print((1-27/word_num)*100)

# %%
Esther = '''
Genesis	11	13	[W] [JXJ] [>RPKCD] [>XRJ HWLJDW] [>T CLX] [CLC CNJM W CLC M>WT CNH] [W] [JWLJD] [BNJM W BNWT] [W] [JHJW] [KL JMJ >RPKCD] [CMNH W CLCJM CNH W >RB< M>WT CNH] [W] [JMT]
Genesis	11	15	[W] [JXJ] [CLX] [>XRJ HWLJDW] [>T <BR] [CLC CNJM W CLC M>WT CNH] [W] [JWLJD] [BNJM W BNWT] [W] [JHJW] [KL JMJ CLX] [CLC W CLCJM CNH W >RB< M>WT CNH] [W] [JMT]
Genesis	11	19	[W] [JXJ] [PLG] [>XRJ HWLJDW] [>T R<W] [TC< CNJM W M>T CNH] [W] [JWLJD] [BNJM W BNWT] [W] [JHJW] [KL JMJ PLG] [TC< W CLCJM W M>TJM CNH] [W] [JMT]
Genesis	11	21	[W] [JXJ] [R<W] [>XRJ HWLJDW] [>T FRWG] [CB< CNJM W M>T CNH] [W] [JWLJD] [BNJM W BNWT] [W] [JHJW] [KL JMJ R<W] [TC< W CLCJM W M>TJM CNH] [W] [JMT]
Genesis	11	25	[W] [JXJ] [NXWR] [>XRJ HWLJDW] [>T TRX] [TC< CNJM W CCJM CNH] [W] [JWLJD] [BNJM W BNWT] [W] [JHJW] [KL JMJ NXWR] [CMNH W >RB<JM CNH W M>T CNH] [W] [JMT]
Genesis	30	36	[W] [JFM] [DRK CLCT JMJM] [BJNM W BJN J<QB] [W] [J<QB] [R<H] [>T Y>N LBN] [H] [NWTRT] [W] [J>MR] [ML>K >LHJM] [>L J<QB] [B XLWM] [W] [J>MR] [J<QB] [W] [J>MR] [HNNJ] [W] [J>MR] [F>] [N>] [<JNJK] [W] [R>H] [>T KL H <TWDJM] [H] [<LJM] [<L H Y>N] [<QWDJM NQWDJM W BRWDJM] [KJ] [R>JTJ] [>T KL] [>CR] [LBN] [<FH] [LK] [>NKJ] [H >L] [BJT >L] [>CR] [MCXT] [CM] [MYBH] [W] [>CR] [NDRT] [LJ] [CM] [NDR] [W] [<TH] [QWM] [Y>] [MN H >RY H Z>T] [W] [CWB] [>L >RY >BJK] [W] [>VJB] [<MK]
Genesis	42	16	[CLXW] [MKM] [>XD] [W] [JQX] [>T >XJKM] [W] [>TM] [H>SRW] [W] [JBXNW] [DBRJKM] [H] [>MT] [>TKM] [W] [>M] [L>] [XJ] [PR<H] [KJ] [MRGLJM] [>TM] [W] [J>MRW] [L>] [JWKL] [H N<R] [L <ZB] [>T >BJW] [W] [<ZB] [>T >BJW] [W] [MT]
Exodus	6	9	[W] [JDBR] [MCH] [KN] [L BNJ JFR>L] [W] [L>] [CM<W] [>L MCH] [M QYR RWX W M <BDH QCH] [W] [J>MRW] [>L MCH] [XDL] [N>] [MMNW] [W] [N<BDH] [>T MYRJM] [KJ] [VWB] [LNW] [<BD] [>T MYRJM] [M MWTNW] [B MDBR]
Exodus	7	18	[W] [H DGH] [>CR] [B J>R] [TMWT] [W] [B >C] [H J>R] [W] [NL>W] [MYRJM] [L CTWT] [MJM] [MN H J>R] [W] [JLK] [MCH W >HRN] [>L PR<H] [W] [J>MRW] [>LJW] [JHWH] [>LHJ H <BRJM] [CLXNW] [>LJK] [L >MR] [CLX] [>T <MJ] [W] [J<BDNJ] [B MDBR] [W] [HNH] [L>] [CM<T] [<D KH] [KH] [>MR] [JHWH] [B Z>T] [TD<] [KJ] [>NJ] [JHWH] [HNH] [>NKJ] [MKH] [B MVH] [>CR] [B JDJ] [<L H MJM] [>CR] [B J>R] [W] [NHPKW] [L DM] [W] [H DGH] [>CR] [B J>R] [TMWT] [W] [B >C] [H J>R] [W] [NL>W] [MYRJM] [L CTWT] [MJM] [MN H J>R]
Exodus	7	29	[W] [BK W B <MK W B KL <BDJK] [J<LW] [H YPRD<JM] [W] [JB>] [MCH W >HRN] [>L PR<H] [W] [JDBRW] [>LJW] [KH] [>MR] [JHWH] [CLX] [>T <MJ] [W] [J<BDNJ] [W] [>M] [M>N] [>TH] [L CLX] [HNH] [>NKJ] [NGP] [>T KL GBWLK] [B YPRD<JM] [W] [CRY] [H J>R] [YPRD<JM] [W] [<LW] [W] [B>W] [B BTJK W B XDRJ MCKBJK] [W] [<L MVTJK] [W] [B BTJ <BDJK W B <MK W B TNWRJK W B MC>RTJK] [W] [BK W B <MK W B KL <BDJK] [J<LW] [H YPRD<JM]
Exodus	8	1	[W] [J>MR] [JHWH] [>L MCH] [>MR] [>L >HRN] [NVH] [>T JDK] [B MVK] [<L H NHRWT W <L H J>RJM W <L H >GMJM] [W] [H<L] [>T H YPRD<JM] [<L >RY MYRJM] [W] [J>MR] [MCH] [>L >HRN] [NVH] [>T JDK] [B MVK] [W] [T<L] [H YPRD<] [<L >RY MYRJM]
Exodus	8	19	[W] [FMTJ] [PDWT]] [BJN <MJ W BJN <MK] [L MXR] [JHJH] [H >WT H ZH] [W] [JB>] [MCH W >HRN] [>L PR<H] [W] [J>MRW] [>LJW] [KH] [>MR] [JHWH] [CLX] [>T <MJ] [W] [J<BDNJ] [KJ] [>M] [>JNK] [MCLX] [>T <MJ] [HNNJ] [MCLX] [BK W B <BDJK W B <MK W B BTJK] [>T H <RB] [W] [ML>W] [BTJ MYRJM] [>T H <RB] [W] [GM H >DMH] [>CR] [HM] [<LJH] [W] [HPLJTJ] [B JWM H HW>] [>T >RY GCN] [>CR] [<MJ] [<MD] [<LJH] [L BLTJ HJWT] [CM] [<RB] [LM<N] [TD<] [KJ] [>NJ] [JHWH] [B QRB H >RY] [W] [FMTJ] [PDWT] [BJN <MJ W BJN <MK] [L MXR] [JHJH] [H >WT H ZH]
Exodus	9	5	[W] [JFM] [JHWH] [MW<D] [L >MR] [MXR] [J<FH] [JHWH] [>T H DBR H ZH] [B >RY] [W] [JB>] [MCH W >HRN] [>L PR<H] [W] [J>MRW] [>LJW] [KH] [>MR] [JHWH] [>LHJ H <BRJM] [CLX] [>T <MJ] [W] [J<BDNJ] [KJ] [>M] [M>N] [>TH] [L CLX] [W] [<WDK] [MXZQ] [BM] [HNH] [JD JHWH] [HJH] [B MQNJK] [>CR] [B FDH] [B SWSJM W B XMRJM W B GMLJM B BQR W B Y>N] [DBR KBD M>D] [W] [H PL>] [JHWH] [BJN MQNH JFR>L W BJN MQNH MYRJM] [W] [L>] [JMWT] [M KL] [L BNJ JFR>L] [DBR] [MXR] [J<FH] [JHWH] [>T H DBR H ZH] [B >RY]
Exodus	9	19	[W] [<TH] [CLX] [H<Z] [>T MQNJK W >T KL] [>CR] [LK] [B FDH] [KL H >DM W H BHMH] [>CR] [JMY>] [B FDH] [W] [L>] [J>SP] [H BJTH] [W] [JRD] [<LJHM] [H BRD] [W] [MTW] [W] [JB>] [MCH W >HRN] [>L PR<H] [W] [J>MRW] [>LJW] [KH] [>MR] [JHWH] [>LHJ H <BRJM] [CLX] [>T <MJ] [W] [J<BDNJ] [KJ] [B P<M H Z>T] [>NJ] [CLX] [>T KL MGPTJ] [<L LBK] [W] [B <BDJK W B <MK] [B <BWR] [TD<] [KJ] [>JN] [KMWNJ] [B KL H >RY] [KJ] [<TH] [CLXTJ] [>T JDJ] [W] [>KH] [>TK W >T <MK] [B DBR] [W] [TKXD] [MN H >RY] [W] [>WLM] [B <BWR Z>T] [H<MDTJK] [B <BWR HR>TJK] [>T KXJ] [W] [LM<N SPR] [CMJ] [B KL H >RY] [<WDK] [MSTWLL] [B <MJ] [L BLTJ CLXM] [HNNJ] [MMVJR] [K <T MXR] [BRD KBD M>D] [>CR] [L>] [HJH] [KMHW] [B MYRJM] [L M JWM H JSDH] [W] [<D <TH] [W] [<TH] [CLX] [H <Z] [>T MQNJK W >T KL] [>CR] [LK] [B FDH] [KL H >DM W H BHMH] [>CR] [JMY>] [B FDH] [W] [L>] [J>SP] [H BJTH] [W] [JRD] [<LJHM] [H BRD] [W] [MTW]
Exodus	10	2	[W] [LM<N] [TSPR] [B >ZNJ BNK W BN BNK] [>T >CR] [HT<LLTJ] [B MYRJM] [W] [>T >TWTJ] [>CR] [FMTJ] [BM] [W] [JD<TM] [KJ] [>NJ] [JHWH] [>LHJKM] [W] [>MRT] [>L PR<H] [KH] [>MR] [JHWH] [>LHJ H <BRJM] [<D MTJ] [M>NT] [L <NWT] [M PNJ] [CLX] [>T <MJ] [W] [J<BDNJ] [KJ] [>M] [M>N] [>TH] [L CLX] [>T <MJ] [HNNJ] [MBJ>] [MXR] [>RBH] [B GBWLK] [W] [KSH] [>T <JN H >RY] [W] [L>] [JKL] [L R>WT] [>T H >RY] [W] [>KL] [>T JTR H PLVH] [H] [NC>RT] [LKM] [MN H BRD] [W] [>KL] [>T KL <FB H >RY W >T KL PRJ H <Y] [H] [YMX] [LKM] [MN H FDH] [W] [ML>W] [BTJK W BTJ KL <BDJK W BTJ KL MYRJM] [>CR] [L>] [R>W] [>BTJK W >BWT >BTJK] [M JWM] [HJWTM] [<L H >DMH] [<D H JWM H ZH]
'''

# %%
word_num = 0
for i in Esther.split('\n'):
    if len(i)>0:
        result = ''
        start_phrase_atom = False
        book = i.split('\t')[0]
        chapter = i.split('\t')[1]
        verse = i.split('\t')[2]
        verse_text = i.split('\t')[-1]
        phrase_atom = verse_text.split(' ')
        for word in phrase_atom:
            word_num = word_num+1
            if word[0] == '[':
                start_phrase_atom = True
            if word[-1] == ']' and start_phrase_atom == True:
                result = result + 'Y'
                start_phrase_atom = False
            else:
                result = result + 'X'
        
        print(book, chapter, verse, result, sep='\t')
print(word_num)

# %%
print((1-51/word_num)*100)

# %%
UPDATE = '6'

# %%
inputfilePath = "../new_data/output_morph_phrase_SP"
resultfile = f"../new_data/output_morph_phrase_SP_test_up{UPDATE}"
outputfilePath = f"../new_data/output_morph_phrase_SP_up{UPDATE}"

verse_old = None
result_old = ""
book_old = None
chapter_old = None

output = open(outputfilePath, 'w')

linhas_arquivo = []
with open(resultfile, 'r') as f:
    for line in f:
        line = line.strip().split('\t')
        linhas_arquivo.append(line)

file_dict = {}
for line in linhas_arquivo:
    key = (line[0], line[1], line[2])
    value = line[3]
    file_dict[key] = value

with open(inputfilePath, 'r') as input:
    
    inputlines = input.readlines()
    for i in range (0, len(inputlines)):
        word = inputlines[i].strip().split()
        data = word[:5]
        key = tuple(word[1:4])
        result = file_dict[key]
        verse = word[3]
        if verse != verse_old:
            j=0
            verse_old = verse
        else:
            j=j+1
        print(data[0],data[1],data[2],data[3],data[4],result[j],sep='\t',file=output)

output.close()

# %%
# Leitura dos arquivos
with open(f'../new_data/output_morph_phrase_SP_up{UPDATE}', 'r') as f1, open('../new_data/node_words_SP', 'r') as f2:
    linhas_arquivo1 = [linha.strip().split('\t') for linha in f1.readlines()]
    linhas_arquivo2 = [linha.strip().split('\t') for linha in f2.readlines()]

# Caracteres a serem removidos
morphsign = "~!>+~[]/<>"
translation_table = str.maketrans('', '', morphsign)

# Dicionário para correspondência entre a coluna 5 do arquivo 1 e a coluna 2 do arquivo 2
dict_arq2 = {}
index = 405426
maqaf = False

# Preenchendo o dicionário com valores de node_words_SP
for coluna1, coluna2 in linhas_arquivo2:
    coluna2 = coluna2.translate(translation_table)
    
    # Separar palavras compostas e armazenar no dicionário
    if '_' in coluna2:
        coluna2, coluna2b = map(str, coluna2.split('_'))
        maqaf = True
    else:
        maqaf = False
    dict_arq2[index] = [coluna1, coluna2, maqaf]
    index = index+1

index = 405426
arquivo1 = {}

for linha in linhas_arquivo1:
    arquivo1[index] = [str(index), linha[4].translate(translation_table), linha[5]]
    index = index+1

index = 405426
newfile = {}
j=0

for i in range(index, index+len(dict_arq2)):
    if dict_arq2[i][2] == True:
        j=j+1
        newfile[i] = [str(i), dict_arq2[i][1], arquivo1[i+j][2]]
    else:
        newfile[i] = [i, dict_arq2[i][1], arquivo1[i+j][2]]

index = 405426
old_value = 'X'
old_node = index

caminho_arquivo3 = f'../new_data/phrase_atom_up{UPDATE}'
f3 = open(caminho_arquivo3, 'w')

for i in range(index, index+len(newfile)):
    current_value = newfile[i][2]
    current_node = newfile[i][0]
    
    if current_value == 'X' and old_value == 'X':
        pass
    elif current_value == 'X' and old_value == 'Y':
        old_value = current_value
        old_node = current_node
    elif current_value == 'Y' and old_value == 'X':
        print(f"{old_node}-{current_node}", file=f3)
        old_node = current_node
        old_value = current_value
    elif current_value == 'Y' and old_value == 'Y':
        print(f"{current_node}", file=f3)
        old_node = current_node

f3.close()

# %%
Ronaldo = '''
Exodus	11	3	[W] [NTTJ] [>T XN H <M H ZH] [B <JNJ MYRJM] [W] [HC>LWM] [W] [K XYJT H LJLH] [>NJ] [JY>] [B TWK >RY MYRJM] [W] [MT] [KL BKWR] [B >RY MYRJM] [M BKWR PR<H] [H] [JCB] [<L KS>W] [W] [<D BKWR H CPXH] [>CR] [>XR H RXJM] [W] [<D BKWR KL BHMH] [W] [HJTH] [Y<QH GDLH] [B MYRJM] [>CR] [KMWH] [L>] [NHJTH] [W] [KMWH] [L>] [TWSJP] [W] [L KL BNJ JFR>L] [L>] [JXRY] [KLB] [LCNW] [L M >JC W <D BHMH] [LM<N] [TD<] [>CR] [JPL>] [JHWH] [BJN MYRJM W BJN JFR>L] [W] [GM H >JC] [MCH] [GDL M>D] [B >RY MYRJM] [B <JNJ <BDJ PR<H W B <JNJ H <M] [W] [J>MR] [MCH] [>L PR<H] [KH] [>MR] [JHWH] [BNJ] [BKWRJ] [JFR>L] [W] [>MR] [>LJK] [CLX] [>T BNJ] [W] [J<BDNJ] [W] [TM>N] [L CLXW] [HNH] [JHWH] [HRG] [>T BNK] [BKWRK]
Exodus	18	24	[W] [JCM<] [MCH] [L QWL XTNW] [W] [J<F] [KL] [>CR] [>MR] [W] [J>MR] [MCH] [>L H <M] [L>] [>WKL] [>NKJ] [L BDJ] [F>T] [>TKM] [JHWH] [>LHJKM] [HRBH] [>TKM] [W] [HNKM] [H JWM] [K KWKBJ H CMJM] [L RB] [JHWH] [>LHJ >BTJKM] [JSP] [<LJKM] [KKM] [>LP P<MJM] [W] [JBRK] [>TKM] [K >CR] [DBR] [LKM] [>JK] [>F>] [L BDJ] [VRXKM MF>KM W RJBKM] [HBW] [LKM] [>NCJM XKMJM W NBWNJM W JD<JM] [L CBVJKM] [W] [>FJMM] [B R>CJKM] [W] [J<NW] [W] [J>MRW] [VWB] [H DBR] [>CR] [DBRT] [L <FWT] [W] [JQX] [>T R>CJ CBVJHM] [>NCJM XKMJM W JD<JM]
Exodus	18	25	[W] [JTN] [>TM] [R>CJM] [<LJHM] [FRJ >LPJM W FRJ M>WT FRJ XMCJM W FRJ <FRWT W CVRJM] [L CBVJHM] [W] [JYW] [>T CPVJHM] [L >MR] [CM<W] [BJN >XJKM] [W] [CPVTM] [YDQ] [BJN >JC W BJN >XJW W BJN GRW] [L>] [TKJRW] [PNJM] [B MCPV] [K QVN] [K GDWL] [TCM<WN] [L>] [TGWRW] [M PNJ >JC] [KJ] [H MCPV] [L >LHJM] [HW>] [W] [H DBR] [>CR] [JQCH] [MKM] [TQRJBWN] [>LJ] [W] [CM<TJW] [W] [JYW] [>TM] [>T KL H DBRJM] [>CR] [J<FWN]
Exodus	20	17	[L>] [TXMD] [BJT R<K] [W L>] [TXMD] [>CT R<K FDHW <BDW W >MTW CWRW W XMWRW W KL] [>CR] [L R<K] [W] [HJH] [KJ] [JBJ>K] [JHWH] [>LHJK] [>L >RY H KN<NJ] [>CR] [>TH] [B>] [CMH] [L RCTH] [W] [HQMT] [LK] [>BNJM GDLWT] [W] [FDT] [>TM] [B FJD] [W] [KTBT] [<L H >BNJM] [>T KL DBRJ H TWRH H Z>T] [W] [HJH] [B <BRKM] [>T H JRDN] [TQJMW] [>T H >BNJM H >LH] [>CR] [>NKJ] [MYWH] [>TKM] [H JWM] [B HRGRJZJM] [W] [BNJT] [CM] [MZBX] [L JHWH] [>LHJK] [MZBX >BNJM] [L>] [TNJP] [<LJHM] [BRZL] [>BNJM CLMWT] [TBNH] [>T MZBX JHWH] [>LHJK] [W] [H<LJT] [<LJW] [<LWT] [L JHWH] [>LHJK] [W] [ZBXT] [CLMJM] [W] [>KLT] [CM] [W] [FMXT] [L PNJ JHWH] [>LHJK] [H HR H HW>] [B <BR H JRDN] [>XRJ DRK MBW> H CMC] [B >RY H KN<NJ] [H] [JCB] [B <RBH] [MWL H GLGL] [>YL >LWN MWR>] [MWL CKM]
Exodus	20	19	[W] [J>MRW] [>L MCH] [HN] [HR>NW] [JHWH] [>LHJNW] [>T KBWDW W >T GDLW] [W] [>T QWLW] [CM<NW] [M TWK H >C] [H JWM H ZH] [R>JNW] [KJ] [JDBR] [>LHJM] [>T H >DM] [W] [XJ] [W] [<TH] [LMH] [NMWT] [KJ] [T>KLNW] [H >C H GDLH H Z>T] [>M] [JSPJM] [>NXNW] [L CM<] [>T QWL JHWH] [>LHJNW] [<WD] [W] [MTNW] [KJ] [MJ] [KL BFR] [>CR] [CM<] [QWL >LHJM XJJM] [MDBR] [M TWK H >C] [KMWNW] [W] [XJ] [QRB] [>TH] [W] [CM<] [>T KL] [>CR] [J>MR] [JHWH] [>LHJNW] [W] [>TH] [TDBR] [>LJNW] [>T KL] [>CR] [JDBR] [JHWH] [>LHJNW] [>LJK] [W] [CM<NW] [W] [<FJNW] [W] [>L] [JDBR] [<MNW] [H >LHJM] [PN] [NMWT]
Exodus	20	21	[W] [J<MD] [H <M] [M RXQ] [W] [MCH] [NGC] [>L H <RPL] [>CR] [CM] [H >LHJM] [W] [JDBR] [JHWH] [>L MCH] [L >MR] [CM<TJ] [>T QWL DBRJ H <M H ZH] [>CR] [DBRW] [>LJK] [HVJBW] [KL] [>CR] [DBRW] [MJ] [JTN] [W] [HJH] [LBBM] [ZH] [LHM] [L JR>H] [>TJ] [W] [L CMR] [>T MYWTJ] [KL H JMJM] [LM<N] [JJVB] [LHM W L BNJHM] [L <WLM] [NBJ>] [>QJM] [LHM] [M QRB >XJHM] [KMWK] [W] [NTTJ] [DBRJ] [B PJW] [W] [DBR] [>LJHM] [>T KL] [>CR] [>YWNW] [W] [HJH] [H >JC] [>CR] [L>] [JCM<] [>L DBRJW] [>CR] [JDBR] [B CMJ] [>NKJ] [>DRC] [M <MW] [>K H NBJ>] [>CR] [JZJD] [L DBR] [DBR] [B CMJ] [>T >CR] [L>] [YWJTJW] [L DBR] [W] [>CR] [JDBR] [B CM >LHJM >XRJM] [W] [MT] [H NBJ> H HW>] [W] [KJ] [T>MR] [B LBBK] [>JK] [NWD<] [>T H DBR] [>CR] [L>] [DBRW] [JHWH] [>CR] [JDBR] [H NBJ>] [B CM JHWH] [L>] [JHJH] [H DBR] [W] [L>] [JBW>] [HW>] [H DBR] [>CR] [L>] [DBRW] [JHWH] [B ZDWN] [DBRW] [H NBJ>] [L>] [TGWR] [MMNW] [LK] [>MR] [LHM] [CWBW] [LKM] [L >HLJKM] [W] [>TH] [PH] [<MD] [<MDJ] [W] [>DBRH] [>LJK] [>T KL H MYWH H XQJM W H MCPVJM] [>CR] [TLMDM] [W] [<FW] [B >RY] [>CR] [>NKJ] [NTN] [LHM] [L RCTH]
Exodus	22	4	[W] [KJ] [JB<JR] [>JC] [FDH >W KRM] [W] [CLX] [>T B<JRW] [W] [B<R] [B FDH >XR] [CLM] [JCLM] [M FDHW] [K TBW>TH] [W] [>M] [KL H FDH] [JB<H] [MJVB FDHW W MJVB KRMW] [JCLM]
Exodus	23	28	[W] [CLXTJ] [>T H YR<H] [L PNJK] [W] [GRCH] [>T H KN<NJ W >T H >MRJ W >T H XTJ W >T H GRGCJ W >T H PRZJ W >T H XWJ W >T H JBWSJ] [M L PNJK]
Exodus	26	35	[W] [FMT] [>T H CLXN] [M XWY] [L PRKT] [W] [>T H MNWRH] [NKX H CLXN] [<L JRK H MCKN] [TJMNH] [W] [>T H CLXN] [TTN] [<L YL< YPWNH] [W] [<FJT] [MZBX MQVJR QVRT] [<YJ CVJM] [T<FH] [>TW] [>MH] [>RKW] [W] [>MH] [RXBW] [RBW<] [JHJH] [W] [>MTJM] [QWMTW] [MMNW] [QRNTJW] [W] [YPJT] [>TW] [ZHB VHWR] [>T GGW W >T QJRTJW SBJB W >T QRNTJW] [W] [<FJT] [LW] [ZR ZHB] [SBJB] [W] [CTJ VB<WT ZHB] [T<FH] [LW] [M TXT] [L ZRW] [<L CTJ YL<TJW] [T<FH] [<L CNJ YDJW] [W] [HJW] [L BTJM] [L BDJM] [L F>T] [>TW] [BHM] [W] [<FJT] [>T H BDJM] [<YJ CVJM] [W] [YPJT] [>TM] [ZHB] [W] [NTTH] [>TW] [L PNJ H PRKT] [>CR] [<L >RWN H <DWT] [>CR] [>W<D] [LK] [CMH] [W] [HQVJR] [<LJW] [>HRN] [QVRT SMJM] [B BQR B BQR] [B HVJBW] [>T H NRWT] [JQVJRNW] [W] [B H<LWT] [>HRN] [>T H NRWT] [BJN H <RBJM] [JQVJRNH] [QVRT TMJD] [L PNJ JHWH] [L DWRTJKM] [L>] [T<LW] [<LJW] [QVRT ZRH W <LH W MNXH] [W] [NSK] [L>] [TSKW] [<LJW] [W] [KPR] [>HRN] [<L QRNTJW] [>XT B CNH] [M DM XV>T H KPWRJM] [>XT B CNH] [JKPR] [<LJW] [L DWRTJKM] [QDC QDCJM] [HW>] [L JHWH]
Exodus	27	19	[W] [<FJT] [>T KL KLJ H MCKN] [B KL <BDTW] [W] [B KL JTDTJW W KL JTDWT H XYR] [NXCT] [W] [<FJT] [BGDJ TKLT W >RGMN W TWL<T CNJ] [L CRT] [BHM] [B QDC]
Exodus	32	10	[W] [<TH] [HNJXH] [LJ] [W] [JXR] [>PJ] [BM] [W] [>KLM] [W] [><FH] [>TK] [L GWJ GDWL] [W] [B >HRN] [HT>NP] [JHWH] [M>D] [L HCMJDW] [W] [JTPLL] [MCH] [B<D >HRN]
Exodus	39	21	[W] [JRKSW] [>T H XCN] [M VB<TW] [>L VB<T H >PWD] [B PTJL TKLT] [L HJWT] [<L XCB H >PWD] [W] [L>] [JZH] [H XCN] [M <L H >PWD] [K >CR] [YWH] [JHWH] [>T MCH] [W] [J<FW] [>T H >RJM W >T H TMJM] [K >CR] [YWH] [JHWH] [>T MCH]
Leviticus	17	4	[W] [>L PTX >HL MW<D] [L>] [HBJ>W] [L <FWT] [>TW] [<LH >W CLMJM] [L JHWH] [L RYWNKM] [L RJX NJXX] [W] [JCXVHW] [B XWY] [W] [>L PTX >HL MW<D] [L>] [HBJ>W] [L HQRJBW] [QRBN] [L JHWH] [L PNJ MCKN JHWH] [DM] [JXCB] [L >JC H HW>] [DM] [CPK] [W] [NKRT] [H >JC H HW>] [M QRB <MJW]
Numbers	4	14	[W] [NTNW] [<LJW] [>T KL KLJW] [>CR] [JCRTW] [<LJW] [BHM] [>T H MXTWT W >T H MZLGWT W >T H J<JM W >T H MZRQWT] [KL KLJ H MZBX] [W] [PRFW] [<LJW] [KSWJ <WR TXC] [W] [FMW] [BDJW] [W] [LQXW] [BGD >RGMN] [W] [KSW] [>T H KJWR W >T KNW] [W] [NTNW] [>TM] [>L MKSH <WR TXC] [W] [NTNW] [<L H MWV]
Numbers	10	10	[W] [B JWM FMXTJKM W B MW<DJKM W B R>CJ XDCJKM] [W] [TQ<TM] [B XYYRWT] [<L <LTJKM W <L ZBXJ CLMJKM] [W] [HJW] [LKM] [L ZKRWN] [L PNJ JHWH] [>LHJKM] [>NJ] [JHWH] [>LHJKM] [W] [JDBR] [JHWH] [>L MCH] [L >MR] [RB] [LKM] [CBT] [B HR H ZH] [PNW] [W] [S<W] [LKM] [W] [B>W] [HR H >MRJ] [W] [>L KL CKJNW] [B <RBH B HR W B CPLH B NGB W B XWP H JM] [>RY H KN<NJ] [W] [H LBNWN] [<D H NHR H GDWL] [NHR PRT] [R>W] [NTTJ] [L PNJKM] [>T H >RY] [B>W] [W] [RCW] [>T H >RY] [>CR] [NCB<TJ] [L >BTJKM] [L >BRHM L JYXQ W L J<QB] [L TT] [L ZR<M] [>XRJHM]
'''

# %%
word_num = 0
for i in Ronaldo.split('\n'):
    if len(i)>0:
        result = ''
        start_phrase_atom = False
        book = i.split('\t')[0]
        chapter = i.split('\t')[1]
        verse = i.split('\t')[2]
        verse_text = i.split('\t')[-1]
        phrase_atom = verse_text.split(' ')
        for word in phrase_atom:
            word_num = word_num+1
            if word[0] == '[':
                start_phrase_atom = True
            if word[-1] == ']' and start_phrase_atom == True:
                result = result + 'Y'
                start_phrase_atom = False
            else:
                result = result + 'X'
        
        print(book, chapter, verse, result, sep='\t')
print(word_num)

# %%
print((1-88/word_num)*100)

# %%
UPDATE = '7'

# %%
inputfilePath = "../new_data/output_morph_phrase_SP"
resultfile = f"../new_data/output_morph_phrase_SP_test_up{UPDATE}"
outputfilePath = f"../new_data/output_morph_phrase_SP_up{UPDATE}"

verse_old = None
result_old = ""
book_old = None
chapter_old = None

output = open(outputfilePath, 'w')

linhas_arquivo = []
with open(resultfile, 'r') as f:
    for line in f:
        line = line.strip().split('\t')
        linhas_arquivo.append(line)

file_dict = {}
for line in linhas_arquivo:
    key = (line[0], line[1], line[2])
    value = line[3]
    file_dict[key] = value

with open(inputfilePath, 'r') as input:
    
    inputlines = input.readlines()
    for i in range (0, len(inputlines)):
        word = inputlines[i].strip().split()
        data = word[:5]
        key = tuple(word[1:4])
        result = file_dict[key]
        verse = word[3]
        if verse != verse_old:
            j=0
            verse_old = verse
        else:
            j=j+1
        print(data[0],data[1],data[2],data[3],data[4],result[j],sep='\t',file=output)

output.close()

# %%
# Leitura dos arquivos
with open(f'../new_data/output_morph_phrase_SP_up{UPDATE}', 'r') as f1, open('../new_data/node_words_SP', 'r') as f2:
    linhas_arquivo1 = [linha.strip().split('\t') for linha in f1.readlines()]
    linhas_arquivo2 = [linha.strip().split('\t') for linha in f2.readlines()]

# Caracteres a serem removidos
morphsign = "~!>+~[]/<>"
translation_table = str.maketrans('', '', morphsign)

# Dicionário para correspondência entre a coluna 5 do arquivo 1 e a coluna 2 do arquivo 2
dict_arq2 = {}
index = 405426
maqaf = False

# Preenchendo o dicionário com valores de node_words_SP
for coluna1, coluna2 in linhas_arquivo2:
    coluna2 = coluna2.translate(translation_table)
    
    # Separar palavras compostas e armazenar no dicionário
    if '_' in coluna2:
        coluna2, coluna2b = map(str, coluna2.split('_'))
        maqaf = True
    else:
        maqaf = False
    dict_arq2[index] = [coluna1, coluna2, maqaf]
    index = index+1

index = 405426
arquivo1 = {}

for linha in linhas_arquivo1:
    arquivo1[index] = [str(index), linha[4].translate(translation_table), linha[5]]
    index = index+1

index = 405426
newfile = {}
j=0

for i in range(index, index+len(dict_arq2)):
    if dict_arq2[i][2] == True:
        j=j+1
        newfile[i] = [str(i), dict_arq2[i][1], arquivo1[i+j][2]]
    else:
        newfile[i] = [i, dict_arq2[i][1], arquivo1[i+j][2]]

index = 405426
old_value = 'X'
old_node = index

caminho_arquivo3 = f'../new_data/phrase_atom_up{UPDATE}'
f3 = open(caminho_arquivo3, 'w')

for i in range(index, index+len(newfile)):
    current_value = newfile[i][2]
    current_node = newfile[i][0]
    
    if current_value == 'X' and old_value == 'X':
        pass
    elif current_value == 'X' and old_value == 'Y':
        old_value = current_value
        old_node = current_node
    elif current_value == 'Y' and old_value == 'X':
        print(f"{old_node}-{current_node}", file=f3)
        old_node = current_node
        old_value = current_value
    elif current_value == 'Y' and old_value == 'Y':
        print(f"{current_node}", file=f3)
        old_node = current_node

f3.close()

# %%
UPDATE = '8'

# %%
inputfilePath = "../new_data/output_morph_phrase_SP"
resultfile = f"../new_data/output_morph_phrase_SP_test_up{UPDATE}"
outputfilePath = f"../new_data/output_morph_phrase_SP_up{UPDATE}"

verse_old = None
result_old = ""
book_old = None
chapter_old = None

output = open(outputfilePath, 'w')

linhas_arquivo = []
with open(resultfile, 'r') as f:
    for line in f:
        line = line.strip().split('\t')
        linhas_arquivo.append(line)

file_dict = {}
for line in linhas_arquivo:
    key = (line[0], line[1], line[2])
    value = line[3]
    file_dict[key] = value

with open(inputfilePath, 'r') as input:
    
    inputlines = input.readlines()
    for i in range (0, len(inputlines)):
        word = inputlines[i].strip().split()
        data = word[:5]
        key = tuple(word[1:4])
        result = file_dict[key]
        verse = word[3]
        if verse != verse_old:
            j=0
            verse_old = verse
        else:
            j=j+1
        print(data[0],data[1],data[2],data[3],data[4],result[j],sep='\t',file=output)

output.close()

# %%
# Leitura dos arquivos
with open(f'../new_data/output_morph_phrase_SP_up{UPDATE}', 'r') as f1, open('../new_data/node_words_SP', 'r') as f2:
    linhas_arquivo1 = [linha.strip().split('\t') for linha in f1.readlines()]
    linhas_arquivo2 = [linha.strip().split('\t') for linha in f2.readlines()]

# Caracteres a serem removidos
morphsign = "~!>+~[]/<>"
translation_table = str.maketrans('', '', morphsign)

# Dicionário para correspondência entre a coluna 5 do arquivo 1 e a coluna 2 do arquivo 2
dict_arq2 = {}
index = 405426
maqaf = False

# Preenchendo o dicionário com valores de node_words_SP
for coluna1, coluna2 in linhas_arquivo2:
    coluna2 = coluna2.translate(translation_table)
    
    # Separar palavras compostas e armazenar no dicionário
    if '_' in coluna2:
        coluna2, coluna2b = map(str, coluna2.split('_'))
        maqaf = True
    else:
        maqaf = False
    dict_arq2[index] = [coluna1, coluna2, maqaf]
    index = index+1

index = 405426
arquivo1 = {}

for linha in linhas_arquivo1:
    arquivo1[index] = [str(index), linha[4].translate(translation_table), linha[5]]
    index = index+1

index = 405426
newfile = {}
j=0

for i in range(index, index+len(dict_arq2)):
    if dict_arq2[i][2] == True:
        j=j+1
        newfile[i] = [str(i), dict_arq2[i][1], arquivo1[i+j][2]]
    else:
        newfile[i] = [i, dict_arq2[i][1], arquivo1[i+j][2]]

index = 405426
old_value = 'X'
old_node = index

caminho_arquivo3 = f'../new_data/phrase_atom_up{UPDATE}'
f3 = open(caminho_arquivo3, 'w')

for i in range(index, index+len(newfile)):
    current_value = newfile[i][2]
    current_node = newfile[i][0]
    
    if current_value == 'X' and old_value == 'X':
        pass
    elif current_value == 'X' and old_value == 'Y':
        old_value = current_value
        old_node = current_node
    elif current_value == 'Y' and old_value == 'X':
        print(f"{old_node}-{current_node}", file=f3)
        old_node = current_node
        old_value = current_value
    elif current_value == 'Y' and old_value == 'Y':
        print(f"{current_node}", file=f3)
        old_node = current_node

f3.close()

# %% [markdown]
# ## May 7, 8, 12 - verses that needed review

# %% [markdown]
# I finished the manual correction of the predicted phrase atoms in most of the SP verses.\
# There are a few verses that I am struggling in understanding if the prediction is correct.\
# The rest of the manual corrections were made available in the version 0.5.8
# 
# Verses that needed review:
# 
# - Genesis 6:9
# - Genesis 12:16
# - Exodus 22:6
# - Exodus 25:27
# - Exodus 27:12
# - Exodus 29:5
# - Leviticus 13:6
# - Leviticus 15:3
# - Numbers 7:85
# - Numbers 21:24
# - Deuteronomy 13:7
# - Deuteronomy 19:18
# - Deuteronomy 28:20
# - Deuteronomy 32:35
# 
# Using those manual corrections, I made available a new version, version 0.5.9
# 
# There are still some verses that need review, since they are a bit tricky.
# 
# - Exodus 12:15  - does vav break phrase_atom boundaries?
# - Leviticus 1:8  - does vav unite phrase_atom boundaries?
# - Leviticus 27:30  - does vav divide phrase_atom boundaries?
# - Numbers 8:4 - expression <D ... <D ... - I think that SP is correct (see Deuteronomy 28:20, but that case has verbs rather than nouns)
# 
# __A new version will be released with those corrections soon.__

# %%
UPDATE = '9'

# %%
inputfilePath = "../new_data/output_morph_phrase_SP"
resultfile = f"../new_data/output_morph_phrase_SP_test_up{UPDATE}"
outputfilePath = f"../new_data/output_morph_phrase_SP_up{UPDATE}"

verse_old = None
result_old = ""
book_old = None
chapter_old = None

output = open(outputfilePath, 'w')

linhas_arquivo = []
with open(resultfile, 'r') as f:
    for line in f:
        line = line.strip().split('\t')
        linhas_arquivo.append(line)

file_dict = {}
for line in linhas_arquivo:
    key = (line[0], line[1], line[2])
    value = line[3]
    file_dict[key] = value

with open(inputfilePath, 'r') as input:
    
    inputlines = input.readlines()
    for i in range (0, len(inputlines)):
        word = inputlines[i].strip().split()
        data = word[:5]
        key = tuple(word[1:4])
        result = file_dict[key]
        verse = word[3]
        if verse != verse_old:
            j=0
            verse_old = verse
        else:
            j=j+1
        print(data[0],data[1],data[2],data[3],data[4],result[j],sep='\t',file=output)
        
output.close()

# %%
# Leitura dos arquivos
with open(f'../new_data/output_morph_phrase_SP_up{UPDATE}', 'r') as f1, open('../new_data/node_words_SP', 'r') as f2:
    linhas_arquivo1 = [linha.strip().split('\t') for linha in f1.readlines()]
    linhas_arquivo2 = [linha.strip().split('\t') for linha in f2.readlines()]

# Caracteres a serem removidos
morphsign = "~!>+~[]/<>"
translation_table = str.maketrans('', '', morphsign)

# Dicionário para correspondência entre a coluna 5 do arquivo 1 e a coluna 2 do arquivo 2
dict_arq2 = {}
index = 405426
maqaf = False

# Preenchendo o dicionário com valores de node_words_SP
for coluna1, coluna2 in linhas_arquivo2:
    coluna2 = coluna2.translate(translation_table)
    
    # Separar palavras compostas e armazenar no dicionário
    if '_' in coluna2:
        coluna2, coluna2b = map(str, coluna2.split('_'))
        maqaf = True
    else:
        maqaf = False
    dict_arq2[index] = [coluna1, coluna2, maqaf]
    index = index+1

index = 405426
arquivo1 = {}

for linha in linhas_arquivo1:
    arquivo1[index] = [str(index), linha[4].translate(translation_table), linha[5]]
    index = index+1

index = 405426
newfile = {}
j=0

for i in range(index, index+len(dict_arq2)):
    if dict_arq2[i][2] == True:
        j=j+1
        newfile[i] = [str(i), dict_arq2[i][1], arquivo1[i+j][2]]
    else:
        newfile[i] = [i, dict_arq2[i][1], arquivo1[i+j][2]]

index = 405426
old_value = 'X'
old_node = index

caminho_arquivo3 = f'../new_data/phrase_atom_up{UPDATE}'
f3 = open(caminho_arquivo3, 'w')

for i in range(index, index+len(newfile)):
    current_value = newfile[i][2]
    current_node = newfile[i][0]
    
    if current_value == 'X' and old_value == 'X':
        pass
    elif current_value == 'X' and old_value == 'Y':
        old_value = current_value
        old_node = current_node
    elif current_value == 'Y' and old_value == 'X':
        print(f"{old_node}-{current_node}", file=f3)
        old_node = current_node
        old_value = current_value
    elif current_value == 'Y' and old_value == 'Y':
        print(f"{current_node}", file=f3)
        old_node = current_node

f3.close()

# %% [markdown]
# ## May 18 - creating a new SP dataset using version 3.4.2

# %%
SP_old = use('DT-UCPH/sp', version='3.4.1')
Fsp_old, Lsp_old, Tsp_old = SP_old.api.F, SP_old.api.L, SP_old.api.T

SP = use('DT-UCPH/sp', version='3.4.2')
Fsp, Lsp, Tsp = SP.api.F, SP.api.L, SP.api.T

# %%
outputfilePath = "../new_data/data_SP_341"
output = open(outputfilePath, "w")

print('node', 'bo', 'ch', 've', 'SP', sep='\t', file=output)

for verse_node in Fsp_old.otype.s('verse'):
    bo, ch, ve = Tsp_old.sectionFromNode(verse_node)
    word_nodes = Lsp_old.d(verse_node, 'word')
    for word_node in word_nodes:
        word_text = Fsp_old.g_cons.v(word_node)
        print(word_node, bo, ch, ve, word_text, sep='\t', file=output)

output.close()

# %%
outputfilePath = "../new_data/data_SP_342"
output = open(outputfilePath, "w")

print('node', 'bo', 'ch', 've', 'SP', sep='\t', file=output)

outputfile2 = "../new_data/output_morph_phrase_SP_342"
output2 = open(outputfile2, "w")

i=0

maqof = 0

for verse_node in Fsp.otype.s('verse'):
    bo, ch, ve = Tsp.sectionFromNode(verse_node)
    word_nodes = Lsp.d(verse_node, 'word')
    for word_node in word_nodes:
        word_text = Fsp.g_cons.v(word_node)
        print(word_node, bo, ch, ve, word_text, sep='\t', file=output)

        #Updating output_morph_phrase_SP file
        if Fsp.g_lex.v(word_node) == 'absent': #if g_lex is absent, we are using g_cons
            word_morph = Fsp.g_pfm.v(word_node) + Fsp.g_vbs.v(word_node) + Fsp.g_cons.v(word_node) + Fsp.g_vbe.v(word_node) + Fsp.g_nme.v(word_node) + Fsp.g_uvf.v(word_node) + Fsp.g_prs.v(word_node)
        else:
            word_morph = Fsp.g_pfm.v(word_node) + Fsp.g_vbs.v(word_node) + Fsp.g_lex.v(word_node) + Fsp.g_vbe.v(word_node) + Fsp.g_nme.v(word_node) + Fsp.g_uvf.v(word_node) + Fsp.g_prs.v(word_node)
        
        #print(f"{word_node - 405426}\t{bo} {ch} {ve}\t{word_morph}", file=output2)

        if "_" in word_morph:
            word_morph1 = word_morph.split("_")[0]
            word_morph2 = word_morph.split("_")[1]
            print(f"{word_node - 405426 + maqof}\t{bo} {ch} {ve}\t{word_morph1}", file=output2)
            maqof = maqof + 1
            print(f"{word_node - 405426 + maqof}\t{bo} {ch} {ve}\t{word_morph2}", file=output2)
        else:
            print(f"{word_node - 405426 + maqof}\t{bo} {ch} {ve}\t{word_morph}", file=output2)

output.close()
output2.close()

# %%
!diff -y --suppress-common-lines ../new_data/data_SP_342 ../new_data/data_SP_341 | head -n 5

# %% [markdown]
# The only difference between versions 3.4.1 and 3.4.2 is that the `g_cons` of the first word in Genesis 6:18 is "'WHQMTJ'".\
# In 3.4.1 this is one word, whereas in 3.4.2 it is split in two words: "W" and "'HQMTJ'".

# %%
ref = "Genesis 6:18"

bo = ref.split(" ")[0]
ch = ref.split(" ")[1].split(":")[0]
ve = ref.split(" ")[1].split(":")[1]

if bo == 'Deuteronomy':
    bo = "Deuteronomy|Deuteronomium"
elif bo == 'Numbers':
    bo = "Numbers|Numeri"

text = f"""
book book={bo}
    chapter chapter={ch}
        verse verse={ve}
"""
results_SP_old = SP_old.search(f'{text}')
results_SP = SP.search(f'{text}')
#results_MT = MT.search(f'{text}')
#MT.show(results_MT, end=1, multiFeatures=False, queryFeatures=False, condensed=True,
#        hiddenTypes={'phrase','half_verse','subphrase', 'sentence', 'sentence_atom', 'clause', 'clause_atom'})
SP.show(results_SP, end=1, multiFeatures=False, queryFeatures=False, condensed=True)
SP_old.show(results_SP_old, end=1, multiFeatures=False, queryFeatures=False, condensed=True)

# %%
!cat ../new_data/comparison_output_morph_phrase_SP_output_phrase_space_pdp_SP_MT_word_up | awk '{print $1 "\t" $2, $3, $4 "\t" $5 "\t" $6}' | tail -n +2 > ../new_data/output_morph_phrase_SP

# %% [markdown]
# Updating `node_words_SP_342` file

# %%
results = SP.search("""
word
""")

outputfilePath = f"../new_data/node_words_SP_342"
output = open(outputfilePath, "w")

for i in range(0,len(results)):
    print(results[i][0], Fsp.g_cons.v(results[i][0]), file=output, sep='\t')

# %% [markdown]
# Version 3.4.1 has 114891 words, while version 3.4.2 has 114892 words.\
# In version 3.4.1, words are defined by the node 405426 to 520316, while in version 3.4.2, words are defined by the node 405426 to 520317.

# %%
UPDATE = '10'

# %%
inputfilePath = "../new_data/output_morph_phrase_SP_342"
resultfile = f"../new_data/output_morph_phrase_SP_test_up{UPDATE}" #file added ו in Genesis 6:18 (correspondent to X) and without manual corrections for Exod 12:15; Lev 1:8; Lev 27:30; Num 8:4
outputfilePath = f"../new_data/output_morph_phrase_SP_342_up{UPDATE}"

verse_old = None
result_old = ""
book_old = None
chapter_old = None

output = open(outputfilePath, 'w')

linhas_arquivo = []
with open(resultfile, 'r') as f:
    for line in f:
        line = line.strip().split('\t')
        linhas_arquivo.append(line)

file_dict = {}
for line in linhas_arquivo:
    key = (line[0], line[1], line[2])
    value = line[3]
    file_dict[key] = value

with open(inputfilePath, 'r') as input:
    
    inputlines = input.readlines()
    for i in range (0, len(inputlines)):
        word = inputlines[i].strip().split()
        data = word[:5]
        key = tuple(word[1:4])
        result = file_dict[key]
        verse = word[3]
        if verse != verse_old:
            j=0
            verse_old = verse
        else:
            j=j+1
        print(data[0],data[1],data[2],data[3],data[4],result[j],sep='\t',file=output)
        
output.close()

# %%
# Leitura dos arquivos
with open(f'../new_data/output_morph_phrase_SP_342_up{UPDATE}', 'r') as f1, open('../new_data/node_words_SP_342', 'r') as f2:
    linhas_arquivo1 = [linha.strip().split('\t') for linha in f1.readlines()]
    linhas_arquivo2 = [linha.strip().split('\t') for linha in f2.readlines()]

# Caracteres a serem removidos
morphsign = "~!>+~[]/<>"
translation_table = str.maketrans('', '', morphsign)

# Dicionário para correspondência entre a coluna 5 do arquivo 1 e a coluna 2 do arquivo 2
dict_arq2 = {}
index = 405426
maqaf = False

# Preenchendo o dicionário com valores de node_words_SP
for coluna1, coluna2 in linhas_arquivo2:
    coluna2 = coluna2.translate(translation_table)
    
    # Separar palavras compostas e armazenar no dicionário
    if '_' in coluna2:
        coluna2, coluna2b = map(str, coluna2.split('_'))
        maqaf = True
    else:
        maqaf = False
    dict_arq2[index] = [coluna1, coluna2, maqaf]
    index = index+1

index = 405426
arquivo1 = {}

for linha in linhas_arquivo1:
    arquivo1[index] = [str(index), linha[4].translate(translation_table), linha[5]]
    index = index+1

index = 405426
newfile = {}
j=0

for i in range(index, index+len(dict_arq2)):
    if dict_arq2[i][2] == True:
        j=j+1
        newfile[i] = [str(i), dict_arq2[i][1], arquivo1[i+j][2]]
    else:
        newfile[i] = [i, dict_arq2[i][1], arquivo1[i+j][2]]

index = 405426
old_value = 'X'
old_node = index

caminho_arquivo3 = f'../new_data/phrase_atom_up{UPDATE}'
f3 = open(caminho_arquivo3, 'w')

for i in range(index, index+len(newfile)):
    current_value = newfile[i][2]
    current_node = newfile[i][0]
    
    if current_value == 'X' and old_value == 'X':
        pass
    elif current_value == 'X' and old_value == 'Y':
        old_value = current_value
        old_node = current_node
    elif current_value == 'Y' and old_value == 'X':
        print(f"{old_node}-{current_node}", file=f3)
        old_node = current_node
        old_value = current_value
    elif current_value == 'Y' and old_value == 'Y':
        print(f"{current_node}", file=f3)
        old_node = current_node

f3.close()

# %%



