# %%
import os

# %%
def create_temp_file():
    temp_file_path = 'temp_file.txt'
    with open(temp_file_path, 'w') as temp_file:
        temp_file.write('This is a temporary file.')
    return temp_file_path
