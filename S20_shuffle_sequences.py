from utils import get_num_of_lines
import random
import os
import math

cached_lines = []
def get_line_content(line_index, source):
    global cached_lines
    if len(cached_lines) == 0:
        file = open(source, "r", newline="")
        cached_lines = file.readlines()
        
    return cached_lines[line_index]

def shuffle(source, target, buffer_max_size = 100000):
    outfile = open(target, "a", newline='')
    
    
    lines = get_num_of_lines(source)
    arr = [i for i in range(1, lines)]

    random.shuffle(arr)

    buffer_max_size = max(buffer_max_size, 1)
    buffer = ""
    buffer_size = 0

    # write header
    outfile.write(get_line_content(0, source))

    for i in range(lines-1):
        line_index = arr[i]        
        line = get_line_content(line_index, source)
        line = line.replace("\n", "")
        
        buffer += line + '\n'
        buffer_size += 1

        if buffer_size > buffer_max_size:
            outfile.write(buffer)
            buffer = ""
            buffer_size = 0

        if i % 100 == 0: print(i)
    
    if buffer_size > 0:
        outfile.write(buffer)


if __name__ == "__main__":
    shuffle("data/sliding_sequences_1000.csv", "data/shuffled_sliding_1000.csv")
    # shuffle("data/test.csv", "data/test_sh.csv")
