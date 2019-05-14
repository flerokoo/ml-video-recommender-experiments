import numpy as np
import tensorflow as tf
import pickle
import pandas as pd
from options import DEFAULT_SEQ_LEN, MIN_SEQ_LEN
import os

def one_to_multiple_sequences(arr):
    ret_arr = []
    for i in range(len(arr)-1):
        new_seq = []        
        for seq_i in range(DEFAULT_SEQ_LEN):
            real_i = i - seq_i
            if real_i < 0:
                # new_seq.append(0)
                pass
            else:
                new_seq.append(arr[real_i])
        new_seq.reverse()
        ret_arr.append({
            "sequence": new_seq,
            "next": arr[i+1]
        })    
    return ret_arr


def pad_sequence(seq, leng=DEFAULT_SEQ_LEN):
    new_seq = seq[:]
    while len(new_seq) < leng:
        new_seq.insert(0, 0)
    return new_seq
    
    
def slide_window_over_file(input_file_name, output_file_name, verbalize=True):

    if os.path.exists(output_file_name):
        os.remove(output_file_name)

    outfile = open(output_file_name, "a", newline='')

    is_first_row = True
    num_of_seqs = 0
    for chunk in pd.read_csv(input_file_name, chunksize=1000000):
        for index, row in chunk.iterrows():
            user_id = row["user_id"]
            sequence = list(map(int, row["sequence"].split(",")))
            seqs = one_to_multiple_sequences(sequence)
            for seq in seqs:
                padded = pad_sequence(seq["sequence"])
                data = {
                    "user_id": [user_id],
                    "sequence": ",".join(str(x) for x in seq["sequence"]),
                    "padded": ",".join(str(x) for x in padded),
                    "next": [seq["next"]]
                }
                df = pd.DataFrame(data=data)
                df.to_csv(outfile, index=False, header=is_first_row)
                is_first_row = False
                num_of_seqs += 1

    if verbalize:
        print(f"Generated {num_of_seqs} sequences of max len {DEFAULT_SEQ_LEN}")



if __name__ == "__main__":
    slide_window_over_file("data/sequences.csv", "data/sliding_sequences.csv")

# all_sequences = pickle.load(open("data/raw_video_sequences.pickle", "rb"))
# sequences = list(filter(lambda a: len(a["sequence"]) > MIN_SEQ_LEN, all_sequences))

# processed_sequences = list(map(lambda a: one_to_multiple_sequences(a), sequences))

# print(processed_sequences[1:5])

# pickle.dump(processed_sequences, open("data/sliding_window_sequences.pickle", "wb"))

# total = 0
# for s in processed_sequences:
#     total += len(s["sequences"])
# print(f"Generated {total} unpadded sequences with max length of {DEFAULT_SEQ_LEN}")


# output: [{
# "user_id" : "asdasd",
# "sequences": [
#      "sequence": [1, 2, 3...] <- sequence of videos (X dataset)
#       next": 12 <- id of next watched video (Y dataset)
# ]]