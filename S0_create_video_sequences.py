import pandas as pd
import pickle
import os
from options import MAX_TIMESTAMP_DIFFERENCE

def generate_sequences(row, min_len=2):
    user_id = row["ID"]
    favorites, watched_raw = row["Favorites"], row["Watched videos"].split(",")
    watched_vids_ids = list(map(lambda a: int(a), watched_raw[0::2]))
    watched_vids_timestamps = list(map(lambda a: int(a), watched_raw[1::2]))

    prev_timestamp = 0;        
    current_sequence = []
    sequences = []

    for i in range(len(watched_vids_ids)):
        idx = watched_vids_ids[i]
        timestamp = watched_vids_timestamps[i]
        if abs(timestamp - prev_timestamp) > MAX_TIMESTAMP_DIFFERENCE:
            sequences.append({
                "user_id": user_id,
                "sequence": current_sequence
            })
            current_sequence = []
        else:
            current_sequence.append(idx)
        prev_timestamp = timestamp        

    sequences = list(filter(lambda a: len(a["sequence"]) >= min_len, sequences))

    return sequences


def generate_sequences_from_file(source, target, verbalize=True):
    
    if os.path.exists(target):
        os.remove(target)
        
    file = open(target, "a", newline='')

    is_first_row = True
    num_of_seqs = 0
    for chunk in pd.read_csv(source, chunksize=100000):
        for index, row in chunk.iterrows():
            seqs = generate_sequences(row)
            for seq in seqs:
                seq["sequence"] = ",".join(str(x) for x in seq["sequence"])
                seq["user_id"] = [seq["user_id"]]
                df = pd.DataFrame(data=seq)
                df.to_csv(file, header=is_first_row, index=False)
                is_first_row = False
                num_of_seqs += 1

    if verbalize:
        print(f"Generated {num_of_seqs} sequences")
    


if __name__ == "__main__":
    # output: csv
    # user_id, sequence
    # xxx, "1,2,3"
    generate_sequences_from_file("data/raw/ml_users.csv", "data/sequences.csv")

# users = pd.read_csv("data/raw/ml_users.csv")
# sequences = generate_sequences(users)
# print(sequences[0:10])

# print(f"Generated {len(sequences)} sequences")

# pickle.dump(sequences, open("data/raw_video_sequences.pickle", "wb"))

