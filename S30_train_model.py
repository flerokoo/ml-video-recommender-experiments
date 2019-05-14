import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, \
    Flatten, Input, Masking, LSTM, CuDNNLSTM
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
import pandas as pd
import numpy as np
import options
import os
import math
import time
import gc
import utils

videos = pd.read_csv("data/raw/ml_videos.csv", dtype={"Tags": str, "Studio ID": str})
max_video_id = max(videos["ID"])
videos = None
# CLASSES = math.ceil(max_video_id / 10000)*10000
CLASSES = max_video_id + 1
PADDED_SEQUENCE_COL_INDEX = 2
ANSWER_COL_INDEX = 3

def get_generators(input_file_name, validation_split=0.1, per_step=100):
    lines_num = utils.get_num_of_lines(input_file_name)
    split_index = round(lines_num * (1-validation_split))

    train_i0 = 1 # skip headers
    train_i1 = split_index - 1
    
    valid_i0 = split_index
    valid_i1 = lines_num
    
    n_train = train_i1 - train_i0
    n_valid = valid_i1 - valid_i0
    
    steps_train = math.floor(n_train / per_step)
    steps_valid = math.floor(n_valid / per_step)

    def generate(start, stop, steps):
        step = 0
        while True:
            i0 = start + step * per_step
            i1 = i0 + per_step
            
            df = pd.read_csv(input_file_name, skiprows=i0, nrows=(i1 - i0), header=None)
            sequences = []
            answers = []

            for index, row in df.iterrows():      
                sequence = list(map(int, row[PADDED_SEQUENCE_COL_INDEX].split(",")))                    
                answer = int(row[ANSWER_COL_INDEX])
                sequences.append(to_categorical(sequence, CLASSES))                
                answers.append([answer])

            yield (
                np.array(sequences),
                np.array(answers)
            )

            step += 1
            if step == steps:
                step = 0

    return (
        generate(train_i0, train_i1, steps_train),
        generate(valid_i0, valid_i1, steps_valid),
        steps_train,
        steps_valid
    )

def get_model():      
    model = Sequential()
    model.add(CuDNNLSTM(1200, return_sequences=True, input_shape=(options.DEFAULT_SEQ_LEN, CLASSES)))
    model.add(CuDNNLSTM(600, return_sequences=False)) # replace to False and remove Flatten
    # model.add(Dropout(0.1))
    # model.add(Flatten())
    model.add(Dense(CLASSES))
    model.add(Activation("softmax"))
    model.compile(loss="sparse_categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"])
    return model



if __name__ == "__main__":
    
    train_gen, valid_gen, train_steps, valid_steps = get_generators("data/sliding_sequences.csv")

    timestamp = time.ctime(time.time()).replace(" ", "_").replace(":", "-")

    print(f"nn-{timestamp}")
    tensorboard = TensorBoard(f"logs/model-{timestamp}/") 
    checkpoint = ModelCheckpoint(f"models/trained-best-{timestamp}", save_best_only=True)
    model = get_model()
    print(model.summary())
    print(f"""
        TRAINING 
        train steps = {train_steps}
        validations steps = {valid_steps}
        per step = 100
        timestamp = {timestamp}
    """)

    model.fit_generator(train_gen,
        steps_per_epoch=train_steps,
        validation_data=valid_gen,
        validation_steps=valid_steps,
        epochs=20,
        callbacks=[tensorboard, checkpoint])    

    model.save("models/trained-final-{timestamp}.model")

# videos = pd.read_csv("data/raw/ml_videos.csv", dtype={"Tags": str, "Studio ID": str})
# max_video_id = max(videos["ID"])
# videos = None
# # CLASSES = math.ceil(max_video_id / 10000)*10000
# CLASSES = max_video_id+1
# MAX_SEQS = 2500

# def get_dataset(max=MAX_SEQS, preprocess=None):
#     actions_data = pickle.load(open("data/sliding_window_sequences_padded.pickle", "rb"))
#     sequences_list = []
#     answer_list = []

#     if preprocess != None:
#         preprocess(actions_data)

#     for action in actions_data:    
#         for seqi in range(len(action["sequences"])):
#             seq_data = action["sequences"][seqi]
#             seq_array = seq_data["sequence"]
#             seq_answer = seq_data["next"]
#             sequences_list.append(to_categorical(seq_array, CLASSES))
#             # answer_list.append(to_categorical([seq_answer], CLASSES))
#             answer_list.append([seq_answer]) # alright when using sparse_categorical_crossentropy
        
#         if len(sequences_list) > max:
#             break
#     return (np.array(sequences_list), np.array(answer_list))



# def get_dataset_generators(steps, max_seqs_per_batch=100):
#     actions_data = pickle.load(open("data/sliding_window_sequences_padded.pickle", "rb"))
    
#     # convert to plain arrays of sequences and answers
#     answer_list = []
#     sequences_list = []
#     for action in actions_data:    
#         for seqi in range(len(action["sequences"])):
#             seq_data = action["sequences"][seqi]
#             seq_array = seq_data["sequence"]
#             seq_answer = seq_data["next"]
#             sequences_list.append(seq_array)
#             # answer_list.append(to_categorical([seq_answer], CLASSES))
#             answer_list.append([seq_answer]) # alright when using sparse_categorical_crossentropy
        

#     print(f"TOTAL SEQUENCES: {len(sequences_list)}")

#     # split dataset
#     split_index = round(len(sequences_list) * 0.9) - 1;
#     sequences_list_valid = sequences_list[split_index+1:]
#     sequences_list = sequences_list[:split_index]
#     answer_list_valid = answer_list[split_index+1:]
#     answer_list = answer_list[:split_index]
    
    
#     def get_generator(seqs, anss, steps):
#         if len(seqs) != len(anss):
#             raise Exception("WTF BRO")

#         if len(seqs) / max_seqs_per_batch < steps:
#             raise Exception(
#                 f"not enough data: steps={steps} max_seqs_per_batch={max_seqs_per_batch} data_len={len(seqs)}")
            
#         step = 0
#         while True:
#             start = step * max_seqs_per_batch
#             stop = (step + 1) * max_seqs_per_batch

#             yield (
#                 np.array(to_categorical(seqs[start:stop], CLASSES)),
#                 np.array(anss[start:stop])
#             )
            
#             step += 1
#             if step == steps:
#                 step = 0
#                 gc.collect()

#     validation_steps = math.floor(len(sequences_list_valid) / max_seqs_per_batch)
    
#     return (
#         get_generator(sequences_list, answer_list, steps),
#         get_generator(sequences_list_valid, answer_list_valid, validation_steps),
#         validation_steps
#     )
            
    


# if __name__ == "__main__":
#     timestamp = time.time()
#     print(f"nn-{timestamp}")
#     tensorboard = TensorBoard(f"logs/seq-aware-{timestamp}/")
   
    

#     model = Sequential()
#     model.add(LSTM(100, return_sequences=True, input_shape=(options.DEFAULT_SEQ_LEN, CLASSES)))
#     model.add(LSTM(100, return_sequences=True))
#     # model.add(Dropout(0.1))
#     model.add(Flatten())
#     model.add(Dense(CLASSES))
#     model.add(Activation("softmax"))

#     model.compile(loss="sparse_categorical_crossentropy",
#         optimizer="adam",
#         metrics=["accuracy"])

#     print(model.summary())

#     # TRAIN ONCE
#     # sequences_list, answer_list = get_dataset()
#     # model.fit(sequences_list, answer_list, epochs=18,
#     #     batch_size=10, validation_split=0.1, callbacks=[tensorboard])

#     # TRAIN BY GENERATORS
#     steps_per_epoch = 1200
#     train_gen, valid_gen, valid_steps = get_dataset_generators(steps_per_epoch)
#     model.fit_generator(train_gen,
#         steps_per_epoch=steps_per_epoch,
#         validation_data=valid_gen,
#         validation_steps=valid_steps,
#         epochs=20,
#         callbacks=[tensorboard])    

#     model.save("trained.model")



