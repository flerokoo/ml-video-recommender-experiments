import pickle
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, \
    Flatten, Input, Masking, LSTM, CuDNNLSTM, Embedding, GRU, CuDNNGRU
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

def get_generators(input_file_name, validation_split=0.05, per_step=100):
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
    inp_shape = (options.DEFAULT_SEQ_LEN, CLASSES)
    model = Sequential()
    model.add(CuDNNLSTM(800, return_sequences=True, input_shape=inp_shape))
    model.add(CuDNNLSTM(600, return_sequences=True)) 
    model.add(CuDNNLSTM(400, return_sequences=True)) 
    model.add(CuDNNLSTM(600, return_sequences=False)) # replace to False and remove Flatten
    model.add(Dropout(0.05))
    # model.add(Flatten())
    model.add(Dense(CLASSES))
    model.add(Activation("softmax"))
    model.compile(loss="sparse_categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"])
    return model



if __name__ == "__main__":
    
    batch_size = 1000
    train_gen, valid_gen, train_steps, valid_steps = get_generators("data/sliding_sequences_1000.csv", per_step=batch_size)

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
        per step = {batch_size}
        timestamp = {timestamp}
        classes = {CLASSES}
    """)

    model.fit_generator(train_gen,
        steps_per_epoch=train_steps,
        validation_data=valid_gen,
        validation_steps=valid_steps,
        epochs=20,
        # callbacks=[tensorboard, checkpoint]
        )    

    model.save("models/trained-final-{timestamp}.model")
