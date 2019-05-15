from S30_train_model import get_generators, MAX_SEQS
from tensorflow.keras import models
import random

train_gen, valid_gen, train_steps, valid_steps = get_generators("data/sliding_sequences.csv")

X = train_gen()
Y = valid_gen()

model = models.load_model("trained.model")

result = model.evaluate(X, Y, batch_size=10)

print(result)

