from S30_train_model import get_dataset, MAX_SEQS
from tensorflow.keras import models
import random

X, Y = get_dataset(MAX_SEQS, lambda a: random.shuffle(a, random.random))

model = models.load_model("trained.model")

result = model.evaluate(X, Y, batch_size=10)

print(result)

