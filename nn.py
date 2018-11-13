import os
import num2words
import random
import numpy as np
from glob import glob
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Activation, Dropout

# All unique characters used in numbers
chars = list(" abdefghilmnorstuvwxyz,-\n")

# the fixed vector, the window of characters we're looking at
max_len = 20

model = Sequential()
model.add(LSTM(512, return_sequences=True, input_shape=(max_len, len(chars))))
model.add(Dropout(0.2))
model.add(LSTM(512, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(len(chars)))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer="rmsprop")

steps = 3
inputs = []
outputs = []

char_labels = {ch:i for i, ch in enumerate(chars)}
labels_char = {i:ch for i, ch in enumerate(chars)}

# Generates a random number word
# The random numbers are weighted towards lower values
# so that the network learns about lower numbers occasionally
def rand_word():
  power = random.randint(1, 8)
  num = random.randint(0, 10 ** power)
  return num2words.num2words(num) + "\n"

def generate_examples():
  global inputs
  global outputs
  global X
  global y
  word = rand_word()
  while len(word) <= max_len+1:
    word += rand_word()
  # Example count is the number I did a lot of the tuning on. It sets how many examples the network has to learn from
  example_count = 6000
  for i in range(0, example_count, 1):
    inputs.append(word[0:max_len])
    outputs.append(word[max_len])
    word = word[steps:]
    while len(word) <= max_len+1:
      word += rand_word()

  X = np.zeros((len(inputs), max_len, len(chars)), dtype=np.bool)
  y = np.zeros((len(inputs), len(chars)), dtype=np.bool)

generate_examples()

# creates some one hot vectors
for i, example in enumerate(inputs):
  for t, char in enumerate(example):
    X[i, t, char_labels[char]] = 1
  y[i, char_labels[outputs[i]]] = 1

# Generates a chunk of text
def generate(temperature=0.35, seed=None, num_chars=200):
  predicate = lambda x: len(x) < num_chars

  if seed is not None and len(seed) < max_len:
    raise Exception('seed text must be at least {} chars long'.format(max_len))
  else:
    word = rand_word()
    while len(word) < max_len:
      word += rand_word()
    seed = word[-max_len:]

  sentence = seed
  generated = sentence

  while predicate(generated):
    x = np.zeros((1, max_len, len(chars)))
    for t, char in enumerate(sentence):
      x[0, t, char_labels[char]] = 1

    probs = model.predict(x, verbose=0)[0]
    next_idx = sample(probs, temperature)
    next_char = labels_char[next_idx]

    generated += next_char
    sentence = sentence[1:] + next_char
  # Removes the seed from the generated
  return generated[max_len:]

def sample(probs, temperature):
  a = np.log(probs)/temperature
  dist = np.exp(a)/np.sum(np.exp(a))
  choices = range(len(probs))
  return np.random.choice(choices, p=dist)

# Trains the network for 10 epochs
epochs = 10
for i in range(epochs):
  print('EPOCH %d'%(i+1))
  model.fit(X, y, batch_size = 128, epochs = 1, verbose=1)
  # Prints out some examples as it generates for tuning purposes
  for temp in [0.2, 0.5, 1., 1.2]:
    print('temperature: %0.2f'%temp)
    print('%s'%generate(temperature=temp))

# Saves the model to disk, so we can generate from it later.
# Models take a long time to generate, so it's nice to be able to mess with how it generates without having too rebuild it everytime
model_json = model.to_json()
with open("model.json", "w") as json_file:
  json_file.write(model_json)
model.save_weights('model.h5')
print("saved model to disk")