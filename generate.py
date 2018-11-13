import os
import num2words
import re
import numpy as np
import random
from keras.models import Sequential
from keras.models import model_from_json
from word2number import w2n

# Loads the model that was created in nn.py
json_file = open('model.json', 'r')
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("model.h5")

# These elements are the same as the are in nn.py
max_len = 20
chars = list(" abdefghilmnorstuvwxyz,-\n")
char_labels = {ch:i for i, ch in enumerate(chars)}
labels_char = {i:ch for i, ch in enumerate(chars)}

def rand_word():
  power = random.randint(1, 8)
  num = random.randint(0, 10 ** power)
  return num2words.num2words(num) + "\n"

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
  return generated[max_len:]

def sample(probs, temperature):
  a = np.log(probs)/temperature
  dist = np.exp(a)/np.sum(np.exp(a))
  choices = range(len(probs))
  return np.random.choice(choices, p=dist)

# Checks if an string is a valid number, so we don't accidentally put any real numbers in our fake number book
def is_number(word):
  try:
    num = w2n.word_to_num(re.sub(",", "", word))
    # the word to number converter is accepts more variation than the number to word one
    # So all the 'and's, commas, and spaces are removed and then we compare
    # This might be an overzealous way to checking if a string is a valid number
    # But I'm fine with throwing out some good fakes in exchange for fewer reals
    just_letters = re.sub(r',|[\s,]and[\s,]|\s', "", word.lower())
    output_just_letters = re.sub(r',|[\s,]and[\s,]|\s', "", num2words.num2words(num).lower())
    return just_letters == output_just_letters
  except (ValueError, IndexError):
    return False

number_count = 0
f = open('50_Thousand_Fake_Numbers_With_Normal_Deviates.txt', 'a')
# We want 50,000 numbers, cause it's NaNoWriMo!
# However my computer prefers to take breaks sometimes, so I made it in chunks
# 15,955
while number_count < 20000:
  text = generate(temperature=1.2, num_chars=1000)
  numbers = text.split("\n")
  # we're skipping the last entry, since it's liable to be a fragment
  numbers.pop()
  for i in numbers:
    if not is_number(i):
      formatted = i.strip().capitalize()
      if not formatted == "":
        print(f"{number_count}: {formatted}")
        f.write(formatted + "\n")
        # this is a rough guess on how many words there are in this new number. Counting hyphenated words as a single word
        number_count += formatted.count(" ") + 1
  print("-"*50)
  print("-"*50)