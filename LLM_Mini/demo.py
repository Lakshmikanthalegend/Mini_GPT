import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from transformer_blocks import Block

print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")

corpus = [
    "hello friends how are you",
    "the tea is very hot",
    "my name is Aarohi",
    "the roads of Delhi are busy",
    "it is raining in Mumbai",
    "the train is late again",
    "i love eating samosas and drinking tea",
    "holi is my favorite festival",
    "diwali brings lights and sweets",
    "india won the cricket match"
]

##-----Tokenization-----##
# We will start by tokenizing the corpus. Tokenization is the process of converting text into a sequence of tokens (in this case, words).
corpus = [s + " <END>" for s in corpus]
text = " ".join(corpus)
print (text)

# We will create a set of unique words in the corpus, which will be our vocabulary. 
# The size of the vocabulary will determine the input and output dimensions of our model.
words = list(set(text.split()))
print(words)
print(len(words))

#Giving an index to each word in the corpus, we will create a dictionary that maps each word to a unique index.
word_to_idx = {w: i for i, w in enumerate(words)}
print(word_to_idx)

# We will also create a reverse dictionary that maps each index back to the corresponding word, 
# which will be useful for decoding the model's output later.
idx_to_word = {i: w for w, i in word_to_idx.items()}
print(idx_to_word)

# Here we cannot give words to the model, we need to convert them to indices first.
# So we will create a list of indices for each word in the corpus.
data = [word_to_idx[w] for w in text.split()]
print(data)

# Now we will create input and target sequences for training the model.
# The input sequence will be the list of indices except the last one, and the target sequence
# will be the list of indices except the first one. This way, the model will learn to predict the next word in the sequence.
block_size = 2
embedding_dim = 32
num_heads = 2
num_layers = 2
lr = 1e-3
epochs = 1500

