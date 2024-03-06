from mistral.model import ModelArgs
from mistral.tokenizer import Tokenizer
from main import generate
from pathlib import Path
from wrapper.wrapper_MEEstral import MEEstral
import torch

path_weights = "model_weights\mistral-7B-v0.1"

model = MEEstral(None, path_weights = path_weights, max_batch_size=10, device = "cuda")
tokenizer = Tokenizer(path_weights + "/tokenizer.model")

inputs = ["Hola me llamo Miguel"]
inputs_batch = ["Hola me llamo Miguel","Hola me llamo Iván"]

tokens = tokenizer._model.encode(inputs_batch)
seqlens = [max(len(x) for x in tokens)]*len(tokens)

print(tokens)

# add padding to the batch so all inputs have the same length
max_len = max(seqlens)
for i in range(len(tokens)):
    tokens[i] = tokens[i] + [tokenizer.pad_id] * (max_len - len(tokens[i]))

# tokenize
tokens = torch.tensor(tokens, dtype=torch.long).to(model.device)


# prepare cache


# bookkeeping


# chunking


# forward pass


outputs = model(tokens[0], seqlens=[seqlens[0]], cache=None)

print(outputs)