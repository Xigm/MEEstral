from wrapper.wrapper_MEEstral import MEEstral
from mistral.model import ModelArgs
from mistral.tokenizer import Tokenizer
from main import generate
from pathlib import Path


path_weights = "model_weights\mistral-7B-v0.1"

model = MEEstral(None, path_weights = path_weights, max_batch_size=10, device = "cuda")
tokenizer = Tokenizer(path_weights + "/tokenizer.model")

prompt = ["Hola me llamo Miguel","Hola me llamo Iván y soy extremadamente"]
out, logprobs = generate(prompt, model, tokenizer, chunk_size=3, max_tokens = 25,  temperature = 1)

print(out)
print(logprobs)