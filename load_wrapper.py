from wrapper_MEEstral import MEEstral
from mistral.model import ModelArgs
from mistral.tokenizer import Tokenizer
from main import generate
from pathlib import Path

# args = ModelArgs(    
#                     dim = 4096,
#                     n_layers = 32,
#                     head_dim = 128,
#                     hidden_dim = 14336, 
#                     n_heads = 32,
#                     n_kv_heads = 8,
#                     norm_eps = 1e-05,
#                     sliding_window = 4096,
#                     vocab_size = 32000,
#                     max_batch_size = 3,
#                 )

path_weights = "model_weights\mistral-7B-v0.1"

model = MEEstral(None, path_weights = path_weights, max_batch_size=10, device = "cuda")
tokenizer = Tokenizer(path_weights + "/tokenizer.model")

out = generate([["Hola me llamo Miguel"],["Hola me llamo Iván"]], model, tokenizer, max_tokens = 25,  temperature = 1)

print(out)