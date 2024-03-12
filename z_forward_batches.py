from mistral.mod_model import ModelArgs
from mistral.tokenizer import Tokenizer
from mistral.cache import RotatingBufferCache
from main import generate, sample
from pathlib import Path
from wrapper.wrapper_MEEstral import MEEstral
import torch

path_weights = "./model_weights/mistral-7B-v0.1"
max_tokens = 25
chunk_size = 50

model = MEEstral(None, path_weights = path_weights, max_batch_size=10, device = "cuda")
tokenizer = Tokenizer(path_weights + "/tokenizer.model")

inputs = ["The color of the sky is"]
inputs_batch = ["Hola me llamo Miguel","Hola me llamo Iván"]

tokens = tokenizer._model.encode(inputs)
seqlens = [len(x) for x in tokens]
max_len = max(seqlens)

print(tokens)

# prepare cache
cache_window = max(seqlens) + max_tokens
if cache_window > model.args.sliding_window:
    cache_window = model.args.sliding_window

cache = RotatingBufferCache(
    model.n_local_layers,
    model.args.max_batch_size,
    cache_window,
    model.args.n_kv_heads,
    model.args.head_dim
)
cache.to(device=model.device, dtype=model.dtype)
cache.reset()

# bookkeeping
logprobs = [[] for _ in range(len(inputs_batch))]
last_token_prelogits = None

# chunking
if chunk_size is None:
    chunk_size = max_len

for s in range(0, max_len, chunk_size):
    prompt_chunks = [p[s:s+chunk_size] for p in tokens]
    assert all(len(p) > 0 for p in prompt_chunks)
    prelogits = model.forward(
        torch.tensor(sum(prompt_chunks, []), device=model.device, dtype=torch.long),
        seqlens=[len(p) for p in prompt_chunks],
        cache=cache
    )
    logits = torch.log_softmax(prelogits, dim=-1)

    if last_token_prelogits is not None:
        # Pass > 1
        last_token_logits = torch.log_softmax(last_token_prelogits, dim=-1)
        for i_seq in range(len(tokens)):
            logprobs[i_seq].append(last_token_logits[i_seq, prompt_chunks[i_seq][0]].item())

    offset = 0
    for i_seq, sequence in enumerate(prompt_chunks):
        logprobs[i_seq].extend([logits[offset + i, sequence[i + 1]].item() for i in range(len(sequence) - 1)])
        offset += len(sequence)

    last_token_prelogits = prelogits.index_select(0, torch.tensor([len(p) for p in prompt_chunks], device=prelogits.device).cumsum(dim=0) - 1)


# decode
generated_tokens = []
assert last_token_prelogits is not None
for i_token in range(max_tokens):
    next_token = sample(last_token_prelogits, temperature=0.01, top_p=0.8)

    last_token_logits = torch.log_softmax(last_token_prelogits, dim=-1)
    for i in range(len(seqlens)):
        logprobs[i].append(last_token_logits[i, next_token[i]].item())

    generated_tokens.append(next_token[:, None])
    last_token_prelogits = model.forward(next_token, seqlens=[1] * len(seqlens), cache=cache)
    # assert last_token_prelogits.shape == (B, V)

generated_words = []
if generated_tokens:
    generated_tokens = torch.cat(generated_tokens, 1)
    for i, x in enumerate(tokens):
        generated_words.append(tokenizer.decode(x + generated_tokens[i].tolist()))


print(logits)
print(generated_words)

# print(tokenizer.model._decode(generated_tokens))