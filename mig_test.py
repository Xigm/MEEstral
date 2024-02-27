import main

model_path = "model_weights\mistral-7B-v0.1"

main.interactive(model_path, max_tokens = 100, temperature = 0.7, instruct = False)