import sys
import os
sys.path.append(os.path.join(sys.path[0], '../../EleutherAI_Eval_harness'))
sys.path.append(os.path.join(sys.path[0], '../../'))

from lm_eval.models.mistral_models_EE import Mistral_7b
from lm_eval import evaluate, simple_evaluate
from lm_eval.tasks import get_task_dict
from lm_eval.utils import make_table

# os.chdir(os.path.join(sys.path[0], './EE_Clean'))

from models.mistral.model import ModelArgs
from models.mistral.model_EE import Transformer
from models.mistral.tokenizer import Tokenizer
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer


import json
import torch


path_weights = "./weights/mistral/7B-v0.3"
max_length = 2048*2
max_gen_tokens = 64
device = "cuda"
batch_size = 1

# create your model (could be running finetuning with some custom modeling code)
if path_weights is not None:
    with open(path_weights + "/params.json", "r") as f:
        args = ModelArgs.from_dict(json.load(f))

args.max_batch_size = batch_size

print("Loading model...")

path = f"./weights/mistral"
path_weigths_EE = path + f"/EE_1_layers_middle_2_pos_15_19_23_27"
plot_intermediate_states = True
th_for_EE = 0.6
ee_pos = [int(p) for p in path_weigths_EE.split("_pos_")[-1].split("_")]
ee_pos = None


path = "./weights/mistral/7b-v0.3"
with open(path+ "/params.json") as f:
    args = ModelArgs(**dict(json.load(f)))
    args.lora.enable = False
    args.ee_pos = ee_pos
    model = Transformer(args)

print("Loading weights...")
model.from_pretrained(path + "/consolidated.safetensors")
n_layer = model.args.n_layers

# model.th = model.th * th_for_EE

# if ee_pos is not None:
#     for i in range(len(ee_pos)):
#         model.ee[i].load_state_dict(torch.load(f"{path_weigths_EE}/layer_{i}_EE"))

print("Sending to GPU...")
model.eval()
if device == "cuda":
    model.to(device = device, dtype = torch.bfloat16)

print("Loading tokenizer...")
tokenizer = Tokenizer(path_weights + "/tokenizer.model.v3")
# tokenizer = MistralTokenizer.v3().instruct_tokenizer

dataset = "mmlu"
n_shots = 5

# Create directories if they do not exist
output_dir = f"{path_weigths_EE}/results/{dataset}/baseline"
os.makedirs(output_dir, exist_ok=True)

# Define the directory and filenames
output_dir = f"{path_weigths_EE}/results/{dataset}/baseline"
results_file = f"{output_dir}/{n_shots}_results_list.json"
drops_file = f"{output_dir}/{n_shots}_layers_dropped.json"
os.makedirs(output_dir, exist_ok=True)

# Load previous progress if files exist
if os.path.exists(results_file) and os.path.exists(drops_file):
    with open(results_file, "r") as f:
        results_buffer = json.load(f)
    with open(drops_file, "r") as f:
        drops_buffer = json.load(f)
else:
    results_buffer = []
    drops_buffer = []

# Determine starting point based on the length of previously saved data
start_iter = len(results_buffer)
last_iter = 32
drops = torch.arange(start_iter, last_iter, 1)
for layers_dropped in drops:

    print(f"Layers dropped: {layers_dropped}")

    lm_obj = Mistral_7b(model=model,
                        tokenizer = tokenizer,
                        batch_size=batch_size,
                        max_length = max_length,
                        max_gen_tokens = max_gen_tokens,
                        temperature = 1.0,
                        top_k = None,
                        n_blocks = n_layer - layers_dropped,
                        device = device)

    # optional: the task_manager indexes tasks including ones
    # specified by the user through `include_path`.
    # task_manager = lm_eval.tasks.TaskManager(
    #     include_path="/path/to/custom/yaml"
    #     )

    # To get a task dict for `evaluate`
    # task_dict = get_task_dict(
    #     ["truthfulqa"]
    #     )

    print("Evaluating...")
    # results = evaluate(
    #     lm=lm_obj,
    #     task_dict=task_dict,
    #     # num_fewshot=3,
    # )


    # for triviaqa in need to generete text
    # also for truthfulqa
    # nq_open

    # triviaqa
    # coqa
    # generate_until
    dataset = "mmlu"
    results = simple_evaluate(
        model = lm_obj,
        tasks = [dataset],
        num_fewshot = n_shots,
    )


    results_buffer.append(results)
    drops_buffer.append(layers_dropped.item())

    print(make_table(results))

    # Append new result to JSON file each iteration
    with open(f"{output_dir}/{n_shots}_results_list.json", "w") as f:
        f.write(json.dumps(results_buffer))  # Append as new line for each result
    
    with open(f"{output_dir}/{n_shots}_layers_dropped.json", "w") as f:
        f.write(json.dumps(drops_buffer))  # Append as new line for each result



# # save results list, the exits done and the positions, if it does not exist, create it
# if not os.path.exists(path_weigths_EE + f"/results/" + dataset + "/baseline"):
#     os.makedirs(path_weigths_EE + f"/results/" + dataset + "/baseline")

# with open(path_weigths_EE + f"/results/"+dataset+ "/baseline" +"/"+ str(n_shots)+"_results_list.json", "w") as f:
#     json.dump(results_list, f)

# with open(path_weigths_EE + f"/results/"+dataset+ "/baseline"+ +"/"+ str(n_shots)+"_layers_dropped.json", "w") as f:
#     json.dump(drops.tolist(), f)
