
import os
import sys
import datetime
from dialz import SteeringModel, SteeringVector
from utils import bbq_axes, load_and_tokenize_contrastive

if len(sys.argv) > 1:
    model_name = sys.argv[1]
else:
    raise ValueError("Model name must be provided as a command-line argument.")

# Map model names to short names
model_short_names = {
    "Qwen/Qwen2.5-7B-Instruct": "qwen",
    "meta-llama/Llama-3.1-8B-Instruct": "llama",
    "mistralai/Mistral-7B-Instruct-v0.1": "mistral",
}

model_short_name = model_short_names.get(model_name)
if not model_short_name:
    raise ValueError(f"Unknown model name: {model_name}")


dirs = {
    "train": f"../vectors/{model_short_name}/train",
    "train+prompt": f"../vectors/{model_short_name}/train+prompt",
    "generate_ss": f"../vectors/{model_short_name}/generate_ss",
    "generate_qa": f"../vectors/{model_short_name}/generate_qa",
}

for d in dirs.values():
    os.makedirs(d, exist_ok=True)

model = SteeringModel(model_name, [5]) # Second element is arbritary as we're not generating yet

for axis in bbq_axes:

    print(f"Creating 2 vectors for {axis} at:", datetime.datetime.now())

    ## BBQ Type Datasets - Ambig + Disambig
    path  = f"../data/bbq_train/{axis}_train.json"
    train_dataset = load_and_tokenize_contrastive(model_name, path)
    vector = SteeringVector.train(model, train_dataset)
    vector.export_gguf(os.path.join(dirs['train'], f"{axis}.gguf"))
 
    ## BBQ + Prompt Type Datasets - Ambig + Disambig
    path  = f"../data/bbq_train/{axis}_train.json"
    train_dataset = load_and_tokenize_contrastive(model_name, path, prompt=f"Consider the bias related to {axis} in the following. ")
    vector = SteeringVector.train(model, train_dataset)
    vector.export_gguf(os.path.join(dirs['train+prompt'], f"{axis}.gguf"))
