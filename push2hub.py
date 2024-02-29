from dataclasses import dataclass
import simple_parsing

import torch
import wandb
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class Config:
    repo_id: str
    model_path: str = None
    artifact_name: str = 'llm_surgery/gemma-zephyr/gemma-2b:v1'
    torch_dtype: str = 'bfloat16'


# kwargs = {
#     "finetuned_from":
#     "dataset": list(data_args.dataset_mixer.keys()),
#     "dataset_tags": list(data_args.dataset_mixer.keys()),
#     "tags": ["alignment-handbook"],
# }

config = simple_parsing.parse(Config)

if config.artifact_name is not None:
    run = wandb.init()
    artifact = run.use_artifact(config.artifact_name, type='model')
    config.model_path = artifact.download()

model = AutoModelForCausalLM.from_pretrained(config.model_path, torch_dtype=getattr(torch, config.torch_dtype))
tokenizer = AutoTokenizer.from_pretrained(config.model_path)

model.push_to_hub(config.repo_id)
tokenizer.push_to_hub(config.repo_id)