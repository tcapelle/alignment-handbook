import wandb

WANDB_PROJECT = "shearllama"
WANDB_ENTITY = "llm_surgery"


MODEL_FOLDER = "data/zephyr-7b-sft-full"
RUN_ID = "5q2vwifx"

wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, id=RUN_ID)

# save model as artifact to wandb
model_at = wandb.Artifact(
    name = f"model-{wandb.run.id}", 
    type="model",
    description="SFT model trained with alignment-handbook recipe",
    metadata= {"finetuned_from": "mistral_7b_12_layers_start:v0",
               "dataset":"HuggingFaceH4/ultrachat_200k"}
)
model_at.add_dir(MODEL_FOLDER)
wandb.log_artifact(model_at)