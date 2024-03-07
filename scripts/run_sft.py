#!/usr/bin/env python
# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Supervised fine-tuning script for decoder language models.
"""

import logging
import random
import sys

import datasets
import torch
import transformers
from transformers import set_seed, AutoModelForCausalLM

from alignment import (
    DataArguments,
    H4ArgumentParser,
    ModelArguments,
    SFTConfig,
    apply_chat_template,
    get_checkpoint,
    get_datasets,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
    get_tokenizer,
)
from trl import SFTTrainer
from accelerate import Accelerator

import wandb

logger = logging.getLogger(__name__)

def maybe_from_artifact(model_at_address: str):
    "Download the model from wandb if it's an artifact, otherwise return the path."
    try:
        if wandb.run:
            model_dir = wandb.use_artifact(model_at_address).download()
            logging.info(f"Downloading model from wandb: {model_at_address}")
        else:
            logging.info(f"Pulling without creating a run from wandb: {model_at_address}")
            api = wandb.Api()
            model_dir = api.artifact(model_at_address).download()
        return model_dir
    except:
        logging.info(f"Using model from local path: {model_at_address}")
        return model_at_address

def main():
    accelerator = Accelerator()

    parser = H4ArgumentParser((ModelArguments, DataArguments, SFTConfig))
    model_args, data_args, training_args = parser.parse()

    input_artifact_name = model_args.model_name_or_path.split("/")[-1].split(":")[0]
    output_artifact_name = input_artifact_name

    if accelerator.is_main_process:
        run_name = input_artifact_name + "_sft"
        group_name = input_artifact_name
        wandb.init(project="mistral_zephyr_v2", 
                   entity="llm_surgery", 
                   job_type="train-sft", 
                   group=group_name,
                   name=run_name,
                   tags=["align-sft"])
        

    
    with accelerator.main_process_first():
        model_args.model_name_or_path = maybe_from_artifact(model_args.model_name_or_path)

    # Set seed for reproducibility
    set_seed(training_args.seed)

    ###############
    # Setup logging
    ###############
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process a small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f" distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Model parameters {model_args}")
    logger.info(f"Data parameters {data_args}")
    logger.info(f"Training/evaluation parameters {training_args}")

    # Check for last checkpoint
    last_checkpoint = get_checkpoint(training_args)
    if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
        logger.info(f"Checkpoint detected, resuming training at {last_checkpoint=}.")

    ###############
    # Load datasets
    ###############
    raw_datasets = get_datasets(data_args, splits=data_args.dataset_splits)
    logger.info(
        f"Training on the following datasets and their proportions: {[split + ' : ' + str(dset.num_rows) for split, dset in raw_datasets.items()]}"
    )
    column_names = list(raw_datasets["train"].features)

    ################
    # Load tokenizer
    ################
    tokenizer = get_tokenizer(model_args, data_args)
    
    # quick fix
    from tokenizers import AddedToken
    tokenizer.add_tokens([AddedToken("<|im_start|>", rstrip=False, lstrip=False, normalized=False),
                          AddedToken("<|im_end|>", rstrip=False, lstrip=False, normalized=False)])

    #####################
    # Apply chat template
    #####################
    raw_datasets = raw_datasets.map(
        apply_chat_template,
        fn_kwargs={
            "tokenizer": tokenizer,
            "task": "sft",
            "auto_insert_empty_system_msg": data_args.auto_insert_empty_system_msg,
        },
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=column_names,
        desc="Applying chat template",
    )
    train_dataset = raw_datasets["train"]
    eval_dataset = raw_datasets["test"]

    with training_args.main_process_first(desc="Log a few random samples from the processed training set"):
        for index in random.sample(range(len(raw_datasets["train"])), 3):
            logger.info(f"Sample {index} of the processed training set:\n\n{raw_datasets['train'][index]['text']}")

    #######################
    # Load pretrained model
    #######################
    logger.info("*** Load pretrained model ***")
    torch_dtype = (
        model_args.torch_dtype if model_args.torch_dtype in ["auto", None] else getattr(torch, model_args.torch_dtype)
    )
    quantization_config = get_quantization_config(model_args)

    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        use_flash_attention_2=model_args.use_flash_attention_2,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map=get_kbit_device_map() if quantization_config is not None else None,
        quantization_config=quantization_config,
    )

    model = AutoModelForCausalLM.from_pretrained(        
        model_args.model_name_or_path,
        **model_kwargs,
    )
    # nearest 32x
    model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=32)

    # model = torch.compile(model)
    
    ########################
    # Initialize the Trainer
    ########################
    import os
    from pathlib import Path
    import json
    import torch.nn as nn
    from typing import Dict, Union, Any

    MAX_GRAD_NORM = 30.0
    class SpikeTrainer(SFTTrainer):
        # def __init__(self, *args, **kwargs):
        #     super().__init__(*args, **kwargs)
        #     if accelerator.is_main_process:
        #         self.table = wandb.Table(columns=["step", "loss", "grad_norm", "input_ids", "text"])

        def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
            # get loss from parent class
            loss = super().training_step(model, inputs)
            # check we have some metrics:
            if self.state.log_history:
                if "grad_norm" in self.state.log_history[-1]:
                    current_grad_norm = self.state.log_history[-1]["grad_norm"]
                    step_folder = f"logs/step{self.state.global_step}"
                    os.makedirs(step_folder, exist_ok=True)
                    gpu_rank = accelerator.process_index
                    if current_grad_norm > MAX_GRAD_NORM:
                        data = []
                        for input_id in inputs["input_ids"]:
                            data.append({
                                "Step": self.state.global_step,
                                "rank": gpu_rank, 
                                "Loss": loss.item(),
                                "Grad norm": current_grad_norm,
                                "Input id": input_id.tolist(),  # Assuming input_id is a tensor
                                "Decoded": tokenizer.decode(input_id)
                            })
                        file_name = f"{step_folder}/rank{gpu_rank}.json"
                        with open(file_name, 'w') as file:
                            json.dump(data, file, indent=4)
                        # log to wandb
                        # we have to dump the data to a file and then read it back to 
                        # gather all the data from different processes
                        accelerator.wait_for_everyone()
                        if accelerator.is_main_process:
                            table = wandb.Table(columns=['step', 'rank', 'loss', 'grad_norm', 'input_ids', 'decoded'])
                            for i in range(accelerator.num_processes):
                                file = Path(step_folder)/f"rank{i}.json"
                                logger.info(f"Logging spike from file: {file}")
                                one_gpu_batch = json.loads(file.read_text())
                                for one_seq in one_gpu_batch:
                                    table.add_data(*one_seq.values())
                            wandb.log({f"inputs_{self.state.global_step}":table})
            return loss
    
    trainer = SpikeTrainer(
        # model=model_args.model_name_or_path,
        # model_init_kwargs=model_kwargs,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=training_args.max_seq_length,
        tokenizer=tokenizer,
        packing=True,
        peft_config=get_peft_config(model_args),
        dataset_kwargs=training_args.dataset_kwargs,
    )

    ###############
    # Training loop
    ###############
    logger.info("*** Train ***")
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    ##################################
    # Save model and create model card
    ##################################
    logger.info("*** Save model ***")
    trainer.save_model(training_args.output_dir)
    logger.info(f"Model saved to {training_args.output_dir}")

    # Save everything else on main process
    kwargs = {
        "finetuned_from": model_args.model_name_or_path,
        "dataset": list(data_args.dataset_mixer.keys()),
        "dataset_tags": list(data_args.dataset_mixer.keys()),
        "tags": ["alignment-handbook"],
    }
    if trainer.accelerator.is_main_process:
        # trainer.create_model_card(**kwargs)
        # Restore k,v cache for fast inference
        trainer.model.config.use_cache = True
        trainer.model.config.save_pretrained(training_args.output_dir)

        # save model as artifact to wandb
        logger.info("Saving model as artifact to wandb")
        model_at = wandb.Artifact(
            name = output_artifact_name, 
            type="model",
            description="SFT model trained with alignment-handbook recipe",
            metadata=kwargs)
        model_at.add_dir(training_args.output_dir)
        wandb.log_artifact(model_at, aliases=["sft"])
    if training_args.push_to_hub is True:
        logger.info("Pushing to hub...")
        trainer.push_to_hub(**kwargs)

    logger.info("*** Training complete ***")


if __name__ == "__main__":
    main()
