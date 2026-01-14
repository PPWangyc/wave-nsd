#!/usr/bin/env python3

"""
learning-from-brain

Training of models on given data. See get_args() for 
details on command line arguments.

To train a model, multiple core components from ..src/ 
are invoked:

src/batcher: Building PyTorch dataloaders for given data.
src/embedder: Embedding of inputs into embedding space, 
    training-style specific addition of training tokens
    and masking, and computation of training-style specific 
    losses.
    Valid training styles:
        - CSM (Causal Sequence Modeling)
"""

import os
import argparse
from typing import Dict
import json
from batcher import make_batcher
from trainer.trainer import Trainer
from model.fvl import FVL_NSD, compute_loss
import wandb
import torch
from model import make_model
from util.utils import (get_config,
                        set_seed, 
                        get_vis_mask, 
                        )
from dataset.dataset import NsdDataset

if __name__ == '__main__':
     
    """Model training according to config.
        -> see get_args() below for all command 
        line arguments.
    """
    
    config = get_config()

    if config['wandb']:
        wandb.init(project="WAVE", name="FVL_NSD", config=config)

    if config['set_seed']:
        set_seed(config['seed'])

    if config['do_train']:
        os.makedirs(
            config["log_dir"],
            exist_ok=True
        )

        resume_path = str(config["resume_from"]) if config["resume_from"] is not None else None
        
        config_filepath = os.path.join(
            config["log_dir"],
            'train_config.json'
        )
            
        with open(config_filepath, 'w') as f:
            json.dump(config, f, indent=2)

        config["resume_from"] = None

    assert config["training_style"] in {
        'CSM',
    }, f'{config["training_style"]} is not supported.'
    
    assert config["architecture"] in {
        'GPT',
    }, f'{config["architecture"]} is not supported.'

    train_dataset = NsdDataset(config['avg_data_dir'], subject_id=config['subject_id'], mode='train', transform=None)
    validation_dataset = NsdDataset(config['avg_data_dir'], subject_id=config['subject_id'], mode='val', transform=None)

    def model_init(params: Dict=None):
        model_config = dict(config)

        if params is not None:
            model_config |= params

        return make_model(model_config)

    model = model_init()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if config["vis_mask_json_path"] != None and config["vis_mask_json_path"] != 'None':
        vis_mask = get_vis_mask(config["vis_mask_json_path"], batch_size=config["per_device_training_batch_size"],
                                device=device)
    else:
        vis_mask = None

    if config['voxel_encoder_path'] != None:
        print("Loading voxel encoder from {}".format(config['voxel_encoder_path']))
        voxel_model_metafile = torch.load(config['voxel_encoder_path'], map_location='cpu')
    else:
        voxel_model_metafile = None
    voxel_model_metafile = None
    model = FVL_NSD(
        model, 
        fmri_resume_from=config["pretrained_model"], 
        train_mode=config['train_mode'],
        is_prompt=config['is_prompt'], 
        norm_embs=config['norm_embs'], 
        voxel_model_metafile=voxel_model_metafile,
        sparse_ratio=config['sparse_ratio'], 
        device=device, 
        fp16=config['fp16'], 
        mask_vis=vis_mask).to(device)
    print("Using {} device".format(device))

    # convert dict to argparse.Namespace
    train_config = {
        "wandb": config["wandb"],
        "train_batch_size": config["per_device_training_batch_size"],
        "eval_batch_size": config["per_device_validation_batch_size"],
        "device": device,
        "learning_rate": config["learning_rate"],
        "optimizer": "Adam",
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "adam_epsilon": 1e-8,
        "num_workers": 4,
        "weight_decay": config["weight_decay"],
        "l1_lambda": config["l1_lambda"],
        "seed": 42,
        "log_steps": config["log_every_n_steps"],
        "train_style": "step",
        "eval_steps": config["eval_every_n_steps"],
        "num_steps": len(train_dataset) // config["per_device_training_batch_size"] * 3 * config["training_epochs"],
        "log_dir": config["log_dir"],
        "save_steps": 10000,
        "scheduler": config["scheduler"],
        "scheduler_warmup_ratio": config["warmup_ratio"],
        "scheduler_warmup_steps": 0,
        "scheduler_last_epoch": -1,
        "scheduler_step_size": config["scheduler_step_size"],
        "scheduler_gamma": config['scheduler_gamma'],
        "only_visual": config["only_visual"],
        "train_mode": config["train_mode"],
    }
    trainer_args = argparse.Namespace(**train_config)
    trainer = Trainer(
        model=model,
        args=trainer_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        compute_loss=compute_loss
    )
    trainer.train()