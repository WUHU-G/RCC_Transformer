# -*- coding: utf-8 -*-
"""
Created on Wed Aug  9 15:23:30 2023

@author: 18523
"""

import os
import sys
from typing import  Dict
import fire
import torch
import transformers
from torch.utils.data import Dataset
import numpy as np
import random
from transformers import AutoTokenizer,AutoConfig
from torch.utils.data import Dataset
from  model.modeling_mcp2 import  GPTNeoXForCausalLM
from transformers import GPTNeoXForCausalLM as GPTNeoXForCausalLM2

world_size = int(os.environ.get("WORLD_SIZE", 1))




def generate_sublists(lst, N, M, X):
    """
    Generate random sublists from a list that satisfy the given conditions.

    Parameters:
    - lst: The original list.
    - N: The total length of all sublists combined.
    - M: Minimum length of each sublist.
    - X: Maximum length of each sublist.

    Returns:
    - A list of sublists that satisfy the conditions.
    """
    # Check if it's possible to generate sublists with the given constraints
    if  N < M or N > len(lst) * X:
        return []

    # Initialize variables
    sublists = []
    total_length = 0

    while total_length < N:
        # Randomly choose a length for the next sublist
        sublist_length = random.randint(M, X)

        # Ensure the sublist length doesn't exceed the remaining length needed
        sublist_length = min(sublist_length, N - total_length)

        # Randomly choose a starting index for the next sublist
        start_index = random.randint(0, len(lst) - sublist_length)

        # Add the sublist to the list of sublists
        sublists.append(lst[start_index:start_index + sublist_length])

        # Update the total length of sublists
        total_length += sublist_length

    return sublists


class PretrainDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_dir: str, tokenizer: transformers.PreTrainedTokenizer, sequence_length=2048+512):#2048+512):
        super(PretrainDataset, self).__init__()
        self.all_ids = np.memmap(data_dir, dtype=np.uint16, mode='r')
        self.sequence_length = sequence_length
        self.arr_length = len(self.all_ids)
        
        
        self.mem_len = 64
        self.trench = 32
        self.mamba_input_len = self.trench * self.mem_len
        
        self.ckp_len = self.trench
        print("self.ckp_len:",self.ckp_len)
        self.sample_number = self.arr_length // self.sequence_length
        self.tokenizer = tokenizer
                
        
    def __len__(self):
        return self.sample_number 

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        #sample_index = random.randint(0, self.sample_number - 1)
        #batch_section_start_positon_idx
        input_ids = self.all_ids[i * self.sequence_length: (i + 1) * self.sequence_length].astype(np.int64)
        input_ids = torch.from_numpy(input_ids) 
        
        
        start_idx = random.randint(0,2048-512)
        #should_process = random.choice([True,False])

        #should_process = random.choice([True,False,False,False,False])
        #should_process = random.choice([True,True,True,True,True,True,True,True,True])
        should_process = True
        if should_process:
            input_ids[self.mamba_input_len :] = input_ids[start_idx:start_idx+512].clone()

        labels = input_ids[-512-self.ckp_len:].clone()
        start_idx2 = random.randint(10,20)
        if should_process:
            labels[:self.ckp_len+start_idx2] = -100
        else:
            labels[:self.ckp_len-1] = -100
        attention_mask = np.ones_like(labels)
        
        
        
        return dict(input_ids=input_ids, labels=labels,attention_mask = attention_mask)





def make_pretrain_data_module(tokenizer: transformers.PreTrainedTokenizer, train_data_path=None,val_data_path=None,cutoff_len=1024) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = PretrainDataset(tokenizer=tokenizer, data_dir=train_data_path,sequence_length=cutoff_len)
    eval_dataset = PretrainDataset(tokenizer=tokenizer, data_dir=val_data_path,sequence_length=cutoff_len)
    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset, data_collator=None)

def train(
    # model/data params
    data_path: str = "train_data/pretrain_data_pile/train.bin",
    val_data_path: str = "train_data/pretrain_data_pile/test.bin",
    output_dir: str = "output/pretrain_pile",
    save_steps: int = 2000,
    warmup_steps: int = 500,
    # training hyperparams
    micro_batch_size: int = 6,
    num_epochs: int = 1,
    learning_rate: float = 1e-4,
    cutoff_len: int = 2048+512,#,64*64+1920,#64*512+1536,#2048+512,#,#64*128+512,#64*512+1536,#2048+512,#,
    save_total_limit : int = 4,
):
    if int(os.environ.get("LOCAL_RANK", 0)) == 0:
        print(
            f"data_path: {data_path}\n"
            f"val_data_path: {val_data_path}\n"
            f"output_dir: {output_dir}\n"
            f"micro_batch_size: {micro_batch_size}\n"
            f"num_epochs: {num_epochs}\n"
            f"learning_rate: {learning_rate}\n"
            f"cutoff_len: {cutoff_len}\n"
        )


    os.environ["WANDB_DISABLED"]="true"
    enc = AutoTokenizer.from_pretrained("base_model/pythia-1.4b")
    enc.pad_token = enc.eos_token
    stop_id = enc.eos_token_id
    print(stop_id)



    gradient_accumulation_steps = 1
    print(gradient_accumulation_steps)
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    print("world_size",world_size)
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        #gradient_accumulation_steps = gradient_accumulation_steps // world_size
        #gradient_accumulation_steps = gradient_accumulation_steps*4
        
    print("len(tokenizer)",len(enc),gradient_accumulation_steps)    
    


    path = "EleutherAI/pythia-1.4b"
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = GPTNeoXForCausalLM.from_pretrained(path,_attn_implementation="flash_attention_2",torch_dtype=torch.bfloat16).cuda()
    model.gpt_neox.rcc_encoder = GPTNeoXForCausalLM2.from_pretrained(path,_attn_implementation="flash_attention_2",torch_dtype=torch.bfloat16).cuda()
    #The length of each input to the decoder
    model.gpt_neox.rcc_encoder_length = 2048
    #compression ratio
    model.gpt_neox.mem_len = 32
    
    #for param in model.gpt_neox.rcc_encoder.parameters():
     #   param.requires_grad = False

    # for param in model.parameters():
    #     param.requires_grad = False


    print(model)
    total = sum([param.nelement() for param in model.parameters()])
    print('Model {} : params num is {} M'.format(model._get_name(),total/1000000))


    data_module = make_pretrain_data_module(tokenizer=enc, train_data_path=data_path,val_data_path=val_data_path,cutoff_len=cutoff_len)
    
    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True
        print("ddpp-----------------------")

    
    trainer = transformers.Trainer(
        model=model,
        tokenizer=enc,
        
        args=transformers.TrainingArguments(
            bf16=True,
            per_device_train_batch_size=micro_batch_size,
            per_device_eval_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=warmup_steps,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            logging_steps=20,
            #optim="adamw_torch",
            evaluation_strategy="steps",
            save_strategy="steps",
            eval_steps=8000000,
            save_steps=save_steps,
            output_dir=output_dir,
            save_total_limit=save_total_limit,
            report_to="none",
            #deepspeed='configs/deepspeed_config_stage2_no_offload.json',
            #deepspeed='configs/deepspeed_config_stage3.json',
            #group_by_length = True,
            #deepspeed='configs/deepspeed_config_stage2.json',
            #logging_dir=output_dir,
            #load_best_model_at_end=True if val_set_size > 0 else False,
            #ddp_find_unused_parameters=False if ddp else None,
            ddp_find_unused_parameters=True,
            #dataloader_num_workers=1,
            #fsdp="full_shard auto_wrap",
            #fsdp_transformer_layer_cls_to_wrap='DecoderLayer',
        ),
        **data_module
    )

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)
    #trainer.train(resume_from_checkpoint = "checkpoint-21000")
    trainer.train()


if __name__ == "__main__":
    fire.Fire(train)
