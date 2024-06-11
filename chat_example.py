import torch
from transformers import AutoTokenizer#,Qwen2ForCausalLM
from  model.modeling_mcp2 import  GPTNeoXForCausalLM
from  transformers import GPTNeoXForCausalLM as GPTNeoXForCausalLM2
from safetensors.torch import load_file,save_file
import os
import json


#Load model checkpoints
def load_and_merge_safetensors(directory_path):
    model_params_paths = [
        os.path.join(directory_path, file)
        for file in os.listdir(directory_path)
        if file.endswith('.safetensors')
    ]

    if not model_params_paths:
        raise ValueError("No .safetensors files found in the specified directory.")

    merged_state_dict = {}
    for path in model_params_paths:
        state_dict = load_file(path)
        for key, value in state_dict.items():
            # Replace 'mamba' with 'rcc_encoder' in key names
            #key = key.replace('mamba', 'rcc_encoder')
            if key in merged_state_dict:
                merged_state_dict[key] = torch.cat((merged_state_dict[key], value), dim=0)
            else:
                merged_state_dict[key] = value

    return merged_state_dict




#model.save_pretrained("ckp/RCC_Ins_Reconstruction")



if __name__ == "__main__":


    #RCC-pythia, Example Usage
    tokenizer = AutoTokenizer.from_pretrained("../base_model/pythia-1.4b")
    path = "../base_model/pythia-1.4b"
    model = GPTNeoXForCausalLM.from_pretrained(path,_attn_implementation="flash_attention_2",torch_dtype=torch.bfloat16).cuda()
    model.gpt_neox.rcc_encoder = GPTNeoXForCausalLM2.from_pretrained(path,_attn_implementation="flash_attention_2",torch_dtype=torch.bfloat16).cuda()
    model.gpt_neox.rcc_encoder_length = 2048
    model.gpt_neox.mem_len = 32
    directory_path = "ckp/RCC_Ins_Reconstruction"
    merged_state_dict = load_and_merge_safetensors(directory_path)
    model.load_state_dict(merged_state_dict)


    input_context = "The cat sat on the sofa."+ "The dog sat on the stool."*1000
    input_context = input_context+"\n\n"+"Where does the cat sit?"
    compressed_id = tokenizer(input_context, return_tensors="pt")["input_ids"].cuda()
    #A chat example of instruction reconstruction
    prompt_fixed = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
    #Chat example of manual instruction input
    #prompt_fixed = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nWhere is the cat sitting?"
    prompt_ids = tokenizer(prompt_fixed ,return_tensors="pt")["input_ids"].cuda() 
    #Note that you must add this code or you may run out of memory
    model.eval()
    with torch.no_grad():
        generate_id = model.generate(compressed_id=compressed_id,prompt_ids=prompt_ids,max_new_tokens=100,eos_token=0)
    pred = tokenizer.decode(generate_id[0], skip_special_tokens=True)
    #pred = pred.split("assistant\n")[-1]
    print(pred)

    with open("data/PWC_eval_data.jsonl","r") as f:
        datas = f.readlines()
        datas = [json.loads(i) for i in datas]
        
    count = 0
    for example in datas:
        count = count+1
        input_context = example['input']
        input_context = input_context+"\n\n"+example['prompt']
                        
        compressed_id = tokenizer(input_context, return_tensors="pt")["input_ids"].cuda()
        #A chat example of instruction reconstruction
        prompt_fixed = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n"
        #Chat example of manual instruction input
        #prompt_fixed = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nWhere is the cat sitting?"
        prompt_ids = tokenizer(prompt_fixed ,return_tensors="pt")["input_ids"].cuda() 
        #Note that you must add this code or you may run out of memory
        model.eval()
        with torch.no_grad():
            generate_id = model.generate(compressed_id=compressed_id,prompt_ids=prompt_ids,max_new_tokens=100,eos_token=0)
        pred = tokenizer.decode(generate_id[0], skip_special_tokens=True)
        #print(pred)
        pred = pred.split("assistant\n")[-1]
        print(f"--------------{count}------------------\n")
        print("pred:\n",pred)
        print("\n")
        print("answer:\n",example['answer'])
        print(f"--------------{count}--------------------\n")
        