[**🇨🇳中文**](./README_ch.md) | [**🌐English**](./README.md) |

# Recurrent Context Compression: Efficiently Expanding the Context Window of LLM
[[paper](https://arxiv.org/abs/2406.06110)]


## TL;DR
This work introduces a method called Recurrent Context Compression (RCC), designed to efficiently expand the context window length of LLMs within constrained storage space. We have released a model based pythia-1.4b with 32x context compression. Specifically, this model can compress the input text to 1/32 of its original size, significantly reducing memory consumption for long sequences. 

Due to the limitations of the fine-tuning dataset, the model currently only supports QA tasks



## News

- [2024/6/9] We released the RCC-Pythia-1.4b [model](https://huggingface.co/fcyp) and the [paper](https://arxiv.org/abs/2406.06110)

## Model Overview
RCC employs an encoder-decoder framework, with both the encoder and decoder weights initialized from a large language model. The trained encoder compresses fixed-length context information into a more compact form, applicable to both instructions and regular text. When context exceeds the fixed length, the encoder performs cyclic compression and concatenates all compressed feature vectors. The decoder uses these compressed vectors as input for final text generation.

We also addressed the issue of poor model performance when both instructions and context are compressed in downstream tasks, proposing an instruction reconstruction method to mitigate this. For more details, please refer to our paper.

**The structure of the encoder and decoder in RCC layer i.**
<img src=figures/model_structure.png alt="" width="800">


**Memory Consumption of Different Models with Increasing Length. Left: Pythia-1.4b, Right: RCC model using Pythia-1.4b for both encoder and decoder. Both models utilize FlashAttention-2**

<img src=figures/memory_size.png alt="" width="800">



## model download

| Model                   |            Type             |       Data       |         Required Original Model<sup>[1]</sup>          | Size<sup>[2]</sup> |                 Download Links<sup>[3]</sup>                 |
| :---------------------- | :-------------------------: | :--------------: | :----------------------------------------------------: | :----------------: | :----------------------------------------------------------: |
| RCC_Ins_Reconstruction  | QA model & Support instruction reconstruction | [Pile](https://huggingface.co/datasets/EleutherAI/pile)(about 10G),[Pwc](https://huggingface.co/datasets/sggetao/PwC),[hotpot_qa](https://huggingface.co/datasets/hotpotqa/hotpot_qa)  |      [pythia-1.4B](https://huggingface.co/EleutherAI/pythia-1.4b)      |        2.8B        |  <br/>[[🤗HF]](https://huggingface.co/fcyp/RCC_Ins_Reconstruction) |

## Usage

### environment
### environment
```bash
transformers==4.40.2
safetensors==0.4.1
torch==2.1.2
flash-attn==2.5.7
```

### Model use Cases

```bash
import torch
from transformers import AutoTokenizer
from  model.modeling_mcp2 import  GPTNeoXForCausalLM
from  transformers import GPTNeoXForCausalLM as GPTNeoXForCausalLM2
from safetensors.torch import load_file,save_file
import os
import json
from chat_example import load_and_merge_safetensors



#RCC-pythia, Example Usage
path = "EleutherAI/pythia-1.4b"
tokenizer = AutoTokenizer.from_pretrained(path)
model = GPTNeoXForCausalLM.from_pretrained(path,_attn_implementation="flash_attention_2",torch_dtype=torch.bfloat16).cuda()
model.gpt_neox.rcc_encoder = GPTNeoXForCausalLM2.from_pretrained(path,_attn_implementation="flash_attention_2",torch_dtype=torch.bfloat16).cuda()
#The length of each input to the decoder
model.gpt_neox.rcc_encoder_length = 2048

#The compression ratio is fixed to 32
model.gpt_neox.mem_len = 32

#Weights need to be downloaded
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
"""
Where does the cat sit?<|im_end|>
<|im_start|>assistant
The cat sits on the sofa.
"""

```



### Run more cases

```bash
CUDA_VISIBLE_DEVICES=0 python test.py
```





## FAQ

1. **How long can RCC extend the base model？**
The final theoretical sequence length is determined by the product of the decoder’s compression rate and the encoder’s window length. For instance, if the compression rate is set to 32 and the encoder is designed to handle 2048 tokens, the resulting theoretical length would be 32 times 2048.

2. **How effective is RCC so far?**
Due to the length constraints of the fine-tuning dataset, when performing inference with sequences longer than the length of the fine-tuning dataset, the model’s performance may degrade.


## TODOs
We will also continue to train and release RCC models based on stronger open-source models, including the qwen2 series, llama3 series, and mistral series.

**Please stay tuned for our updates.**

- [x] Release RCC-Pythia-1.4b.



## Motivation
Our encoder design is inspired by the Mamba-based LLM. Mamba is essentially a state space model, similar to an RNN. In Mamba, the current token only needs to access the state vector from the previous timestep to complete the current inference step. However, as the context length increases, the performance of Mamba deteriorates. This indicates that the state vector at each timestep in Mamba can store only a limited length of historical context information. Therefore, we propose a compromise: for long sequences, we can divide them into fixed-length short sequences and iteratively compress each short sequence into a state vector. We concatenate the state vectors of each short sequence as the historical state information during inference. This approach maximizes the retention of complete historical information while leveraging the model’s compression capabilities to save memory. In this paper, we use compression rate to reflect the maximum context length that a state vector at each timestep can store. Our experiments show that Transformers also have this capability because a Transformer can be viewed as a special state space model or RNN. 
For the decoder, we use a Transformer, allowing access to compressed vectors at any position.




## Citation

If you find RCC useful or relevant to your project and research, please kindly cite our paper:

```bibtex
@misc{huang2024recurrent,
      title={Recurrent Context Compression: Efficiently Expanding the Context Window of LLM}, 
      author={Chensen Huang and Guibo Zhu and Xuepeng Wang and Yifei Luo and Guojing Ge and Haoran Chen and Dong Yi and Jinqiao Wang},
      year={2024},
      eprint={2406.06110},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```