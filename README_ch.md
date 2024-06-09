[**🇨🇳中文**](./README_CH.md) | [**🌐English**](./README.md) |

# Recurrent Context Compression: Efficiently Expanding the Context Window of LLM
[[paper]()]


## TL;DR
这项工作介绍了一种名为循环上下文压缩（RCC）的方法，旨在有限的存储空间内高效地扩大LLM的上下文窗口长度，我们发布了一个拥有32x上下文压缩的模型，准确的来说这个模型可以将模型的输入文本压缩为1/32，能够在长序列中大幅减少显存消耗。

由于微调数据集的限制，该模型目前仅支持QA任务。



## News

- [2024/6/9] 我们发布了RCC-Pythia-1.4b模型和[论文]()

## Model Overview
RCC由一个编码器-解码器框架，编码器与解码器的权重都由大型语言模型初始化而来。经过训练的编码器能够将固定长度的上下文信息压缩成更紧凑的形式，指令和普通文本都可以当作上下文进行压缩。当上下文信息超过固定长度时，编码器执行循环压缩并将所有压缩特征向量连接起来。解码器利用压缩特征向量作为历史状态向量的输入，完成最终的文本生成任务。
同时我们研究了在下游任务中指令与上下文同时被压缩时模型的回答较差这一问题，并提出了指令重建的方法缓解了这一问题。更多信息可以在我们的论文中找到。

**The structure of the encoder and decoder in RCC layer i.**
<img src=figures/model_structure.png alt="" width="800">


**Memory Consumption of Different Models with Increasing Length. Left: Pythia-1.4b, Right: RCC model using Pythia-1.4b for both encoder and decoder. Both models utilize FlashAttention-2**

<img src=figures/memory_size.png alt="" width="800">





## Usage

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

1. **RCC在基础模型上扩展多长的上下文**
最终的理论序列长度由解码器的压缩率和编码器窗口长度的乘积决定。例如，如果压缩率设置为32，并且编码器设计为处理2048个token，那么得到的理论长度将是32乘以2048。

2. **目前RCC的性能如何？**
由于微调数据集的长度约束，当使用长于微调数据集长度的序列进行推理时，模型的性能可能会下降。


## TODOs
我们还将继续基于更强大的开源模型(包括qwen2系列、llama3系列和mistral系列)训练和发布RCC模型。

**请继续关注我们的最新消息。**

- [x] Updating inference code.
- [] Updating the training code


## Motivation
我们编码器的设计，受到了基于Mamba的LLM启发。 Mamba本质上是一个状态空间模型，它和RNN类似。在Mamba中，当前token只需获取上一个时刻的状态向量即可完成当前推理步骤，但是当上下文长度变长时，Mamba回答的效果会变差。这说明了Mamba中一个时刻的状态向量只能存储的一定长度的历史上下文信息。因此，我们提出了一个折中的方法，在处理长序列时，可以将它分割为固定长度的短序列，并循环的将每个短序列压缩到一个状态向量中。我们将每个短序列的状态向量拼接起来作为推理时期的历史状态信息，这样最大限度的保留了完整的历史信息，同时也能利用模型的压缩能力节约内存。 在本文中我们使用压缩率来反映一个时刻的状态向量存储的最大上下文长度，我们在实验中发现Transformer也具备这种能力，这是因为Transformer可以看作一个特殊状态空间模型或者RNN模型，最近的研究也表明注意力可以被视为RNN。
解码器我们采用Transformer，这样就能访问到任何位置上的压缩向量。





## Citation

如果您发现RCC对您的项目和研究有用或相关，请引用我们的论文:

```bibtex

```