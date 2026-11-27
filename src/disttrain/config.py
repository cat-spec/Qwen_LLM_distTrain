from dataclasses import dataclass

import torch


#融合模型参数，和我自身设定参数
@dataclass
class Myconfig:
    bos_token_id:int=0
    eos_token_id:int=0
    hidden_size:int=0
    torch_dtype:str='bfloat16'
    vocab_size:int=1024
    num_attention_heads:int=8
    max_length:int=512
    min_prompt_length:int=128
    num_hidden_layers:int=8
    num_key_value_heads:int=8
    rms_norm_eps:float=1e-6
    rope_theta:int=1000000
    def __post_init__(self):
        pass