from transformers import AutoModelForCausalLM
import torch

def init_model(model_path,args):
    if args=='bf16':
        dtype=torch.float16
    else:
        dtype=torch.float32
    model=AutoModelForCausalLM.from_pretrained(
        model_path,device_map=args.device,torch_dtype=dtype)
    return model
