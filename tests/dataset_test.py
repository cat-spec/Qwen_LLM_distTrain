from pathlib import Path
from datasets import load_dataset
from torch.cuda import device

from src.disttrain import Mydataset,collate
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
#加载数据
path=Path(__file__).parent.parent/'src/disttrain/data'
dataset=load_dataset('json',data_dir=str(path),split='train')
data=dataset
dataset=dataset.select(range(10))
tokenizer_dir=Path(__file__).parent.parent/'src/disttrain/model'

tokenizer=AutoTokenizer.from_pretrained(str(tokenizer_dir))
print(f"Dataset{dataset}")
#自定义dataset类
dataset=Mydataset(tokenizer=tokenizer,total_sample=dataset.num_rows,dataset=dataset)

for index,item in enumerate(dataset):
    value=tokenizer(data[index]['output'])['input_ids']


    input_ids=item["input_ids"]
    label_ids=item["label_ids"]

    if len(input_ids)!=len(label_ids):
        raise ValueError(f"input_ids和label_ids长度必须一致input{len(input_ids)},{len(label_ids)}")
    if type(input_ids[0])!=int:
        raise ValueError(f'类型错误{type(input_ids[0])}')
    #正确性检验
    value_num=sum(1 if l!=-100 else 0 for l in label_ids)
    #eos所以+1
    if len(value)+1!=value_num:
        raise ValueError(f"逻辑错误，有效token数量不对")
dataloader=DataLoader(dataset,collate_fn=lambda batch:collate(batch,token_pad=tokenizer.pad_token_id),batch_size=2)
for index,item in enumerate(dataloader):
    batch={k:v.to(device) for k,v in item.items()}
print('验证通过')
#{'input_ids': [20002, 5122, 89012, 22382, 46944, 10],'label_ids': [-100, -100, -100, -100, -100]}
# print(dataset[0])
