from torch.utils.data import Dataset

import torch

class Mydataset(Dataset):
    def __init__(self,tokenizer,total_sample,dataset,max_length,min_prompt_length):
        super().__init__()
        self.total_sample=total_sample
        self.tokenizer=tokenizer
        self.dataset=dataset
        self.max_length=max_length
        self.min_prompt_length=min_prompt_length
    def __len__(self):
        return self.total_sample
    def my_split(self,input_ids,label_ids):
        #eos长度
        token_budget=self.max_length-1

        prompt_budget=min(len(input_ids),self.min_prompt_length)
        label_budget=token_budget-prompt_budget
        label_ids=label_ids[:label_budget]
        prompt_budget=token_budget-len(label_ids)
        input_ids=input_ids[:prompt_budget]
        return input_ids,label_ids
    def __getitem__(self,index):
        #instruction  output

        prompt=f'用户：{self.dataset[index]["instruction"]}\n助手：'

        #把不需要学习的部分为-100，
        input_ids=self.tokenizer(prompt,add_special_tokens=False)['input_ids']
        #答案部分
        answer=self.tokenizer(self.dataset[index]["output"],add_special_tokens=False)['input_ids']
        input_ids,answer=self.my_split(input_ids,answer)


        inp_ids=input_ids+answer+[self.tokenizer.eos_token_id]
        label_ids=[-100]*len(input_ids)+answer+[self.tokenizer.eos_token_id]

        return {
            'input_ids':inp_ids,
            'label_ids':label_ids,
        }

