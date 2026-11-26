from torch.utils.data import Dataset

import torch

class Mydataset(Dataset):
    def __init__(self,tokenizer,total_sample,dataset):
        super().__init__()
        self.total_sample=total_sample
        self.tokenizer=tokenizer
        self.dataset=dataset
    def __len__(self):
        return self.total_sample
    def __getitem__(self,index):
        #instruction  output

        prompt=f'用户：{self.dataset[index]["instruction"]}\n助手：'

        #把不需要学习的部分为-100，
        input_ids=self.tokenizer(prompt,add_special_tokens=False)['input_ids']
        #答案部分

        answer=self.tokenizer(self.dataset[index]["output"],add_special_tokens=False)['input_ids']
        inp_ids=input_ids+answer+[self.tokenizer.eos_token_id]
        label_ids=[-100]*len(input_ids)+answer+[self.tokenizer.eos_token_id]

        return {
            'input_ids':inp_ids,
            'label_ids':label_ids,
        }

