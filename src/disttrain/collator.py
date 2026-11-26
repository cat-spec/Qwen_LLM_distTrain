import torch

class CollatorFn:
    def __init__(self,token_pad):
        self.token_pad=token_pad
    def __call__(self, batch):
        # batch ={input,lable}*2
        # 同组内最大的字符长度
        max_len = max([len(item['input_ids']) for item in batch])
        # 输入token 机器识别是否进行计算  是否进行loss计算
        input_ids, attention_ids, label_ids = [], [], []
        for item in batch:
            pad_len = max_len - len(item['input_ids'])
            attention_mask = torch.ones((len(item['input_ids']),), dtype=torch.long)

            input_pad = torch.full((pad_len,), self.token_pad, dtype=torch.long)
            attention_pad = torch.zeros((pad_len,), dtype=torch.long)
            label_pad = torch.full((pad_len,), -100, dtype=torch.long)

            item['input_ids'] = torch.cat([torch.tensor(item['input_ids']), input_pad], dim=0)
            item['label_ids'] = torch.cat([torch.tensor(item['label_ids']), label_pad], dim=0)
            attention_mask = torch.cat([attention_mask, attention_pad], dim=0)
            # print(item['label_ids'].shape)
            input_ids.append(item['input_ids'])
            attention_ids.append(attention_mask)
            label_ids.append(item['label_ids'])
        return {
            'input_ids': torch.stack(input_ids),
            'attention_ids': torch.stack(attention_ids),
            'label_ids': torch.stack(label_ids)
        }

