import json
import sys
from pathlib import Path
import argparse
import logging
from dataclasses import asdict

import torch
from torch.utils.data import DataLoader

from transformers import AutoConfig,AutoTokenizer,get_cosine_schedule_with_warmup
from src.disttrain import Myconfig,save_model,CollatorFn,Mydataset,metrics,init_model,signal_train
from datasets import load_dataset
#工程化初始参数
def parse_args():
    parser = argparse.ArgumentParser()
    #模型基本参数
    parser.add_argument('--model-path',type=str,default='./src/disttrain/model')
    #训练参数
    parser.add_argument('--lr',type=float,default=0.001)
    parser.add_argument('--batch-size',type=int,default=2)
    parser.add_argument('--epochs',type=int,default=10)
    parser.add_argument('--seed',type=int,default=42)
    #基本输出参数
    parser.add_argument('--output-dir',type=str,default='./output/signal')
    parser.add_argument('--resume',action='store_true',default=False)
    #设备信息
    parser.add_argument('--device',type=str,default='cuda:0')
    parser.add_argument(
        "--precision",
        type=str,
        choices=["fp32", "bf16"],
        default="bf16",
        help="训练精度，三选一",
    )
    if torch.cuda.is_available()<=0:
        raise ValueError("无设备可以用")
    return parser.parse_args()
#初始化日志
def init_logger(log_path:Path):
    logger=logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(lineno)s - %(message)s'
    )
    fh=logging.FileHandler(log_path/'signal.log',mode='w',encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger

#初始化数据
#主函数
def main():
    args=parse_args()
    #路径处理
    #./output/signal
    output=Path(args.output_dir)
    checkpoint_dir=output/'checkpoints'
    log_dir=output/'logs'
    data_dir=Path('./src/disttrain/data')
    checkpoint_dir.mkdir(exist_ok=True,parents=True)
    log_dir.mkdir(exist_ok=True,parents=True)
    #初始化
    #./src/disttrain/model
    # config=Myconfig()
    model_config=AutoConfig.from_pretrained(args.model_path)
    tokenizer=AutoTokenizer.from_pretrained(args.model_path)
    model=init_model(args.model_path,args)
    dataset=load_dataset('json',data_dir=str(data_dir),split='train')
    dataset=dataset.select(range(100))
    dataset=Mydataset(tokenizer,len(dataset),dataset)
    #./output/signal/logs/signal.log
    logger=init_logger(log_dir)
    #./output/signal/metrics.csv
    writer,f=metrics(output/'metrics.csv',args)
    logger.debug(f'model_config:{model_config.model_type},data{len(dataset)},resum{args.resume}')
    config_dist = asdict(Myconfig())
    temp_dist = {}
    # 初始化自己的配置类
    # config_dist={key:value for key,value in model_config}
    for key, value in model_config.to_dict().items():
        if key in config_dist.keys():
            temp_dist[key] = value
    my_config = Myconfig(**temp_dist)

     #设备初始化
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.set_device(0)
    device=torch.device(0)
    logger.debug(f'config设置{my_config}')
    logger.debug('初始化结束')
    #记录元信息
    meta_dist=asdict(my_config)
    with open(output/'meta.json','w') as f_json:
        json.dump({
            'model':{**meta_dist},
            'gpu_name':torch.cuda.get_device_name(0),
            'pytorch_version':torch.__version__,
            'python_version':sys.version,
        },f_json,indent=2)
    #数据加载器
    dataloader = DataLoader(dataset,collate_fn=CollatorFn(tokenizer.pad_token_id),batch_size=args.batch_size,shuffle=True,pin_memory=True,prefetch_factor=2,num_workers=2)
    #开启训练
    optimizer=torch.optim.Adam(model.parameters(),lr=args.lr)
    num_epochs = args.epochs
    total_steps = len(dataloader) * num_epochs
    scheduler=get_cosine_schedule_with_warmup(
    optimizer,num_warmup_steps=int(0.03 * total_steps),   # 前 3% 步数预热
    num_training_steps=total_steps)
    # 续训练
    resume_epoch=0
    if args.resume:
        load=torch.load(checkpoint_dir/'best.pt')
        model.load_state_dict(load['model'])
        optimizer.load_state_dict(load['optimizer'])
        scheduler.load_state_dict(load['scheduler'])
        resume_epoch = load['epoch']+1
    #开启循环训练
    signal_train(args,model,optimizer,scheduler,dataloader,writer,checkpoint_dir,device,resume_epoch)
    #保存参数
    save_model(model,optimizer,scheduler,checkpoint_dir/'last.pt',args.epochs)
if '__main__' == __name__:
    main()