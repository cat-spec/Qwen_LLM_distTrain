import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    #模型基本参数
    parser.add_argument('--model-path',type=str,default='./src/disttrain/model')
    parser.add_argument('--max-length',type=int,default=512)
    parser.add_argument('--min-prompt-length',type=int,default=128)
    #训练参数
    parser.add_argument('--lr',type=float,default=0.001)
    parser.add_argument('--batch-size',type=int,default=2)
    parser.add_argument('--epochs',type=int,default=20)
    parser.add_argument('--seed',type=int,default=42)
    #基本输出参数
    parser.add_argument('--output-dir',type=str,default='./output/signal')
    parser.add_argument('--resume',action='store_true',default=True)
    #设备信息
    parser.add_argument('--device',type=str,default='cuda:0')
    parser.add_argument(
        "--precision",
        type=str,
        choices=["fp32", "bf16"],
        default="bf16",
        help="训练精度，三选一",
    )
    return parser.parse_args()