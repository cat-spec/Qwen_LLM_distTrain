file=['step','loss','lr','elapsed_time','computer_time','token_per_s','peak_alloc_GB','peak_res_GB']
import csv
from pathlib import Path

def metrics(csv_file:Path,args):
    #接受数据写入数据
    append=('a' if csv_file.exists() and args.resume==1 else 'w')
    csv_f=csv_file.open(mode=append,newline='')
    writer=csv.DictWriter(csv_f,fieldnames=file)
    if append=='w':
        writer.writeheader()
    return writer,csv_f