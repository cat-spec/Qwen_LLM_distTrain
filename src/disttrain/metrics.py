import csv
from pathlib import Path

def metrics(csv_file:Path,args,file):
    #接受数据写入数据
    append=('a' if csv_file.exists() and args.resume==1 else 'w')
    csv_f=csv_file.open(mode=append,newline='')
    writer=csv.DictWriter(csv_f,fieldnames=file)
    if append=='w':
        writer.writeheader()
    return writer,csv_f