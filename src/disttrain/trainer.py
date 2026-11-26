#单gpu
import torch
import time
from .checkpoint import save_model

def signal_train(args,model,optimizer,scheduler,dataloader,writer,checkpoint_dir,device,resume_epoch:int=0,):
    best_loss = float('inf')
    try:
        for epoch in range(resume_epoch, args.epochs):
            model.train()
            total_loss = 0
            start_event=torch.cuda.Event(enable_timing=True)
            end_event=torch.cuda.Event(enable_timing=True)
            start=time.perf_counter()
            for step, batch in enumerate(dataloader):
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize(device)
                t0 = time.perf_counter()
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                optimizer.zero_grad(set_to_none=True)
                start_event.record()
                output = model(input_ids=batch['input_ids'],labels=batch['label_ids'],attention_mask=batch['attention_ids'])
                loss = output.loss
                total_loss += loss.item()
                loss.backward()
                optimizer.step()
                scheduler.step()
                end_event.record()
                torch.cuda.synchronize(device)
                computer_time=start_event.elapsed_time(end_event)
                total_token = batch['input_ids'].numel()
                elapsed_time = time.perf_counter() - t0
                # 记录训练
                csv_dist = {
                    'step': step,
                    'loss': loss.item(),
                    'lr': optimizer.param_groups[0]['lr'],
                    'elapsed_time': f'{round((elapsed_time), 4)}s',
                    'computer_time':computer_time,
                    'token_per_s': total_token *1000/ computer_time,
                    'peak_alloc_GB': torch.cuda.max_memory_allocated() / 1024 ** 3,
                    'peak_res_GB': torch.cuda.max_memory_reserved() / 1024 ** 3,
                }
                if step % 2 == 0:
                    writer.writerow(csv_dist)
            avg_loss = total_loss / len(dataloader)
            print(f'epoch{epoch},avg_loss,{avg_loss},epoch_time{time.perf_counter()-start}')
            if avg_loss < best_loss:
                best_loss = avg_loss
                save_model(model,optimizer,scheduler,checkpoint_dir/'best.pt',epoch)
    finally:
        pass

