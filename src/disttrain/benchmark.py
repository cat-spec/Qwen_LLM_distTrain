
from pathlib import Path
import time
from torch.profiler import profile,ProfilerActivity,record_function,schedule
import torch

def signal_benchmark(args, model, optimizer, scheduler, dataloader,
                     writer, benchmark_dir, device,
                     warmup_steps=10, bench_steps=50):
    benchmark_dir = Path(benchmark_dir)
    device = torch.device(device)

    # ---- 建议先开 cudnn benchmark（如果有 conv；transformer 影响小但无害）----
    torch.backends.cudnn.benchmark = True

    def train_step(batch):
        """一个完整 step，返回值用于记录"""
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        optimizer.zero_grad(set_to_none=True)

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        start_event.record()
        output = model(
            input_ids=batch['input_ids'],
            labels=batch['label_ids'],
            attention_mask=batch['attention_ids'],
        )
        loss = output.loss
        end_event.record()
        end_event.synchronize()
        forward_time_ms = start_event.elapsed_time(end_event)

        start_event.record()
        loss.backward()
        end_event.record()
        end_event.synchronize()
        backward_time_ms = start_event.elapsed_time(end_event)

        start_event.record()
        optimizer.step()
        end_event.record()
        end_event.synchronize()
        optimizer_time_ms = start_event.elapsed_time(end_event)

        scheduler_start = time.perf_counter()
        scheduler.step()
        scheduler_time_ms = (time.perf_counter() - scheduler_start) * 1000

        return {
            'loss': loss.detach(),        # 不在 step 内 item
            'forward_ms': forward_time_ms,
            'backward_ms': backward_time_ms,
            'optimizer_ms': optimizer_time_ms,
            'scheduler_ms': scheduler_time_ms,
            'batch': batch,
        }

    data_iter = iter(dataloader)
    steps_per_epoch = len(dataloader)

    def next_batch():
        nonlocal data_iter
        try:
            return next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            return next(data_iter)

    # ================= 阶段 1：warmup（不记录、不计时）=================
    model.train()
    print(f'[warmup] {warmup_steps} steps ...')
    for _ in range(warmup_steps):
        batch = next_batch()
        # 上面的写法不对，直接调用 train_step（内部会 backward）：
        _ = train_step(batch)

    # warmup 后同步，确保所有 kernel 完成，并清掉 warmup 期间累积的峰值
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.reset_accumulated_memory_stats(device)   # 可选，看分配器统计用

    mode='benchmark'
    # ================= 阶段 2：正式记录 =================
    if mode=='benchmark':
        print(f'[benchmark] {bench_steps} steps ...')
        bench_start = time.perf_counter()
        for i in range(bench_steps):
            step_start = time.perf_counter()

            # 只测 next(dataloader) 的等待时间
            t0 = time.perf_counter()
            batch = next_batch()
            dataloader_wait_ms = (time.perf_counter() - t0) * 1000

            # H2D 单独测（同步测完成时间）
            torch.cuda.synchronize(device)
            h2d_start = time.perf_counter()
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            torch.cuda.synchronize(device)
            h2d_ms = (time.perf_counter() - h2d_start) * 1000

            # forward / backward / optimizer / scheduler
            optimizer.zero_grad(set_to_none=True)
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            output = model(
                input_ids=batch['input_ids'],
                labels=batch['label_ids'],
                attention_mask=batch['attention_ids'],
            )
            loss = output.loss
            end_event.record()
            end_event.synchronize()
            forward_ms = start_event.elapsed_time(end_event)

            start_event.record()
            loss.backward()
            end_event.record()
            end_event.synchronize()
            backward_ms = start_event.elapsed_time(end_event)

            start_event.record()
            optimizer.step()
            end_event.record()
            end_event.synchronize()
            optimizer_ms = start_event.elapsed_time(end_event)

            sched_t0 = time.perf_counter()
            scheduler.step()
            scheduler_ms = (time.perf_counter() - sched_t0) * 1000

            # token 统计（.item() 会同步，放在 step 末尾，明确它在 step_wall 里）
            total_token = batch['input_ids'].numel()
            valid_token = batch['attention_ids'].sum().item()
            supervised_tokens = (batch["label_ids"][:, 1:] != -100).sum().item()
            padding_ratio = 1.0 - valid_token / total_token

            step_wall_s = time.perf_counter() - step_start
            csv_dist = {
                'step': i,
                'loss': loss.item(),
                'lr': optimizer.param_groups[0]['lr'],
                'step_wall_ms': step_wall_s * 1000,
                'dataloader_wait_ms': dataloader_wait_ms,
                'h2d_ms': h2d_ms,
                'forward_gpu_ms': forward_ms,
                'backward_gpu_ms': backward_ms,
                'optimizer_gpu_ms': optimizer_ms,
                'scheduler_cpu_ms': scheduler_ms,
                'input_tokens_per_s': total_token / step_wall_s,
                'valid_input_tokens_per_s': valid_token / step_wall_s,
                'supervised_tokens_per_s': supervised_tokens / step_wall_s,
                'padding_ratio': padding_ratio,
                'peak_alloc_GiB': torch.cuda.max_memory_allocated(device) / 1024 ** 3,
                'peak_reserved_GiB': torch.cuda.max_memory_reserved(device) / 1024 ** 3,
            }
            writer.writerow(csv_dist)
    else:
        print(f'[profile] {bench_steps} steps ...')
        bench_start = time.perf_counter()
        for i in range(bench_steps):
            step_start = time.perf_counter()

            # 只测 next(dataloader) 的等待时间
            t0 = time.perf_counter()
            batch = next_batch()
            dataloader_wait_ms = (time.perf_counter() - t0) * 1000

            # H2D 单独测（同步测完成时间）
            torch.cuda.synchronize(device)
            h2d_start = time.perf_counter()
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            torch.cuda.synchronize(device)
            h2d_ms = (time.perf_counter() - h2d_start) * 1000

            # forward / backward / optimizer / scheduler
            optimizer.zero_grad(set_to_none=True)
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            start_event.record()
            output = model(
                input_ids=batch['input_ids'],
                labels=batch['label_ids'],
                attention_mask=batch['attention_ids'],
            )
            loss = output.loss
            end_event.record()
            end_event.synchronize()
            forward_ms = start_event.elapsed_time(end_event)

            start_event.record()
            loss.backward()
            end_event.record()
            end_event.synchronize()
            backward_ms = start_event.elapsed_time(end_event)

            start_event.record()
            optimizer.step()
            end_event.record()
            end_event.synchronize()
            optimizer_ms = start_event.elapsed_time(end_event)

            sched_t0 = time.perf_counter()
            scheduler.step()
            scheduler_ms = (time.perf_counter() - sched_t0) * 1000

            # token 统计（.item() 会同步，放在 step 末尾，明确它在 step_wall 里）
            total_token = batch['input_ids'].numel()
            valid_token = batch['attention_ids'].sum().item()
            supervised_tokens = (batch["label_ids"][:, 1:] != -100).sum().item()
            padding_ratio = 1.0 - valid_token / total_token

            step_wall_s = time.perf_counter() - step_start
            csv_dist = {
                'step': i,
                'loss': loss.item(),
                'lr': optimizer.param_groups[0]['lr'],
                'step_wall_ms': step_wall_s * 1000,
                'dataloader_wait_ms': dataloader_wait_ms,
                'h2d_ms': h2d_ms,
                'forward_gpu_ms': forward_ms,
                'backward_gpu_ms': backward_ms,
                'optimizer_gpu_ms': optimizer_ms,
                'scheduler_cpu_ms': scheduler_ms,
                'input_tokens_per_s': total_token / step_wall_s,
                'valid_input_tokens_per_s': valid_token / step_wall_s,
                'supervised_tokens_per_s': supervised_tokens / step_wall_s,
                'padding_ratio': padding_ratio,
                'peak_alloc_GiB': torch.cuda.max_memory_allocated(device) / 1024 ** 3,
                'peak_reserved_GiB': torch.cuda.max_memory_reserved(device) / 1024 ** 3,
            }
            writer.writerow(csv_dist)
    bench_time_s = time.perf_counter() - bench_start
    print(f'[benchmark] {bench_steps} steps, total={bench_time_s:.3f}s, '
          f'avg_step={bench_time_s / bench_steps * 1000:.3f}ms')
