import torch

def save_model(model,optimizer,scheduler,checkpoint_file,epoch):
    torch.save(
        {'model': model.state_dict(),
         'optimizer': optimizer.state_dict(),
         'scheduler': scheduler.state_dict(),
         'epoch':epoch
         },checkpoint_file
    )