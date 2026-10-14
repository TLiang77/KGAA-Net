import torch
from tools import MultiItemAverageMeter

def train(base, loaders, config, optimizer):
    base.set_train()
    meter = MultiItemAverageMeter()
    loader = loaders.get_train_loader()
    if "RN" in config.pretrain_choice:
        mode = "1/3"
    elif "VIT" in config.pretrain_choice:
        mode = None
    else: 
        ValueError(f"Pretrain model {config.pretrain_choice} choice not supported")

    for i, batch_dict in enumerate(loader):

        batch_dict = {key: value.to(base.device) for key, value in batch_dict.items()}

        optimizer.zero_grad()

            # get loss
        ret = base(batch_dict, mode)
        losses = [value for key, value in ret.items() if 'loss' in key]
        total_loss = sum(losses)

        total_loss.backward()
        optimizer.step()

        # update meter
        acc_sign = False
        acc_value = 0
        for key, value in ret.items():
            if 'loss' in key:
                meter.update({key: value})
            if 'acc' in key:
                acc_sign = True
                acc_value = value
                # meter.update({key: value})
        meter.update({'total_loss': total_loss})
        if acc_sign:
            meter.update({'acc': acc_value})
            

    return meter.get_val(), meter.get_str()

