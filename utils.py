import shutil
import torch
from config import *

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def save_checkpoint(state_dict, is_best, filename='./save_model/fusion_current_model.pth.tar'):
    torch.save(state_dict, filename)
    if is_best:
        shutil.copyfile(filename, path2best_model)

def adjust_learning_rate(optimizer, cur_lr, lr_decay, epoch):
    """Sets the learning rate to the initial LR decayed by 0.1 for every 10 epochs"""
    l_rate = cur_lr * (lr_decay ** (epoch // 6))
    for param_group in optimizer.param_groups:
        param_group['lr'] = l_rate