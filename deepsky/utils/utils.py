import os
import shutil

import torch
import yaml
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data import   ConcatDataset, DataLoader
from tqdm import tqdm
from torchvision import datasets

def estimate_mean_var( dataset : datasets  ,N_CHANNELS : int= 3, logging: any=None):
    loader  = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    mean = torch.zeros(N_CHANNELS)
    std = torch.zeros(N_CHANNELS)
    if(logging != None):
        logging.info('==> Computing mean and std..')
    for inputs, _labels in tqdm(loader):
        for b in range(inputs.shape[0]):
            for i in range(N_CHANNELS):
                mean[i] += inputs[b,i,:,:].mean()
                std[i] += inputs[b,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def save_config_file(model_checkpoints_folder, args):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        with open(os.path.join(model_checkpoints_folder, 'config.yml'), 'w') as outfile:
            yaml.dump(args, outfile, default_flow_style=False)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
 
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res



def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
        plt.savefig("current_batch.png")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.savefig("current_batch.png")