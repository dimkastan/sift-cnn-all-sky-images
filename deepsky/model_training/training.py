import sys
import os
path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if not path in sys.path:
    sys.path.insert(1, path)

from deepsky.utils.utils   import save_config_file, accuracy, save_checkpoint
import torch
from tqdm import tqdm

from deepsky.dataloaders.contrastive_learning_dataset import *
from deepsky.models.classification import return_resnet_18
from deepsky.utils.utils import estimate_mean_var
from torch.utils.data import DataLoader, ConcatDataset
from abc import ABC, abstractmethod

class ParamsBase(ABC):
    @abstractmethod    
    def create(self):
        return NotImplementedError("Base class")



def train_model(train_loader, 
        epoch,
        model,
        logging,
        optimizer, 
        scheduler, 
        criterion, 
        device, 
        writer):
        """
         returns top1 accuracy, loss
        """
        model.train()
        training_loss = 0.0
        niter = 0
        top1_accuracy = 0.0
        top5_accuracy = 0
        isBest = False
        for counter, (x_batch, y_batch) in enumerate(tqdm(train_loader)):
        # for counter, (x_batch, y_batch) in enumerate(train_loader):
            # print(counter)
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            # logits = model(x_batch.cpu())
            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            top1 = accuracy(logits, y_batch, topk=(1,))
            top1_accuracy += top1[0]
            loss.backward()
            training_loss+=loss
            niter+=1
            # print(loss)
            optimizer.step()
        top1_accuracy=top1_accuracy/float(niter)
        scheduler.step()
        return top1_accuracy.cpu().item(), loss



def eval_model(loader, 
        epoch,
        model,
        logging,
        optimizer, 
        criterion,
        device, 
        max_performace,
        writer, 
        isTest=False):
        """
         if isTest=True
         returns top1 accuracy, loss
        """
        model.eval()
        #   print(f"Epoch {epoch}\tTop1 Train accuracy {top1_train_accuracy.item()} ")
        top1_accuracy =0
        top5_accuracy =0
        current_loss = 0.0
        niter=0
        with torch.no_grad():
            for counter, (x_batch, y_batch) in enumerate(tqdm(loader)):
            # for counter, (x_batch, y_batch) in enumerate(loader):
                niter+=1
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                logits = model(x_batch)
                loss = criterion(logits, y_batch)
                current_loss+=loss
                top1, top5 = accuracy(logits, y_batch, topk=(1,5))
                top1_accuracy += top1[0]
                top5_accuracy += top5[0]
            top1_accuracy = top1_accuracy/float(niter)
            top5_accuracy = top5_accuracy/float(niter)
            if(not(isTest)):
                if(top1_accuracy > max_performace):
                    isBest = True
                    max_performace = top1_accuracy
                    checkpoint_name = 'checkpoint_train_eval_other{:04d}_{}.pth.tar'.format(epoch,top1_accuracy.item())
                    save_checkpoint({
                        'epoch': epoch,
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'top1_accuracy':top1_accuracy,
                        'loss':current_loss/(niter)
                    }, is_best=isBest, filename=os.path.join(writer.log_dir, checkpoint_name))
        return top1_accuracy, current_loss/(niter) 



