import torch
import sys
import os
path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if not path in sys.path:
    sys.path.insert(1, path)

import numpy as np
SEED = 100
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False
np.random.seed(SEED)


from time import time
import logging
from torch.utils.tensorboard import SummaryWriter
from deepsky.utils.utils   import estimate_mean_var, save_config_file, accuracy, save_checkpoint, matplotlib_imshow

from deepsky.model_training.training import train_model, eval_model
from deepsky.model_training import *
from deepsky.model_training.rgb_params import  SIFTCNN_RESNET_Trainer  




import argparse
import json

parser = argparse.ArgumentParser()

parser.add_argument('--config',default="config/gsrcd.json",
    help='Load settings from json file  ')
parser.add_argument('--device',default="cuda:0",
    help='Select device ')
args = parser.parse_args()

if args.config:
    with open(args.config, 'rt') as f:
        t_args = argparse.Namespace()
        t_args.__dict__.update(json.load(f))
        t_args.__dict__.update({'config':args.config})
        args = parser.parse_args(namespace=t_args)
else:
    print("You have to provide a config.json file")

writer = SummaryWriter(comment="rgb.log")
logging.basicConfig(filename=os.path.join(writer.log_dir, 'training.log'), level=logging.INFO)

print("Configuration file variables ...\n")
logging.info("Configuration file variables ...\n")
logging.info(vars(args))


print("Parsing and generating training Configuration  ...\n")
P = SIFTCNN_RESNET_Trainer(**vars(args), logging=logging)
epochs,\
 model,\
 optimizer ,\
 scheduler,\
 criterion,\
 device,\
 batch_size,\
 train_loader,\
 val_loader,\
 test_loader =\
P.create()


print(model)
init_weights = "/home/ellab4gpu/KastanWorkingDir/Dimitris-SIMCLR/SimCLR/PRETRAINED_ADAM_SIFT_CNN/checkpoint_train_eval_other0054.pth.tar"
# res = model.load_state_dict(torch.load(init_weights)['state_dict'], strict=False)



logging.info(vars(args))
logging.info(vars(P))

print("Start training...")


max_performace = 0
for epoch in range(epochs):
    top1train,losstrain = \
    train_model(train_loader,
        epoch,
        model,
        logging,
        optimizer,
        scheduler,
        criterion, 
        device, 
        writer)
    top1val,lossval =\
    eval_model(val_loader,
        epoch,
        model,
        logging,
        optimizer,
        criterion, 
        device, 
        max_performace,
        writer)
    if(top1val > max_performace):
        max_performace = top1val
    top1test,losstest =\
    eval_model(test_loader,
        epoch,
        model,
        logging,
        optimizer,
        criterion, 
        device, 
        max_performace,
        writer, isTest=True)
    # if(top1val > max_performace):
    #     max_performace = top1val
    writer.add_scalars(f'loss/check_info',{'training loss':     losstrain  ,
                         'validation loss':  lossval  ,
                         'test loss':        losstest},
                            epoch * len(train_loader)  )
    writer.add_scalars(f'accuracy/check_info',{'training acc':      top1train  ,
                         'validation acc':   top1val  ,
                         'test acc':         top1test},
                            epoch * len(train_loader)  )                   
    Acc = '[Accuracy][{}]Train, val, test ={} ,{}, {}'.format(epoch,top1train, top1val, top1test)
    Loss = '[Loss]   [{}]Train, val, test ={} ,{}, {}'.format(epoch,losstrain, lossval, losstest)
    print(Acc)
    # print(Loss)
    logging.info(Acc)
    logging.info(Loss)
        