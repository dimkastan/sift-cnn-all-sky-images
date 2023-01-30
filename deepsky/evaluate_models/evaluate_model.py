import torch
import torchvision
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
import sys
import os
path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if not path in sys.path:
    sys.path.insert(1, path)

from tqdm import tqdm
from deepsky.models.classification import return_resnet_18, return_sift_cnn
from deepsky.dataloaders.deepsky_dataset import *
eval_training_set = False

import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('--arch',choices=["rgb","rgb_siftcnn"],
    help='select model architecture')
parser.add_argument('--train_data',required=True,
    help='Path to training data ')
parser.add_argument('--test_data',required=True,
    help='Path to test data ')
parser.add_argument('--weights',required=True,
    help='Path to model file')
parser.add_argument('--eval_training_set',  action='store_true',
    help='Enable evaluation on training set')
    
parser.add_argument('--device',default="cuda:0",
    help='Select device ')
args = parser.parse_args()


train_folder = args.train_data
test_folder  = args.test_data
model_file   = args.weights
device       = args.device
categories = ['cumulus', 'altocumulus',  'cirrus', 'clearsky', 'stratocumulus', 'cumulonimbus', 'mixed']



if args.arch=='rgb':
    print("loading RGB")
    model  = return_resnet_18(device=args.device, NUM_CLASSES=7, feature_only=False, model_file=args.weights)
elif args.arch == 'rgb_siftcnn':
  print("RGB_SIFTCNN")
  model = return_sift_cnn(device=args.device, NUM_CLASSES=7, feature_only=False, model_file=args.weights)




out_path='/'.join(model_file.split('/')[0:-1])


t_test=transforms.Compose([
      #   transforms.Grayscale(),
                      transforms.Resize((280,280)),
                      transforms.Resize(256), # RandomResizedCrop
                      transforms.ToTensor()
                                           ])

train_dataset = DeepSkyDatasetFolder(train_folder, transform=t_test )
test_dataset  = DeepSkyDatasetFolder(test_folder, transform=t_test )



test_loader  = DataLoader(test_dataset,  batch_size=1, shuffle=True, num_workers=4)



d = 1 
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)
train_data  = np.empty(shape=[0, d])
train_labels  = np.empty(shape=[0, 1])
pred_train_labels  = np.empty(shape=[0, 1])

test_data  = np.empty(shape=[0, d])
test_labels  = np.empty(shape=[0, 1])
pred_test_labels  = np.empty(shape=[0, 1])


model.eval()
with torch.no_grad():
  # get training data vectors
  for counter, (x_batch, y_batch) in enumerate(tqdm(train_loader)):
      x_batch = x_batch.to(device)
      y_batch = y_batch.to(device)
      vector = model(x_batch).detach().cpu().numpy()
      pred = vector.argmax()
      pred_train_labels = np.vstack([pred_train_labels, pred]) 
      train_labels = np.append(train_labels,y_batch.item())


with torch.no_grad():
  for counter, (x_batch, y_batch) in enumerate(tqdm(test_loader)):
      x_batch = x_batch.to(device)
      y_batch = y_batch.to(device)
      vector = model(x_batch).detach().cpu()
      pred = vector.argmax()
      h_x = F.softmax(vector, dim=1).data.squeeze()
      probs, idx = h_x.sort(0, True)
      probs = probs.numpy()
      idx = idx.numpy()
      # output the prediction
      # for i in range(0, 5):
      #     print('{:.3f} -> {}'.format(probs[i], categories[idx[i]]))
      pred_test_labels = np.vstack([pred_test_labels, pred]) 
      test_labels = np.append(test_labels,y_batch.item())


pred_train_labels = pred_train_labels.reshape(-1,)
pred_test_labels  = pred_test_labels.reshape(-1,)

if(eval_training_set):
  cm = confusion_matrix(train_labels, pred_train_labels.reshape(-1,))
  train_acc= (pred_train_labels==train_labels).sum()/len(train_labels)
  print(cm)
  print("Training acc = {}".format(train_acc))

cm = confusion_matrix(test_labels, pred_test_labels)
test_acc = (pred_test_labels==test_labels).sum()/len(test_labels)
print("Testing acc = {}".format(test_acc))




df = pd.DataFrame(cm, categories)
df = pd.DataFrame(cm )

cm = confusion_matrix(test_labels, pred_test_labels, normalize='true')*100
cm = np.around(cm, 2)

cmd = ConfusionMatrixDisplay(cm, display_labels=categories  )

cmd.plot(values_format='2.2f', xticks_rotation=90 )
cmd.ax_.set(xlabel='Predicted', ylabel='True')
plt.savefig(out_path+'/Confusion-{}.png'.format( datetime.datetime.now().strftime("%m_%d_%Y_%H_%M_%S") ))
with open(out_path+"/results.txt","w") as f:
  f.write("-------------------------------------- \n")
  f.write("                   Post CNN evaluation \n")
  f.write("-------------------------------------- \n")
  f.write(train_folder)
  f.write("\n")
  f.write(test_folder)
  f.write("\n")
  f.write(str(test_acc))
  f.write("\n")
  f.write(pd.DataFrame(cm ).to_string())


print("Look at")
print(out_path)