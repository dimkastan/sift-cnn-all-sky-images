import torch
import sys
import numpy as np
import os

from skimage import io, transform
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix  

from sklearn.svm import SVC
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV

path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if not path in sys.path:
    sys.path.insert(1, path)
print(sys.path)

from deepsky.models.classification import return_resnet_18, return_sift_cnn

import yaml
import matplotlib.pyplot as plt
import torchvision
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets
from deepsky.dataloaders.contrastive_learning_dataset import *
import torch
import glob
from torch.utils.data import Dataset, DataLoader
import os
import glob
from skimage import io, transform
from skimage.color import rgb2gray
import torch

import matplotlib.pyplot as plt 
import numpy as np
import PIL
import random
from sklearn import svm
from deepsky.utils.utils import accuracy
import argparse
import json


categories = ['cumulus','altocumulus and cirrocumulus','cirrus and cirrostratus', 'clear sky','stratocumulus, stratus and altostratus','cumulonimbus and nimbostratus', 'mixed cloudness']

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print("Using device:", device)



parser = argparse.ArgumentParser()

parser.add_argument('--arch',choices=["rgb","rgb_siftcnn"],
    help='select model architecture')
parser.add_argument('--weights',required=True,
    help='select model weights')
parser.add_argument('--device',choices=["cuda","cpu"],default='cuda',
    help='select model weights')
parser.add_argument('--train_data' ,default='/home/ellab4gpu/KastanWorkingDir/GRSCD/train',
    help='select training data folder')
parser.add_argument('--test_data' ,default='/home/ellab4gpu/KastanWorkingDir/GRSCD/test',
    help='select test data folder')
parser.add_argument('--d',type=int, choices=[512,1024],default=512,
    help='select vector dimension')
args = parser.parse_args()

d = int(args.d)

train_folder = args.train_data
test_folder  = args.test_data


if args.arch=='rgb':
    print("loading RGB")
    model  = return_resnet_18(device=args.device, NUM_CLASSES=7, feature_only=True, model_file=args.weights)
elif args.arch == 'rgb_siftcnn':
  print("RGB_SIFTCNN")
  model = return_sift_cnn(device=args.device, NUM_CLASSES=7, feature_only=True, model_file=args.weights)




s=1
color_jitter = transforms.ColorJitter(0.2 * s, 0.2 * s, 0.2 * s, 0.05 * s)
t_train=transforms.Compose([
                      transforms.RandomApply([color_jitter], p=0.1),
                      transforms.RandomAffine(5),
                      transforms.RandomRotation(90),
                      transforms.Resize((280,280)),
                    #   transforms.Grayscale(),
                      transforms.RandomResizedCrop(size=256,scale=(0.2,1.2)),
                      transforms.RandomHorizontalFlip(),
                      transforms.RandomVerticalFlip(),
                      transforms.ToTensor()
                                           ])

t_test=transforms.Compose([
#   transforms.Grayscale(),
                      transforms.Resize((280,280)),
                      transforms.Resize(256), # RandomResizedCrop
                      transforms.ToTensor()
                                           ])




train_dataset = DeepSkyDatasetFolder(train_folder, transform=t_train )
test_dataset = DeepSkyDatasetFolder(test_folder, transform=t_test )


train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)
test_loader  = DataLoader(test_dataset,  batch_size=1, shuffle=True, num_workers=4)



train_data  = np.empty(shape=[0, d])
train_labels  = np.empty(shape=[0, 1])

test_data  = np.empty(shape=[0, d])
test_labels  = np.empty(shape=[0, 1])


model.eval()
for i in range(2):
  with torch.no_grad():
    # get training data vectors
    for counter, (x_batch, y_batch) in enumerate(train_loader):
      #   print(" Processing {}".format(counter))
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        vector = model(x_batch).detach().cpu().numpy()
        train_data = np.append(train_data, vector,axis=0)
        train_labels = np.append(train_labels,y_batch.item())

with torch.no_grad():
  for counter, (x_batch, y_batch) in enumerate(test_loader):
    #   print("Processing {}".format(counter))
      x_batch = x_batch.to(device)
      y_batch = y_batch.to(device)
      vector = model(x_batch).detach().cpu().numpy()
      test_data = np.append(test_data, vector,axis=0)
      test_labels = np.append(test_labels,y_batch.item())



Xtrain = preprocessing.normalize(train_data, norm='l2')
Xtest = preprocessing.normalize(test_data, norm='l2')


clf = svm.SVC(kernel='linear')
clf.fit(Xtrain, train_labels)

pred = clf.predict(Xtrain)

cm = confusion_matrix(train_labels, pred)
(pred==train_labels).sum()/len(train_labels)

pred = clf.predict(Xtest)
cm = confusion_matrix(test_labels, pred)
(pred==test_labels).sum()/len(test_labels)





df = pd.DataFrame(cm, categories)
df = pd.DataFrame(cm )
# 0.88175 (max value observed)
#      0    1    2    3    4    5    6
# 0  725    1    4    4    0    1   13
# 1    0  275    1    0   32    7   16
# 2    0    2  592    1    6    2   70
# 3    0    0    0  688    0    0    0
# 4    5    0    1   33  379   42    3
# 5    0    0    0    0   49  538    0S
# 6   34   82   63    0    0    1  330

cm = confusion_matrix(test_labels, pred, normalize='true')*100
# cm = np.around(cm, 2)
cmd = ConfusionMatrixDisplay(cm, display_labels=categories  )
cmd.plot(values_format='2.2f', xticks_rotation=90 )
cmd.ax_.set(xlabel='Predicted', ylabel='True')
plt.savefig('LinearSVM-fusion.png')




clf = svm.SVC(kernel='rbf')
clf.fit(Xtrain, train_labels)

pred = clf.predict(train_data)

confusion_matrix(train_labels, pred)
(pred==train_labels).sum()/len(train_labels)

pred = clf.predict(Xtest)
(pred==test_labels).sum()/len(test_labels)

cm = confusion_matrix(test_labels, pred)
print(cm)
print((pred==test_labels).sum()/len(test_labels))


# Hyperparameter search



C_range = np.logspace(-2, 10, 13)
gamma_range = np.logspace(-9, 3, 13)
C_range=[6,10,100]
gamma_range=[0.1,0.5,2,4]
param_grid = dict(gamma=gamma_range, C=C_range)
cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
grid = GridSearchCV(SVC(), param_grid=param_grid, cv=cv)
grid.fit(Xtrain, train_labels)


# dimensionality reduction and normalization?
print(
    "The best parameters are %s with a score of %0.2f"
    % (grid.best_params_, grid.best_score_)
)

clf = svm.SVC(kernel='rbf',C=grid.best_params_['C'], gamma=grid.best_params_['gamma'])
clf.fit(Xtrain, train_labels)
pred = clf.predict(Xtest)
cm=confusion_matrix(test_labels, pred)
(pred==test_labels).sum()/len(test_labels)

print(cm)
print((pred==test_labels).sum()/len(test_labels))

