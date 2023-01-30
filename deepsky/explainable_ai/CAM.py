import os,sys

path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if not path in sys.path:
    sys.path.insert(1, path)

from PIL import Image
from torchvision import models, transforms
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
import cv2
import json

import torch
import torchvision
from deepsky.utils.ellab_utils import returnCAM
from deepsky.dataloaders.deepsky_dataset import *
import glob
from torch.utils.data import Dataset, DataLoader


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--data_path'  ,     help='Input folder: /media/ellab4gpu/Internal second disk/DeepSky/Data_2022/2022/')
parser.add_argument('--model_file' ,     help='model file')
parser.add_argument('--labels'     ,     default=None      ,    help='json with class names')
parser.add_argument('--image'      ,     required=True     ,    help='input image file')
parser.add_argument('--output_image' ,   default='CAM.jpg' ,    help='output image file')
# parser.add_argument('--folder'   ,     required=True     ,    help='input image folder')
parser.add_argument('--device'     ,     default='cpu'     ,    help='model file')

args = parser.parse_args()

path         =  args.data_path  
model_file   =  args.model_file 
device       =  args.device 
output_image       =  args.output_image 
labels       =  args.device 

# data         =  glob.glob(path+'/**/*.jpg',recursive=True)


model = torchvision.models.resnet18(pretrained=True ).to(device)
model.fc    = torch.nn.Linear(512,7).to(device) 
res = model.load_state_dict(torch.load(model_file)['state_dict'], strict=False)


# input image
LABELS_file = labels  
image_file = args.image  



# networks such as googlenet, resnet, densenet already use global average pooling at the end, so CAM could be used directly.
model_id = 2
if model_id == 1:
    net = models.squeezenet1_1(pretrained=True)
    finalconv_name = 'features' # this is the last conv layer of the network
elif model_id == 2:
    net = models.resnet18(pretrained=True)
    finalconv_name = 'layer4'
elif model_id == 3:
    net = models.densenet161(pretrained=True)
    finalconv_name = 'features'
finalconv_name = 'layer4'
model.eval()

netlabels =[]
# if not given, assign network outputs as names
if(LABELS_file==None):
    out = model(torch.randn(1,3,320,320))
    out = out.reshape(-1)
    for i in range(out.size[0]):
        netlabels.append(i)


# hook the feature extractor
features_blobs = []
def hook_feature(module, input, output):
    features_blobs.append(output.data.cpu().numpy())

model._modules.get(finalconv_name).register_forward_hook(hook_feature)

# get the softmax weight
params = list(model.parameters())
weight_softmax = np.squeeze(params[-2].data.numpy())


preprocess = transforms.Compose([
   transforms.Resize(280),
   transforms.Resize(256), # RandomResizedCrop
   transforms.ToTensor()
#    normalize
])

# load test image
img_pil = Image.open(image_file)
img_tensor = preprocess(img_pil)
img_variable = Variable(img_tensor.unsqueeze(0))
logit = model(img_variable)

# load the imagenet category list
classes=[str(x) for  x in range (0,7)]
classes=['1_cumulus', '2_altocumulus', '3_cirrus', '4_clearsky', '5_stratocumulus', '6_cumulonimbus', '7_mixed']

h_x = F.softmax(logit, dim=1).data.squeeze()
probs, idx = h_x.sort(0, True)
probs = probs.numpy()
idx = idx.numpy()

# output the prediction
for i in range(0, 5):
    print('{:.3f} -> {}'.format(probs[i], classes[idx[i]]))

# generate class activation mapping for the top1 prediction
CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0]])

# render the CAM and output
print('output CAM.jpg for the top1 prediction: %s'%classes[idx[0]])
img = cv2.imread(image_file)
height, width, _ = img.shape
heatmap = cv2.applyColorMap(cv2.resize(CAMs[0],(width, height)), cv2.COLORMAP_JET)
result = heatmap * 0.3 + img * 0.5
cv2.imwrite(output_image, np.concatenate([img,result]))
