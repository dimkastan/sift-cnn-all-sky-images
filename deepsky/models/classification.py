import torch
import torchvision
from .sift_flow_torch import SiftFlowTorch

"""
 DeepSky Classification models

"""

def return_resnet_18(device, NUM_CLASSES,model_file, pretrained=True, finetune=False, feature_only=False):
    """
     regular classification model
    """
    print("Building classification model")
    if(pretrained):
        print("Building classification model from Imagenet")
    resnet       = torchvision.models.resnet18(pretrained=pretrained ).to(device) 
    if(finetune):
        for param in resnet.parameters():
          param.requires_grad = False
    resnet.fc    = torch.nn.Linear(512,NUM_CLASSES).to(device) 
    if(feature_only):
        resnet.fc= torch.nn.Identity()
    return resnet


"""
 DeepSky SiftCNN


"""

class Identity(torch.nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x

# 
# Fusion network (combines SIFT-CNN + )
#
class SIFTCNN_RESNET(torch.nn.Module):
    def __init__(self, sift_cnn, resnet, resnet_image, device, feature_only=False ):
        super(SIFTCNN_RESNET, self).__init__()
        # self.stn    = STN().to(device)
        self.sift = sift_cnn
        self.model2 = resnet
        self.model2.conv1=torch.nn.Conv2d(64,64,5,2,1).to(device)  
        self.model1 = resnet_image
        self.model2.fc=Identity() 
        self.model1.fc=Identity()
        self.dropout = torch.nn.Dropout(0.2)
        self.fc1 = torch.nn.Linear(1024,7).to(device) 
    def forward(self, x):
        x1 = self.sift.extract_descriptor(x[:,1,:,:].unsqueeze(1).to('cpu'))
        x1 = self.model2(x1)
        x2 = self.model1(x)
        x3 = torch.cat((x1, x2), dim=1)
        x3 = self.dropout(x3)
        x3=self.fc1(x3)
        return x3






def return_sift_cnn(device, NUM_CLASSES,model_file,  pretrained=True, finetune=False, feature_only=False):
    """
     sift-cnn classification model
    """
    
    # siftflownet = SiftFlowTorch(num_bins=4,cuda=True)
    siftflownet  = SiftFlowTorch(cell_size=8,num_bins=4,cuda=True, device=device)
    resnet       = torchvision.models.resnet18(pretrained=pretrained ).to(device) 
    resnet_image = torchvision.models.resnet18(pretrained=pretrained ).to(device) 
    model = SIFTCNN_RESNET(siftflownet, resnet,resnet_image,  device=device)
    if(model_file!=''):
        print(model_file)
        res = model.load_state_dict(torch.load(model_file,map_location=device)['state_dict'], strict=True   )
        print(res)
    if(feature_only==True):
        model.fc1 = torch.nn.Identity()
    return model