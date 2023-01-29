from deepsky.dataloaders.contrastive_learning_dataset import *
from deepsky.models.classification import return_resnet_18, return_sift_cnn
from ..utils.utils import estimate_mean_var
from torch.utils.data import DataLoader, ConcatDataset
from abc import ABC, abstractmethod


from torchsampler import ImbalancedDatasetSampler

from deepsky.model_training.training import train_model, eval_model


class ParamsBase(ABC):
    @abstractmethod    
    def create(self):
        return NotImplementedError("Base class")

class ParamsGRSCD(ParamsBase):
    def __init__(self,
         lr=0.005,
         weight_decay=0.00002,
         batch_size  =32,
         epochs      =500,
         NUM_CLASSES=7, 
         Normalize=True, 
         pretrained=True, 
         pretrainedModel = "",
         SIZE     = 280,
         CROP_SIZE=256,
         train_data = '', 
         val_data   = '',
         test_data  = '',
         tune_last_layer = False,
         sample_balancer = False, 
         device=None,
         config=None, 
         logging=None ):
        print("pretrainedModel = {}".format(pretrainedModel))
        print("pretrained = {}".format(pretrained))
        if(not(device==None) ):
            self.device = device
        # self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.sample_balancer = sample_balancer
        self.pretrainedModel       = pretrainedModel
        self.epochs           = epochs
        self.lr               = lr
        self.finetune         = tune_last_layer
        self.Normalize        = Normalize
        self.weight_decay     = weight_decay
        self.pretrained       = pretrained
        self.NUM_CLASSES      = NUM_CLASSES
        train_folder          = train_data
        test_folder           = test_data
        self.batch_size       = batch_size
        self.tune_last_layer  = tune_last_layer
        self.model            = return_resnet_18( device=self.device, NUM_CLASSES=7, pretrained=pretrained, finetune= self.finetune, feature_only=False, model_file=self.pretrainedModel)
        if(self.pretrainedModel!=''):
            state_dict = torch.load(self.pretrainedModel)['state_dict']
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if(k.find('backbone')==0):
                    name = k[9:] # remove `backbone.`
                    new_state_dict[name] = v
                    logging.info("adding wo backbone. {}".format(name))
                else:
                    logging.info("adding by just copying name{}".format(name))
                    new_state_dict[k]=v # leave it as is
        

        if(self.pretrainedModel==''):
            logging.info("No pretrained model")
            print("EMPTY MODEL-->{} Training will start from defaults specified by pretrained variable ".format(self.pretrainedModel))
        else:
            logging.info("model provided {}".format(self.pretrainedModel))
            print("model provided {}".format(self.pretrainedModel)) 
            res = self.model.load_state_dict(new_state_dict,strict=False) 
            logging.info(res)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay, momentum=0.9)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)
        # t_simple=transforms.Compose([
        #                     transforms.Resize((320,320)),
        #                     transforms.ToTensor()])

    
        # train_dataset_gsrcd    = DeepSkyDatasetFolder(train_folder_gsrcd     , t_simple  )
        # train_dataset = ConcatDataset([train_dataset_gsrcd])

        # mean = torch.zeros(3) # initialize 
        # std  = torch.ones(3)
        # if(self.Normalize):
        #     mean, std = estimate_mean_var(train_dataset)
        s=1
        color_jitter = transforms.ColorJitter(0.2 * s, 0.2 * s, 0.2 * s, 0.05 * s)

        t_train = transforms.Compose([
                            transforms.RandomApply([color_jitter], p=0.1),
                            transforms.Resize((SIZE,SIZE)),
                            transforms.RandomAffine(5, shear=5 ),
                            transforms.RandomRotation(90),
                            transforms.RandomAdjustSharpness(sharpness_factor=2),
                            transforms.RandomEqualize(),
                            transforms.RandomAutocontrast(),
                            transforms.GaussianBlur(kernel_size=(3, 5), sigma=(0.1, 2) ),
                            #   transforms.Grayscale(),
                            # transforms.RandomResizedCrop(size=CROP_SIZE,scale=(0.05, 0.15), ratio=(1.0,1.0)),
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomVerticalFlip(),
                            transforms.ToTensor(),
                            # transforms.Normalize(mean=mean, std = std),
                                                ])

        t_test = transforms.Compose([
                            transforms.Resize((SIZE,SIZE)),
                            # transforms.CenterCrop((CROP_SIZE,CROP_SIZE)),
                            transforms.ToTensor(),
                            # transforms.Normalize(mean=mean, std = std),
                                                ])
        logging.info(t_train)
        logging.info(t_test)
        train_dataset    = DeepSkyDatasetFolder(train_folder ,t_train  )
        test_dataset     = DeepSkyDatasetFolder(test_folder , t_test )

    
        # train_dataset = torchvision.datasets.ImageFolder(
        #     train_folder,
        #     t_train)
        # test_dataset = torchvision.datasets.ImageFolder(
        #     test_folder,
        #     t_test)  
        # if no vlidation data are provided, split train to train/val
        if(val_data==''):
          train_size = int(0.85 * len(train_dataset))
          val_size = len(train_dataset) - train_size
          train_subset, val_subset= torch.utils.data.random_split(train_dataset, [train_size, val_size])
          if(sample_balancer):
              self.train_loader = DataLoader(train_dataset,  sampler=ImbalancedDatasetSampler(train_dataset), batch_size=self.batch_size, shuffle=False, num_workers=4 )
              logging.info("Loading class balancer")
              print("Loading class balancer")
          else:
              logging.info("class balancer is None")
              print("class balancer is None")
              self.train_loader = DataLoader(train_subset,  batch_size=self.batch_size, shuffle=True, num_workers=4 )
          self.val_loader   = DataLoader(val_subset,    batch_size=self.batch_size, shuffle=False, num_workers=4 )
        else:
            self.train_loader = DataLoader(train_dataset,  batch_size=self.batch_size, shuffle=True, num_workers=4 )
            val_dataset       = DeepSkyDatasetFolder(val_data ,t_train  )
            self.val_loader   = DataLoader(val_dataset,    batch_size=self.batch_size, shuffle=False, num_workers=4 )

        self.test_loader  = DataLoader(test_dataset,  batch_size=self.batch_size, shuffle=False, num_workers=4 )
    def create(self):
        return self.epochs, self.model, self.optimizer , self.scheduler, self.criterion, self.device, self.batch_size, self.train_loader, self.val_loader, self.test_loader




class SIFTCNN_RESNET_Trainer(ParamsBase):
    def __init__(self,
         lr=0.005,
         weight_decay=0.00002,
         batch_size  =32,
         epochs      =500,
         NUM_CLASSES=7, 
         Normalize=True, 
         pretrained=True, 
         pretrainedModel = "",
         SIZE     = 280,
         CROP_SIZE=256,
         train_data = '', 
         val_data   = '',
         test_data  = '',
         tune_last_layer = False,
         sample_balancer = False, 
         device=None,
         config=None, 
         logging=None ):
        print("pretrainedModel = {}".format(pretrainedModel))
        print("pretrained = {}".format(pretrained))
        if(not(device==None) ):
            self.device = device
        # self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.sample_balancer = sample_balancer
        self.pretrainedModel       = pretrainedModel
        self.epochs           = epochs
        self.lr               = lr
        self.finetune         = tune_last_layer
        self.Normalize        = Normalize
        self.weight_decay     = weight_decay
        self.pretrained       = pretrained
        self.NUM_CLASSES      = NUM_CLASSES
        train_folder          = train_data
        test_folder           = test_data
        self.batch_size       = batch_size
        self.tune_last_layer  = tune_last_layer
        self.model            = return_sift_cnn( device=self.device, NUM_CLASSES=7, pretrained=pretrained, finetune= self.finetune, feature_only=False, model_file=self.pretrainedModel)
        
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay, momentum=0.9)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.device)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)
        # t_simple=transforms.Compose([
        #                     transforms.Resize((320,320)),
        #                     transforms.ToTensor()])

    
        # train_dataset_gsrcd    = DeepSkyDatasetFolder(train_folder_gsrcd     , t_simple  )
        # train_dataset = ConcatDataset([train_dataset_gsrcd])

        # mean = torch.zeros(3) # initialize 
        # std  = torch.ones(3)
        # if(self.Normalize):
        #     mean, std = estimate_mean_var(train_dataset)
        s=1
        color_jitter = transforms.ColorJitter(0.2 * s, 0.2 * s, 0.2 * s, 0.05 * s)

        t_train = transforms.Compose([
                            transforms.RandomApply([color_jitter], p=0.1),
                            transforms.Resize((SIZE,SIZE)),
                            transforms.RandomAffine(5, shear=5 ),
                            transforms.RandomRotation(90),
                            transforms.RandomAdjustSharpness(sharpness_factor=2),
                            transforms.RandomEqualize(),
                            transforms.RandomAutocontrast(),
                            transforms.GaussianBlur(kernel_size=(3, 5), sigma=(0.1, 2) ),
                            #   transforms.Grayscale(),
                            # transforms.RandomResizedCrop(size=CROP_SIZE,scale=(0.05, 0.15), ratio=(1.0,1.0)),
                            transforms.RandomHorizontalFlip(),
                            transforms.RandomVerticalFlip(),
                            transforms.ToTensor(),
                            # transforms.Normalize(mean=mean, std = std),
                                                ])

        t_test = transforms.Compose([
                            transforms.Resize((SIZE,SIZE)),
                            # transforms.CenterCrop((CROP_SIZE,CROP_SIZE)),
                            transforms.ToTensor(),
                            # transforms.Normalize(mean=mean, std = std),
                                                ])
        logging.info(t_train)
        logging.info(t_test)
        train_dataset    = DeepSkyDatasetFolder(train_folder ,t_train  )
        test_dataset     = DeepSkyDatasetFolder(test_folder , t_test )

    
        # train_dataset = torchvision.datasets.ImageFolder(
        #     train_folder,
        #     t_train)
        # test_dataset = torchvision.datasets.ImageFolder(
        #     test_folder,
        #     t_test)  
        # if no vlidation data are provided, split train to train/val
        if(val_data==''):
          train_size = int(0.85 * len(train_dataset))
          val_size = len(train_dataset) - train_size
          train_subset, val_subset= torch.utils.data.random_split(train_dataset, [train_size, val_size])
          if(sample_balancer):
              self.train_loader = DataLoader(train_dataset,  sampler=ImbalancedDatasetSampler(train_dataset), batch_size=self.batch_size, shuffle=False, num_workers=4 )
              logging.info("Loading class balancer")
              print("Loading class balancer")
          else:
              logging.info("class balancer is None")
              print("class balancer is None")
              self.train_loader = DataLoader(train_subset,  batch_size=self.batch_size, shuffle=True, num_workers=4 )
          self.val_loader   = DataLoader(val_subset,    batch_size=self.batch_size, shuffle=False, num_workers=4 )
        else:
            self.train_loader = DataLoader(train_dataset,  batch_size=self.batch_size, shuffle=True, num_workers=4 )
            val_dataset       = DeepSkyDatasetFolder(val_data ,t_train  )
            self.val_loader   = DataLoader(val_dataset,    batch_size=self.batch_size, shuffle=False, num_workers=4 )

        self.test_loader  = DataLoader(test_dataset,  batch_size=self.batch_size, shuffle=False, num_workers=4 )
    def create(self):
        return self.epochs, self.model, self.optimizer , self.scheduler, self.criterion, self.device, self.batch_size, self.train_loader, self.val_loader, self.test_loader
