## Implementation of the SIFT-CNN method  

This repo implements the following paper:
<em>
Tsourounis, D.; Kastaniotis, D.; Theoharatos, C.; Kazantzidis, A.; Economou, G. [SIFT-CNN: When Convolutional Neural 
Networks Meet Dense SIFT Descriptors for Image and Sequence Classification](https://www.mdpi.com/2313-433X/8/10/256). J. 
Imaging 2022, 8, 256. https://doi.org/10.3390/jimaging8100256
</em>

This repository contains code to train CNN topologies for the allsky cloud type classification problem using the TJNU-Ground-based-Remote-Sensing-Cloud-Database (TJNU-GRSCD). The TJNU-GRSCD dataset is composed of 4,000 training images and 4,000 test images from 7 classes according to seven sky types [1] [2]. Two CNN topologies are implemented on this repository. 

![](imgs/Pixel_CNN_and_lateFusionWithSIFT_CNN.png)


The first CNN topology is fed with RGB images and provides the final classification result as a typical CNN model for the image classification problem. This approach is referred as Pixel-CNN.

The second CNN topology involves the SIFT-CNN method and is referred as late fusion pixel-CNN and SIFT-CNN. This second topology is an extension of the Pixel-CNN since it combines the Pixel-CNN and the SIFT-CNN. The proposed late fusion scheme with the pixel-CNN and SIFT-CNN consists of two streams. On the one stream, the RGB image is fed into the pixel-CNN and produces a 512-dimensional feature vector (as in the first case below). On the other stream, the RGB image is converted to Grayscale image and then it is fed into the SIFT-CNN which is produced another 512-dimensional feature vector. Finally, these two vectors are concatenated (resulting in a final combined feature vector with 1024 dimensions) and then fed to a fully connected layer with seven neurons for the final class prediction into seven cloud classes. Ultimately, the two-stream scheme is trained following an end-to-end fashion.

Also, this repository contains code to train SVM classifier utilizing the final feature vectors which could be extracted from the CNN topologies after their training.


[1] Liu, S., Li, M., Zhang, Z., Cao, X., & Durrani, T. S. (2020). Ground‐based cloud classification using task‐based 
graph convolutional network. Geophysical Research Letters, 47(5), e2020GL087338.

[2] Liu, S., Li, M., Zhang, Z., Xiao, B., & Durrani, T. S. (2020). Multi-evidence and multi-modal fusion network for 
ground-based cloud recognition. Remote Sensing, 12(3), 464.


#### Dataset

The project uses the TJNU-GRSCD dataset and therefore, you need to download the data from the official repository: https://github.com/shuangliutjnu/TJNU-Ground-based-Remote-Sensing-Cloud-Database.



### Notice:
We rollout pieces of the code step-by-step so stay tuned!


### Environment setup

- Git-LFS
```
sudo apt-get install git-lfs
```

- Python


pip install -r requirements.txt



### setup environment

Tested with:
- 1.13.1+cu117

```
python3 -m venv ./myenv
source ./myenv/bin/activate
pip3 install -r requirements.txt


```

### Train the Pixel-CNN with RGB input images

![](imgs/Pixel_CNN.png)


Current script loads params from rgb_params 

```bash
# train on grscd
python deepsky/model_training/train.py --config config/gsrcd.json
```

- In another terminal run tensorboard

```bash
tensorboard --logdir runs/
```

### Train the late fusion scheme with the combination of Pixel-CNN + SIFT-CNN

![](imgs/lateFusionWithSIFT_CNN.png)
```
python deepsky/model_training/train_siftcnn_fusion.py --config=config/grscd_imagenet_siftcnnfusion.json
```

- In another terminal run tensorboard

```bash
tensorboard --logdir runs/
```

```
[Accuracy][2]Train, val, test =89.86565399169922 ,89.74781036376953, 82.87500762939453
100%|███████████████████████████████████████████████████████████████| 107/107 [00:32<00:00,  3.25it/s]
100%|█████████████████████████████████████████████████████████████████| 19/19 [00:05<00:00,  3.21it/s]
100%|███████████████████████████████████████████████████████████████| 125/125 [00:23<00:00,  5.23it/s]
[Accuracy][3]Train, val, test =92.14369201660156 ,90.0767593383789, 85.1500015258789
100%|███████████████████████████████████████████████████████████████| 107/107 [00:31<00:00,  3.37it/s]
100%|█████████████████████████████████████████████████████████████████| 19/19 [00:06<00:00,  3.13it/s]
100%|███████████████████████████████████████████████████████████████| 125/125 [00:23<00:00,  5.25it/s]
[Accuracy][4]Train, val, test =94.04205322265625 ,94.57237243652344, 87.12500762939453
100%|███████████████████████████████████████████████████████████████| 107/107 [00:32<00:00,  3.28it/s]
100%|█████████████████████████████████████████████████████████████████| 19/19 [00:06<00:00,  2.95it/s]
100%|███████████████████████████████████████████████████████████████| 125/125 [00:23<00:00,  5.38it/s]
[Accuracy][5]Train, val, test =94.91822052001953 ,93.25657653808594, 88.42500305175781
```

### Evaluate model

#### General info
Load a checkpoint, define a train and a test set and run model evaluation

Warning: By default, the results and the confusion matrix will be writen to the model's folder in `results.txt` and `Confusion-01_25_2023_23_13_16.png`

Note 2: If you want to run an evaluation for a trained model you can find where it was trained on by looking at `runs/experiment-number.log/training.log` file. There we mention the train_data and test_data variables
However, you can run the evaluation on any dataset.

#### How to run


You need to select an architecture and provide the weights of the model trained with this architecture.
In the following example the `rgb_siftcnn` architecture is used and the model is `checkpoint_train_eval_other0004_94.57237243652344.pth.tar` located in `runs/Jan30_00-39-02_ellab4gpu-X299X-AORUS-MASTERrgb.log/`

```
python deepsky/evaluate_models/evaluate_model.py --weights="runs/Jan30_00
-39-02_ellab4gpu-X299X-AORUS-MASTERrgb.log/checkpoint_train_eval_other0004_94.57237243652344.pth.tar" --arch=rgb_siftcnn --train_data=/home/ellab4gpu/KastanWorkingDir/GRSCD/train --test_data=/home/ellab4gpu/KastanWorkingDir/GRSCD/test    
```

You will see something like:
```
100%|████████████████████████████████████████████████████████████| 4000/4000 [00:39<00:00, 101.53it/s]
100%|████████████████████████████████████████████████████████████| 4000/4000 [00:37<00:00, 106.09it/s]
Testing acc = 0.86425
Look at
runs/Jan30_00-39-02_ellab4gpu-X299X-AORUS-MASTERrgb.log
```

That means that you can find the model inside `runs/Jan30_00-39-02_ellab4gpu-X299X-AORUS-MASTERrgb.log`

```
--------------------------------------
                   Post CNN evaluation
--------------------------------------
/home/ellab4gpu/KastanWorkingDir/GRSCD/train
/home/ellab4gpu/KastanWorkingDir/GRSCD/test
0.86425
       0      1      2       3      4      5      6
0  97.33   1.74   0.00    0.13   0.00   0.00   0.80
1   0.30  73.11   2.72    0.00  15.71   6.65   1.51
2   0.00   1.63  95.39    0.00   0.30   0.00   2.67
3   0.00   0.00   0.00  100.00   0.00   0.00   0.00
4   0.00   0.00   0.86    2.81  90.28   5.62   0.43
5   0.00   0.00   1.87    0.85  13.97  83.30   0.00
6   6.27  21.18  21.57    0.00   0.20   1.76  49.02
````


### Explainable AI

Note: Currently works only with RGB model.
Soon I will add the RGB+SIFT-CNN

![](imgs/CAM.jpg)

How to use:

```
python deepsky/explainable_ai/CAM.py --image=imgs/cirrus.jpg  --model_file=<you-rgb-model>
```

```
0.992 -> 3_cirrus
0.008 -> 4_clearsky
0.000 -> 2_altocumulus
0.000 -> 7_mixed
0.000 -> 6_cumulonimbus
output CAM.jpg for the top1 prediction: 3_cirrus
```


## Funding

This research has been co-financed by the European Union and Greek national funds through the Operational Program Competitiveness, Entrepreneurship and Innovation, under the call RESEARCH–CREATE–INNOVATE (project code: T1EDK–00681, MIS 5067617).


## Citation

If you use our code, please consider citing the following paper:

```
@Article{jimaging8100256,
AUTHOR = {Tsourounis, Dimitrios and Kastaniotis, Dimitris and Theoharatos, Christos and Kazantzidis, Andreas and 
Economou, George},
TITLE = {SIFT-CNN: When Convolutional Neural Networks Meet Dense SIFT Descriptors for Image and Sequence Classification},
JOURNAL = {Journal of Imaging},
VOLUME = {8},
YEAR = {2022},
NUMBER = {10},
ARTICLE-NUMBER = {256},
URL = {https://www.mdpi.com/2313-433X/8/10/256},
ISSN = {2313-433X},
DOI = {10.3390/jimaging8100256}
}
```