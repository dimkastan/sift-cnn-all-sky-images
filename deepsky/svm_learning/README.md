# Training SVM on the network's outputs


- Traing SIFT-CNN+ RGB

```
python deepsky/meta_learning/train_svm.py  --weights="weights/rgb-sift-cnn-svm-8670/checkpoint_train_eval_best.pth.tar" --arch=rgb_siftcnn --d=1024 --train_data=/home/ellab4gpu/KastanWorkingDir/GRSCD/train --test_data=/home/ellab4gpu/KastanWorkingDir/GRSCD/test
```

- Train RGB

```
python deepsky/meta_learning/train_svm.py  --weights="weights/checkpoint_train_eval_other0054.pth.tar" --arch=rgb --d=512 --train_data=/home/ellab4gpu/KastanWorkingDir/GRSCD/train --test_data=/home/ellab4gpu/KastanWorkingDir/GRSCD/test
```