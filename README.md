# Radial Basis Function Networks for Convolutional Neural Networks to Learn Similarity Distance Metric and Improve Interpretability:
This repository contains the code to reproduce the paper which can be found open access under the following link: https://ieeexplore.ieee.org/document/9133368
## Video presentation:

[<img src="https://github.com/moamirian/CNN_RBFs/blob/master/presentation.jpg" width="50%">](https://drive.google.com/file/d/1jekQdVLb9lerZOzH-1EG82yjqKbtTsal/view?usp=sharing)

## Software requirements:
Python: 3.6.5  
Tensorflow: 1.12.0  
Keras: 2.2.4  
Cuda: 10.0   
cuDNN: 7.4  
Auto-agment: https://github.com/4uiiurz1/keras-auto-augment  
Docker image: docker pull amir88/tf:keras_cu10.0-dnn7.4
## Reproducing experiments:
- Visualizing the training process:  
  python train_process.py
- Hyperparameter search:  
  example:
    
- Training a model for a dataset:  
  python run_experiment.py --parameters=values  
  example: python run_experiment.py --weights=imagenet --dataset=cifar100 --backbone=EfficientNet --augmentation=autogment --learning_rate=0.00002355 --weight_decay=1.090e-7 --rbf_dims=64 --batch_size=32 --loss_constant=0.1141 --centers=20 --dropout=0.0
- Visualitzing the results:  
  python visualitzing.py --dataset=dataset_name --model_directory=directory_name  
  example: python visualization.py --dataset=cifar100 --model_dir=backbone_EfficientNet_rbf_dims_64_centers_20_learning_rate_2.355e-05_weights_imagenet_augmentation_autogment_ 
