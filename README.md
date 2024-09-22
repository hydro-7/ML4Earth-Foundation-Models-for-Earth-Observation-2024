# ML4Earth
This repository contains the final submission we made as a team while participating in the ML4Earth hackathon (https://ml4earth24.devpost.com/).
As a part of the hackathon, we are required to provide a segmentation map for the Landcover.ai Dataset.

## Our Team

- Dr. Shounak Chakraborty
- Vishnu Meher Vemulapalli
- Paarth Batra
- Dhruv Singh
- Siddharth Karmokar
- Vinayak Sharma
  
## Overview of our work 

(enter text)

## Semantic segmentation of LandCover.ai dataset

The dataset used in this project is the [Landcover.ai Dataset](https://landcover.ai.linuxpolska.com/), 
which was originally published with [LandCover.ai: Dataset for Automatic Mapping of Buildings, Woodlands, Water and Roads from Aerial Imagery paper](https://arxiv.org/abs/2005.02264)
also accessible on [PapersWithCode](https://paperswithcode.com/paper/landcover-ai-dataset-for-automatic-mapping-of).

**Please note that I am not the author or owner of this dataset, and I am using it under the terms of the license specified by the original author. 
All credits for the dataset go to the original author and contributors.**

## Dataset Details
1. The images have 3 spectral bands (i.e, RGB)
2. 33 orthophotos with 25 cm per pixel resolution (~9000x9500 px) and 8 orthophotos with 50 cm per pixel resolution (~4200x4700 px)
3. The dataset consists of 5 classes : 1) Woodland 2) Water 3) Roads 4) Buildings and 5) Background
4. The area split here is : 1.85 km2 of buildings, 72.02 km2 of woodlands, 13.15 km2 of water, 3.5 km2 of roads

![WhatsApp Image 2024-09-18 at 22 00 02_e5ac02f8](https://github.com/user-attachments/assets/32650a6d-9c5c-4b7a-9a79-dfddefc02c3a)

## How to run the Notebooks

All the work on the models has been done on online GPUs, so the corresponding .ipynb files have been shared in this repository. 
To run these files, they can be opened with either Google Colab or Kaggle and run using their online GPU.

## Our approach to solve the problem :

To solve the problem we did the following : Applied augmentations on the data, Divided the data into loaders, Used an Attention UNet architecture and Finally calculated the mean IoU score as a result.

### Data Augmentation 
To improve our results, we used several data augmentations using the Albumentations Library present in PyTorch. We used applied the following augmentations on both the images and masks : 
- Random Rotate : (details ?)
- Horizontal Flip : ()
- RandomSizedCrop : ()
- HueSaturation : ()
- RandomizedBrightnessContrast : ()

### Our Model 
In our model, we used an Attention UNet architecture with a RESNET 50 Encoding Block and a Pyramid Pooling Block at the bottleneck of the UNet. The RESNET we used wasnt pretrained, we just imported the Imagenet weights. A good explanation of the architecture we created can be obtained from this image : (change to updated image)

![first_try](https://github.com/user-attachments/assets/0bc6f792-5ffb-479f-b539-29cb1d43db2d)

The pyramid pooling layer is extracted from the popular PSPNet (more details)

(pyramid pooling image)

### Our Loss Functions and Optimizers 
As an optimizer, we used the basic Adam optimizer with the learning rate set to 0.00001. But, for the loss function, along with using the basic Cross Entropy Loss function, we set up weights to be given as parameters. 
These weights were calculated by : 
(weight calc.)

### Metrics and Final Calculations
We calculated the MIoU and IoU per class of the test images (add more, like hwo we got those numbers)

### Results 
(final results)

### Limitations 
-A fairly large limitation that we faced was that off the large imbalance in the classes in the dataset provided. It would be more clear from the following image, where some classes have a very low number of pixels compared to the other classes (insert picture).
- The dataset contained very high quality images, and also a large number of such images, so it took a very long time to train the models on online GPUs such as Google Colab & Kaggle.
- We tried to implement Cyclic GAN onto the dataset to try to increase the pixel ratio among the classes that occurred less, but we werent able to finish the final implementation in the time frame provided.

### References
(add some referencs)



  



