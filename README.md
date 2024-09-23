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

## How to run the Notebooks

All the work on the models has been done on online GPUs, so the corresponding .ipynb files have been shared in this repository. 
To run these files, they can be opened with either Google Colab or Kaggle and run using their online GPU.

## Our approach to solve the problem :

To solve the problem we did the following : Applied augmentations on the data, Divided the data into loaders, Used an Attention UNet architecture and Finally calculated the mean IoU score as a result.

### Data Augmentation 
To improve our results, we used several data augmentations using the Albumentations Library present in PyTorch. We used applied the following augmentations on 50% of the images and masks : 
- **Random Rotate :** Rotates the images by 90 degrees to ensure that the model can identify the images taken from different points of views.
- **Horizontal Flip :** Flips the images horizontally, it helps the model become invariant to horizontal orientation changes.
- **RandomSizedCrop :** Crops a random portion of the image and resizes it to a standard size as cropping different sections of the image makes the model less dependent on the exact location of objects in the image.
- **HueSaturation :** This function is used to modify the hue, saturation, and the HSV value of an image. We set the Hue Shift to 40 units, Saturation Shift to 40 units and the Brightness Shift to 30 units. It enables the model to make accurate predictions even with lighting and colour changes.
- **RandomizedBrightnessContrast :** This technique introduces variations in how bright or dark and how sharp or soft the image appears. It helps make predictions even with differences in image quality, focus, or lighting conditions.

### Creating the Train and Test Subsets
The enitre dataset has 10674 images, where we applied a 80 - 20 Train - Test split resulting in a Train set with 8539 images, and a train set with 2135 images.

### Our Model 
The model for segmentation  that we have designed combines several advanced techniques to improve performance in challenging tasks. Here's a breakdown of the model's features : 
**1. Backbone (ResNet50) Encoder:**
  The model uses ResNet50, initialized with pretrained weights from ImageNet, to extract hierarchical features from the input. The encoder captures rich feature representations at different levels, progressively downsampling the image to identify both low- and high-level features.
**2. Pyramid Pooling Module (Applied at bottleneck):**
  The deepest layer of the encoder outputs feature maps with high-level information. The Pyramid Pooling Module is applied here to capture multi-scale context. It performs pooling at different scales (local and global), and the resulting features are upsampled to the original feature map size and concatenated. This helps improve segmentation by considering both fine details and broader context.
**3. Attention Mechanism:**
  Attention blocks are applied at each decoding step. These blocks help the model focus on important regions by refining the features from the encoder before passing them to the decoder. Attention is used to align feature maps coming from the encoder and decoder to make sure only the relevant spatial features are emphasized.
**4. Upsampling and Decoding:**
  The decoder gradually upscales the feature maps back to the original image size using upconvolution (transpose convolution). Skip connections between corresponding encoder and decoder layers are used to preserve spatial details. Some additional upsampling layers are also applied in specific layers to ensure that the output matches the original input size.
**5. Final Convolution:**
  A final 1x1 convolution layer reduces the number of output channels to match the number of segmentation classes, producing pixel-wise segmentation masks.

![first_try](https://github.com/user-attachments/assets/0bc6f792-5ffb-479f-b539-29cb1d43db2d)

The pyramid pooling layer is extracted from the popular PSPNet (more details)

![Pyramid_png](https://github.com/user-attachments/assets/6d79ff86-9b05-444f-aebb-44512bf85375)


### Our Loss Functions and Optimizers 
As an optimizer, we used the basic Adam optimizer with the learning rate set to 0.00001. But, for the loss function, along with using the basic Cross Entropy Loss function, we set up weights to be given as parameters. 
These weights were calculated by : 
(weight calc.)

### Metrics and Final Calculations
We calculated the MIoU and IoU per class of the test images (add more, like hwo we got those numbers)

### Results 
- We achieved a **MIoU of 0.7620** with a classwise IoU of 0.9130, 0.6775, 0.8732, 0.7952, 0.5256 for Background, Building, Woodland, Water and Road classes respectively.
- Using the **Inbuilt Jaccard** IoU calculation of pytorch, we achieved a **MIoU of 0.72**.
- (f1 score ?)

### Limitations 
- A fairly large limitation that we faced was that off the large imbalance in the classes in the dataset provided. It would be more clear from the following image, where some classes have a very low number of pixels compared to the other classes ![WhatsApp Image 2024-09-18 at 22 00 02_e5ac02f8](https://github.com/user-attachments/assets/32650a6d-9c5c-4b7a-9a79-dfddefc02c3a).
- To try and counter this problem, we tried to use the weighted Cross Entropy loss, but that did not end up making a big improvement in the results.
- The dataset contained very high quality images, and also a large number of such images, so it took a very long time to train the models on online GPUs such as Google Colab & Kaggle.

### References
(add some referencs)



  



