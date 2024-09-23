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

**NOTE : We have uploaded two jupyter notebooks. The notebook named "Final_Submission_Code.ipynb" consists of the model that we trained. We then saved that model and used it again in the next notebook "Performance_Scores_of_Final_Submission.ipynb" where we display all the performance metrics.**

## Our approach to solve the problem :

To solve the problem we did the following : Applied augmentations on the data, Divided the data into loaders, Used a variation of the Attention UNet architecture and Finally calculated the mean IoU score as a result.

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
  - The model uses ResNet50, initialized with pretrained weights from ImageNet, to extract hierarchical features from the input.
  - The encoder captures rich feature representations at different levels, progressively downsampling the image to identify both low and high level features.

**2. Pyramid Pooling Module (Applied at bottleneck):** 
  - The deepest layer of the encoder outputs feature maps with high-level information. The Pyramid Pooling Module is applied here to capture multi-scale context.
  - It performs pooling at different scales (local and global), and the resulting features are upsampled to the original feature map size and concatenated. This helps improve segmentation by considering both fine details and broader context.

**3. Attention Mechanism:** 
  - Attention blocks are applied at each decoding step. These blocks help the model focus on important regions by refining the features from the encoder before passing them to the decoder.
  -  Attention is used to align feature maps coming from the encoder and decoder to make sure only the relevant spatial features are emphasized.

**4. Upsampling and Decoding:** 
  - The decoder gradually upscales the feature maps back to the original image size using upconvolution (transpose convolution).
  - Skip connections between corresponding encoder and decoder layers are used to preserve spatial details.
  - Some additional upsampling layers are also applied in specific layers to ensure that the output matches the original input size.

**5. Final Convolution:** 
  - A final 1x1 convolution layer reduces the number of output channels to match the number of segmentation classes, producing pixel-wise segmentation masks.

![Model_Arc](https://github.com/user-attachments/assets/06142b52-ce1a-42e4-8396-84ca11ee21cb)

![Legend](https://github.com/user-attachments/assets/645aa2de-9e13-4473-a246-05d4ac65bd48)



The pyramid pooling layer is extracted from the popular PSPNet. The detailed architecture of the pyramid pooling block is given below :

![Pyramid_png](https://github.com/user-attachments/assets/6d79ff86-9b05-444f-aebb-44512bf85375)


### Our Loss Functions and Optimizers 
As an optimizer, we used the basic Adam optimizer with the learning rate set to 0.00001. And, for the loss function, we used basic Cross Entropy Loss function.

### Metrics and Final Calculations
We calculated the F1 score, Accuracy and MIoU and IoU per class of the test images. In the main model notebook we also used the inbuilt jaccard index as a performance metric.

### Results 
- ![Results](https://github.com/user-attachments/assets/4db8a0b1-1a60-45bd-bbac-96de19c6070a)

- We achieved a **MIoU of 0.7656** with a classwise IoU of 0.9183, 0.6689, 0.8834, 0.7946, 0.5350 for Background, Building, Woodland, Water and Road classes respectively.
- Using the **Inbuilt Jaccard** IoU calculation of pytorch, we achieved a **MIoU of 0.72**.
- We got a **precision score of 0.89** and **F1 score of 0.88**.

### Segmented Maps 
Some of the segmented maps we got after training on the test set are :
- ![Segment_Map](https://github.com/user-attachments/assets/508ae040-2115-4b59-8351-440152b5cc7f)
- ![Segment_Map2](https://github.com/user-attachments/assets/0d084929-cd50-4112-b967-312ce5a16ea7)



### Limitations 
- A fairly large limitation that we faced was that off the large imbalance in the classes in the dataset provided. It would be more clear from the following image, where some classes have a very low number of pixels compared to the other classes.
- ![WhatsApp Image 2024-09-18 at 22 00 02_e5ac02f8](https://github.com/user-attachments/assets/32650a6d-9c5c-4b7a-9a79-dfddefc02c3a).
- To try and counter this problem, we tried to use the weighted Cross Entropy loss, but that did not end up making a big improvement in the results.
-  We also tried to make use of other more cutting edge techniques to conuter this problem, but couldnt impliment it in the time frame allotted to us.
- The dataset contained very high quality images, and also a large number of such images, so it took a very long time to train the models on online GPUs such as Google Colab & Kaggle.

### References
http://landcover.ai

https://ieeexplore.ieee.org/document/7780459

https://arxiv.org/abs/1612.01105v2

https://arxiv.org/abs/1804.03999



  



