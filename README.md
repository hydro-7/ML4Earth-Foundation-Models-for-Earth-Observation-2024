# ML4Earth
This repository contains all the notebooks we made as a team while participating in the ML4Earth hackathon (https://ml4earth24.devpost.com/).
As a part of the hackathon, we are required to provide a segmentation map for the Landcover.ai Dataset.

Semantic segmentation of LandCover.ai dataset
==============================

The dataset used in this project is the [Landcover.ai Dataset](https://landcover.ai.linuxpolska.com/), 
which was originally published with [LandCover.ai: Dataset for Automatic Mapping of Buildings, Woodlands, Water and Roads from Aerial Imagery paper](https://arxiv.org/abs/2005.02264)
also accessible on [PapersWithCode](https://paperswithcode.com/paper/landcover-ai-dataset-for-automatic-mapping-of).

**Please note that I am not the author or owner of this dataset, and I am using it under the terms of the license specified by the original author. 
All credits for the dataset go to the original author and contributors.**

Dataset Details
==============================
1. The images have 3 spectral bands (i.e, RGB)
2. 33 orthophotos with 25 cm per pixel resolution (~9000x9500 px) and 8 orthophotos with 50 cm per pixel resolution (~4200x4700 px)
3. The dataset consists of 5 classes : 1) Woodland 2) Water 3) Roads 4) Buildings and 5) Background
4. The area split here is : 1.85 km2 of buildings, 72.02 km2 of woodlands, 13.15 km2 of water, 3.5 km2 of roads

![WhatsApp Image 2024-09-18 at 22 00 02_e5ac02f8](https://github.com/user-attachments/assets/32650a6d-9c5c-4b7a-9a79-dfddefc02c3a)

