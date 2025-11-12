# Machine Learning Laboratory Projects

This repository collects the main projects developed during the *Machine Learning Laboratory* course at the University of UNIMI, UNIMIB, UNIPV.  
Each project explores a different application of machine learning and deep learning, ranging from computer vision to audio processing and generative modeling.  

All experiments were implemented in **Python** using **PyTorch** and related libraries.

---

## Project Overview

### Manga Character Generation with Diffusion Models
**Goal:** Generate manga-style faces using a Denoising Diffusion Probabilistic Model (DDPM).  
**Methods:**  
- Forward and reverse diffusion process with Gaussian noise.  
- U-Net–based CNN with residual and attention layers.  
- Training on 12,000 64×64 manga images for 20 epochs.  
**Results:**  
- Tested with different time steps (100, 300, 500).  
- Longer diffusion steps improved realism but increased computational time.  
**Key Skills:** Diffusion Models · Generative AI · Image Synthesis · PyTorch

---

### Automatic Speech Recognition (ASR) with Wave2Vec and CTC
**Goal:** Transcribe speech into text using Wave2Vec 2.0 and Connectionist Temporal Classification (CTC).  
**Methods:**  
- Self-supervised pre-trained model from *torchaudio pipelines*.  
- Greedy Decoding vs. Beam Search Decoding with different beam sizes and `n_best` settings.  
**Results:**  
- Beam Search achieved more coherent transcriptions than Greedy Decoding.  
- Increasing beam size improved accuracy but required more computation.  
**Key Skills:** Speech Recognition · Deep Learning · Sequence Modeling · NLP

---

### Audio Classification using CNNs
**Goal:** Classify speech commands from the *SpeechCommands* dataset.  
**Methods:**  
- CNN architectures **M3** (2 conv layers) and **M5** (4 conv layers).  
- Data preprocessing with padding, normalization, and resampling.  
- Adam optimizer with learning rate scheduling.  
**Results:**  
- M5 achieved around 75–80% accuracy depending on batch size and epochs.  
- Deeper models outperformed shallower ones in recognition accuracy.  
**Key Skills:** Audio Processing · CNN · MFCC · Feature Extraction

---

### Pet Classification (Cats vs Dogs)
**Goal:** Binary classification using Convolutional Neural Networks (CNNs).  
**Dataset:** Oxford Pet Dataset — 6,349 training images, 1,000 test images.  
**Architecture:**  
- Feature extractor with 5 convolutional blocks.  
- Adaptive average pooling + fully connected classifier with sigmoid output.  
**Results:**  
- Achieved around 85–88% accuracy.  
- Misclassifications mainly due to cropped or background-dominant images.  
**Key Skills:** Image Classification · CNN · Transfer Learning · Data Augmentation

---

### Pet Segmentation
**Goal:** Semantic segmentation of pet images using CNN-based architectures.  
**Methods:**  
- Encoder–decoder convolutional network.  
- Data augmentation with random flips and normalization.  
**Results:**  
- The model successfully segmented pet regions; qualitative inspection confirmed accurate boundary detection.  
**Key Skills:** Image Segmentation · Encoder–Decoder Networks · Computer Vision

---

### Variational Autoencoder (VAE)
**Goal:** Learn latent representations and reconstruct images from the latent space.  
**Methods:**  
- Encoder–decoder structure with Gaussian reparameterization trick.  
- KL divergence regularization and reconstruction loss.  
**Results:**  
- The VAE effectively generated plausible image reconstructions and interpolations.  
**Key Skills:** Variational Inference · Generative Models · Representation Learning

---

## Tools and Libraries
- **Python**  
- **PyTorch**, **Torchvision**, **Torchaudio**  
- **NumPy**, **Matplotlib**, **Scikit-learn**  
- **Google Colab** for GPU training  

---

## Evaluation Metrics
Each project was evaluated using standard metrics such as:  
- **Accuracy** (classification)  
- **Loss curves** (training convergence)  
- **Qualitative visual inspection** (for generative and segmentation tasks)

---

## Collaborators
Developed in collaboration with **Francesca Visini**  
as part of the *Machine Learning Laboratory* course during the Bachelor's Degree in Artificial Intelligence.

---

## Key Skills
Machine Learning · Deep Learning · Computer Vision · Audio Processing · Speech Recognition · Generative AI · Python Programming

