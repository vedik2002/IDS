# Federated Multi-Layered Deep Learning (Fed-MLDL) for IoT Intrusion Detection

## Overview

The **Federated Multi-Layered Deep Learning (Fed-MLDL)** project aims to enhance intrusion detection systems (IDS) in the Internet of Things (IoT) environment by utilizing federated learning and physics-based hyperparameter optimization (HPO) techniques. With the increasing prevalence of IoT devices and the corresponding surge in cyber-attacks, this project addresses the need for optimized training methods that maintain data privacy across distributed environments.

## Abstract

Internet of Things (IoT) is reshaping our lives with its omnipresence. The sudden uptick in the ubiquitous nature of IoT devices, ranging from fitness watches to aircraft, has led to a surge of cyber-attacks. Artificial Intelligence-powered Intrusion Detection Systems (IDS) are being used recently to combat this increasing surge of attacks in the IoT environment. However, existing solutions lack optimization for training in distributed decentralized environments.

A popular solution for training a model in a decentralized environment is Federated Learning. Multiple client models collaboratively train a global model while keeping the individual client’s data decentralized and private. This, however, suffers from poor generalization of the individual client data.

This work proposes a new Federated Multi-Layered Deep Learning (Fed-MLDL) model that employs physics-based Hyper Parameter Optimization (HPO) techniques in a distributed federated learning environment for intrusion detection on the CICIoT23 dataset. The physics-based hyperparameter optimization techniques ensure good generalization for all clients’ data by fine-tuning the model’s hyperparameters according to each client.

The experimental results indicate that the Fed-MLDL with Fed-RIME optimization exhibits the highest accuracy of 99.7% for 2-class, 99.5% for 8-class, and 99.3% for all 34 attack type classifications for independent and identically distributed datasets. Additionally, Fed-MLDL with Fed-FLA achieves the highest accuracy of 99.4% for 2-class, 99.3% for 8-class, and 99.1% for 34-class classification for non-independent and identically distributed datasets. Furthermore, the proposed Fed-MLDL with Fed-RIME optimization has demonstrated significant improvements in speed of convergence, stability, and client-specific customization in federated learning.

This study observes that coupling a Deep-Learning model with HPO techniques results in much faster convergence, requiring only 10-15 communication rounds. The proposed Fed-MLDL with Fed-RIME optimization outperforms existing state-of-the-art models on the CIC-IoT23 dataset.

## Repository Contents

- **FedMLDL.ipynb**: Jupyter Notebook containing:
  - CICIoT23 data preprocessing
  - Upsampling techniques
  - Implementation of a Multi-Layer Perceptron (MLP) model in a federated learning environment
  - Utilization of physics-based hyperparameter optimization techniques

## Requirements

To run the `FedMLDL.ipynb` notebook, you need the following libraries:

- Python 3.x
- NumPy
- Pandas
- Scikit-learn
- TensorFlow/Keras
- Matplotlib
- Torch
- Torchvision
- Mealpy

You can install the necessary packages using pip:

```bash
pip install -r requirements.txt
```
