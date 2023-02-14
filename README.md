# 38-Clouds SVM

The repository includes classess of quantum- and classical-kernel based Support Vector Machines for cloud classification.
The data set used for classification is [38-Clouds data set](https://www.kaggle.com/datasets/sorour/38cloud-cloud-segmentation-in-satellite-images).

## How to use

In order to use the provided code, download the 38-Clouds data set and put the repository inside its main directory (The one that contains: '38-Clouds_test', '38-Clouds_training',... directories).

One can get aquainted to the code by running notebooks [run_classical_simulations](https://github.com/Quantum-Cosmos-Lab/38-Clouds-SVM/blob/main/run_simulation_classical.ipynb) and [run_quantum_simulations](https://github.com/Quantum-Cosmos-Lab/38-Clouds-SVM/blob/main/run_simulation_quantum.ipynb).

## The approach for classification

For computational reasons the classifiers do not predict every pixel on test images, they first perform superpixel (SLIC-based) segmentation on the test scenes, and then perform prediction for each superpixel.
With this approach, the training set we use, is not the original training set provided. 
It is a segmentation of the original training set with features obtained with the following statistical measures: mean, median, min, max, iqr, std for each spectral band of the image.
Our training set is contained in 'experiment_tr_data' directory.
