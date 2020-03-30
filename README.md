# DeepLearning_BP_6
bp算法 没有大量标记数据处理方法
# Lab 6 - Big Cat Recognition

## Information

* Course: Understanding Deep Neural Networks
* Teacher: Zhang Yi

## Files

* `lab6.m` - the MATLAB code file for the main procedure of this lab.
* `fc.m` - the MATLAB code file for feedforward computation.
* `bc.m` - the MATLAB code file for backward computation
* `lion.zip`, `tiger.zip`, `random_animal.zip` - the zip file of images.
    * Get from https://pan.baidu.com/s/1XSElxpw0tFPLkw8CXKvWnQ Code: `p2x4`

## Instructions

Implement forward computing and backward computing in `fc.m` and `bc.m`.
You can change the interface according to your program need.

Read images from each folder and prepare the dataset.
Each color image have 3 color channels, which leads to a height * width * channels matrix.
Resize each image to a certain size to match the input to your network.
Use all image in `random_animal` as unlabeled set.
Keep 20% of the images in `lion`/`tiger` as testing set, and the rest as training set.
(In fact, you don't need any template any more.)

0. Read and prepare the data
1. Train the autoencoder by using unlabeled data (the unlabeled set)
2. Remove the layers behind sparse representation layer after training
3. Form a new data set in sparse representation layer by using the labeled data set (the trianing set)
4. Form a new training data set for supervised network (the encoded training set and its labels)
5. Training the network by using the new training data set
6. combine the two networks
7. test the network with the testing set
