# GM-Net
This is some basic code for paper ACPR2017 GM-Net: Learning Features with More Efficiency(https://arxiv.org/abs/1706.06792)
## Abstract
Deep Convolutional Neural Networks (CNNs) are capable of learning unprecedentedly effective features from images. 
Some researchers have struggled to enhance the parameters' efficiency using grouped convolution. 
However, the relation between the optimal number of convolutional groups and the recognition performance remains an open problem. 
In this paper, we propose a series of Basic Units (BUs) and a two-level merging strategy to construct deep CNNs, 
referred to as a joint Grouped Merging Net (GM-Net), 
which can produce joint grouped and reused deep features while maintaining the feature discriminability for classification tasks.
Our GM-Net architectures with the proposed BU_A (dense connection) and BU_B (straight mapping) lead to significant 
reduction in the number of network parameters and obtain performance improvement in image classification tasks. 
Extensive experiments are conducted to validate the superior performance of the GM-Net than the state-of-the-arts on the 
benchmark datasets, e.g., MNIST, CIFAR-10, CIFAR-100 and SVHN.

## Architecture
![architecture](https://github.com/IssacCyj/GM-Net/blob/master/arch.png)

## Results
![res](https://github.com/IssacCyj/GM-Net/blob/master/res.png)
