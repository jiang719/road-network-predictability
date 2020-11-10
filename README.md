## Quantifying spatial homogeneity of urban road networks via graph neural networks

A graph neural network computing the spatial homogeneity index of an urban road network(URN) 

## Introduction

* Spatial homogeneity of a URN measures the similarity of "network style" between the partial network and the whole network. 
It quantifies both the local and global structural properties of the URN, and has promising employments in congestion analysis, network optimization, 
geodatabase refinement, and urban transfer learning.
* This GitHub repository discusses a user-friendly approach to compute spatial homogeneity for URN worldwide. 
* URN classification, URN SNDi calculation, social-economic factor relation analysis, intercity homogeneity analysis are also attached.  

## Publication

**Quantifying spatial homogeneity of urban road networks via graph neural networks.**
*Jiawei Xue, Nan Jiang, Senwei Liang, Qiyuan Pang, Satish V Ukkusuri, Jianzhu Ma.* 
Submitted to Nature Communications, 2020. 

## Requirements
* Window System
* Python 3.6
* NetworkX 2.1 (A Python package)
* OSMnx 0.16.1 (A Python package)
* PyTorch 1.0 

## Directory Structure

* **data-collection**: Collect and preprocess the road network data for 30 cities in the United States, Europe, and Asia. 
* **road-prediction**: Perform the link prediction on URN using 6 different encoders (such as relational GCN) and 1 decoder (DistMult) and compute F1 scores.
* **road-classification**: Implement the URN classification and discover its connections with F1 scores.
* **association-analysis**: Conduct the correlation analysis between F1 scores and social-economic factors, network topology metrics.
* **intercity-homogeneity**: Get the inter-city homogeneity matrix by learning RNS features on city A and testing link prediction on city B.

## Results

<p align="center">
  <img src="https://github.com/jiang719/road-network-predictability/blob/master/main-figure/001.png" width="630" height="560">
</p>

## License
MIT license

