## Quantifying Spatial Homogeneity of Urban Road System via Predictability of Graph Neural Networks

A graph neural network framework to obtain the spatial homogeneity index of a road network system (RNS) 

## Introduction

* Spatial homogeneity of an RNS measures the consistency of "network style" across different regions of RNS. 
It quantifies both the local and global structural properties of RNS, and have promising employments in congestion analysis, network optimization, 
urban planning evaluation, geodatabase refinement.
* This GitHub page discusses a user-friendly approach to compute spatial homogeneity for RNS worldwide. 
* RNS classification, RNS SNDI calculation, social-economic factors relation analysis are also attached.  

## Publication

**Quantifying Spatial Homogeneity of Urban Road System via Predictability of Graph Neural Networks.**
*Jiawei Xue, Nan Jiang, Senwei Liang, Qiyuan Pang, Satish V Ukkusuri, Jianzhu Ma.* 
Submitted to Nature Communications, 2020. 

## Requirements
* Window system
* Python 3.6
* networkx 2.1 (Python package)
* osmnx 0.16.1 (Python package)

## Directory Structure

* **data-collection**: collect and preprocess the road network data for 30 cities in USA, Europe, and Asia. 
* **road-prediction**: perform the link prediction on RNS using 6 different encoders (such as relational GCN) and 1 decoder (DistMult) and get F1 scores.
* **road-classification, SDNiDisconnected_0716data, SDNiDisconnectednessIndex**: implement the RNS classification and discover connection with F1 scores.
* **association-analysis**: conduct the correlation analysis between F1 scores and social-economic factors, network topology metrics.
* **intercity-homogeneity**: learn RNS features on city A and test link prediction on city B, get the asymmetric intercity homogeneity. 

## Demo
Training and testing.

<p align="center">
  <img src="https://github.com/jiang719/road-network-predictability/blob/master/mainFigures/001.png" width="450" height="400">
</p>

Road classification.

<p align="center">
  <img src="https://github.com/jiang719/road-network-predictability/blob/master/mainFigures/002.png" width="450" height="400">
</p>

Association analysis.

<p align="center">
  <img src="https://github.com/jiang719/road-network-predictability/blob/master/mainFigures/003.png" width="450" height="400">
</p>

Intercity homogeneity.

<p align="center">
  <img src="https://github.com/jiang719/road-network-predictability/blob/master/mainFigures/004.png" width="450" height="400">
</p>
## Requirement

We may refer to https://github.com/idekerlab/DCell/ to learn how to write the README file.
