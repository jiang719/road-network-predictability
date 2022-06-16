## Quantifying the Spatial Homogeneity of Urban Road Networks via Graph Neural Networks, Nature Machine Intelligence, 2022.
(Publication DOI: 10.1038/s42256-022-00462-y)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5866593.svg)](https://doi.org/10.5281/zenodo.5866593)

A graph neural network computing the intra-city and inter-city spatial homogeneity of urban road networks (URNs) 

## Introduction

* The spatial homogeneity of URNs measures the similarity of intersection connection styles between the subnetwork and the entire network. 
It captures the multi-hop node neighborhood relationships, and has promising applications in urban science, network science, and urban computing.
* This GitHub repository discusses a user-friendly approach to compute the network homogeneity of URNs worldwide. 
* URN classification, URN NI calculation, socioeconomic factor relation analysis, inter-city homogeneity analysis are also attached.  

## Publication

**Quantifying the Spatial Homogeneity of Urban Road Networks via Graph Neural Networks**
Jiawei Xue, Nan Jiang, Senwei Liang, Qiyuan Pang, Takahiro Yabe, Satish V Ukkusuri\*, Jianzhu Ma\*, March, 2022, Nature Machine Intelligence. 

## Requirements
* Window System
* Python 3.6
* NetworkX 2.1 
* OSMnx 0.11.4
* PyTorch 1.0 

## Directory Structure

* **data-collection**: Collect and preprocess the road network data for 30 cities in the US, Europe, and Asia. 
* **intra-city-network-homogeneity**: Perform the link prediction on URNs using 6 different encoders (such as relational GCN) and 1 decoder (DistMult) and compute F1 scores.
* **road-classification**: Implement the URN classification and discover its connections with F1 scores.
* **association-analysis**: Conduct the correlation analysis between F1 scores and social-economic factors, network topology metrics.
* **inter-city-network-homogeneity**: Get the inter-city homogeneity by learning URN features on city A and testing on city B.

## Results

<p align="center">
  <img src="https://github.com/jiang719/road-network-predictability/blob/master/main-figure/001.png" width="666">
</p>

## License
MIT license

