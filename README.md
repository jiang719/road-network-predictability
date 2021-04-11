## Urban road network homogeneity reveals maturity and inheritance of cities

A graph neural network computing the intra-city and inter-city network homogeneity of an urban road network(URN) 

## Introduction

* The network homogeneity of URNs measures the similarity of intersection connectition styles between the partial network and the whole network. 
It captures the multi-hop node neighborhood relationships, and has promising applications in urban science, network science, and urban computing.
* This GitHub repository discusses a user-friendly approach to compute the network homogeneity of URNs worldwide. 
* URN classification, URN SNDi calculation, social-economic factor relation analysis, inter-city homogeneity analysis are also attached.  

## Publication

**Urban road network homogeneity reveals maturity and inheritance of cities**
*Jiawei Xue, Nan Jiang, Senwei Liang, Qiyuan Pang, Takahiro Yabe, Satish V Ukkusuri, Jianzhu Ma*, 2021. 

## Requirements
* Window System
* Python 3.6
* NetworkX 2.1 
* OSMnx 0.16.1
* PyTorch 1.0 

## Directory Structure

* **data-collection**: Collect and preprocess the road network data for 30 cities in the US, Europe, and Asia. 
* **road-prediction**: Perform the link prediction on URNs using 6 different encoders (such as relational GCN) and 1 decoder (DistMult) and compute F1 scores.
* **road-classification**: Implement the URN classification and discover its connections with F1 scores.
* **association-analysis**: Conduct the correlation analysis between F1 scores and social-economic factors, network topology metrics.
* **intercity-homogeneity**: Get the inter-city homogeneity by learning URN features on city A and testing on city B.

## Results

<p align="center">
  <img src="https://github.com/jiang719/road-network-predictability/blob/master/main-figure/001.png" width="630" height="560">
</p>

## License
MIT license

