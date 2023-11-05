## Quantifying the Spatial Homogeneity of Urban Road Networks via Graph Neural Networks
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
Jiawei Xue, Nan Jiang, Senwei Liang, Qiyuan Pang, Takahiro Yabe, Satish V Ukkusuri\*, Jianzhu Ma\*, March 2022, Nature Machine Intelligence. 

## Journal/Media Coverage
**Nature Machine Intelligence**: https://www.nature.com/articles/s42256-022-00476-6

**Nature Computational Science**: https://www.nature.com/articles/s43588-022-00244-x

**Tech Xplore**: https://techxplore.com/news/2022-05-graph-neural-networks-spatial-homogeneity.html

**Peking University News**: https://news.pku.edu.cn/jxky/b7c965cbb640434ca109da42c94d7e39.htm

**Beijing University of Posts and Telecommunications**: https://lib.bupt.edu.cn/a/zuixingonggao/2022/0905/4240.html

## Requirements
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

## Methods
a. Description of spatial homogeneity.   
b. A road network.   © OpenStreetMap contributors.    
c. Message-passing between two layers in the graph neural network (GNN).   
d. Connecting strength S of a pair of nodes.   
e. We define the road network spatial homogeneity as the F1 score of the best GNN model with a well-tuned strength threshold δ.    



<p align="center">
  <img src="https://github.com/jiang719/road-network-predictability/blob/master/main-figure/001.png" width="666">
</p>

## Takeaway 1: the similarity between road networks in two cities.
* We compute the spatial homogeneity by training the GNN model on road networks in city A, and testing it on road networks in city B.
* We finally get 30*30=900 F1 scores for the following 30 cities.
* Each entry in the following matrix represents the directional similarity of road networks in two cities.
* Please go to the section [**Transfer learning reveals intercity similarity**](https://www.researchgate.net/publication/348169398_Quantifying_the_Spatial_Homogeneity_of_Urban_Road_Networks_via_Graph_Neural_Networks) in our paper for deep interpretation. 
* For those who want to use our similarity score in their studies in different fields, for example, 
  * **Transfer learning (computer science)**, refs [1],[2],
  * **Global road network analysis (urban science)**, refs [3],[4], 
  * **Global congestion analysis, accident analysis (transportation engineering)**, refs [5],[6],  
  * **Urban infrastructure evaluation (economics, sociology)**, refs [7],[8], please refer to [**takeaway-1/F1-30-30.txt**](https://github.com/jiang719/road-network-predictability/blob/master/takeaway-1/F1-30-30.txt) under this GitHub page to access these 900 values.  

Here,

| Index | Authors | Title | Publication |  
| :-----| :-----| :-----| :-----|
| 1 | Wei, Y., Zheng, Y., & Yang, Q.| Transfer knowledge between cities. | SIGKDD, 2016 |
| 2 | He, T., Bao, J., Li, R., Ruan, S., Li, Y., Song, L., ... & Zheng, Y.| What is the human mobility in a new city: Transfer mobility knowledge across cities. | The Web Conference, 2020 |
| 3 | Barrington-Leigh, C., & Millard-Ball, A.| Global trends toward urban street-network sprawl.| PNAS, 2020 |
| 4 | Burghardt, K., Uhl, J. H., Lerman, K., & Leyk, S.| Road network evolution in the urban and rural United States since 1900. | Computers, Environment and Urban Systems, 2022 |
| 5 | Çolak, S., Lima, A., & González, M. C.| Understanding congested travel in urban areas. | Nature Communications, 2016 |
| 6 | Thompson, J., Stevenson, M., Wijnands, J. S., Nice, K. A., Aschwanden, G. D., Silver, J., ... & Morrison, C. N.|  A global analysis of urban design types and road transport injury: an image processing study. | The Lancet Planetary Health, 2020|
| 7 | Bettencourt, L. M., Lobo, J., Helbing, D., Kühnert, C., & West, G. B.|Growth, innovation, scaling, and the pace of life in cities.|PNAS, 2007 |
| 8 | Arcaute, E., Hatna, E., Ferguson, P., Youn, H., Johansson, A., & Batty, M.|Constructing cities, deconstructing scaling laws.|Journal of the Royal Society Interface, 2015 |
    
<p align="center">
  <img src="https://github.com/jiang719/road-network-predictability/blob/master/main-figure/004_part.png" width="666">
</p>

## License
MIT license

