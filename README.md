## Quantifying the Spatial Homogeneity of Urban Road Networks via Graph Neural Networks
(Publication DOI: 10.1038/s42256-022-00462-y)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5866593.svg)](https://doi.org/10.5281/zenodo.5866593)

A graph neural network approach that calculates the intra-city and inter-city spatial homogeneity of urban road networks (URNs) 

## Introduction

* The spatial homogeneity of URNs measures the similarity of intersection connection patterns between the subnetwork and the entire network. 
It captures the multi-hop node neighborhood relationships, and holds potential for applications in urban science, network science, and urban computing.
* This GitHub repository presents a user-friendly method for quantifying the network homogeneity of URNs on a global scale. 
* Additionally, URN classification, URN network irregularity (NI) computation, analysis of socioeconomic factors, and inter-city homogeneity analysis are also incorporated.  

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

* **data-collection**: Collect and preprocess road network data for 30 cities across the United States, Europe, and Asia. 
* **intra-city-network-homogeneity**: Conduct link prediction on URNs by utilizing six distinct encoders, including relational GCN, and a decoder known as DistMult, followed by the computation of F1 scores.
* **road-classification**: Execute URN classification and discover its correlations with F1 scores.
* **association-analysis**: Perform a correlation analysis between F1 scores and socioeconomic factors as well as network topology metrics.
* **inter-city-network-homogeneity**: Obtain inter-city homogeneity by training graph neural network (GNN) models on city A and subsequently testing them on city B.

## Methods
a. Description of spatial homogeneity.   
b. A road network near 40.71798°N, 74.00053°W in New York City. © OpenStreetMap contributors.    
c. Message-passing mechanism between adjacent layers in the GNN.   
d. Connecting strength S of a pair of nodes.   
e. We define the road network spatial homogeneity as the F1 score of the best GNN model with a well-tuned connecting strength threshold δ.    

<p align="center">
  <img src="https://github.com/jiang719/road-network-predictability/blob/master/main-figure/001.png" width="666">
</p>

## Takeaway 1: the similarity between road networks in two cities
* We compute the spatial homogeneity by training the GNN model on road networks in city A, and testing it on road networks in city B.
* We ultimately gain 30*30=900 F1 scores for the following 30 cities.
* Each entry in the following 30*30 matrix represents the directional similarity of road networks in two cities.
* Please refer to the section [**Transfer learning reveals intercity similarity**](https://www.researchgate.net/publication/348169398_Quantifying_the_Spatial_Homogeneity_of_Urban_Road_Networks_via_Graph_Neural_Networks) in our paper.
  
<p align="center">
  <img src="https://github.com/jiang719/road-network-predictability/blob/master/main-figure/004_part.png" width="666">
</p>

* For those interested in applying our homogeneity score in their research across various domains, such as, 
  * **Transfer learning (computer science)**, refs [1],[2],
  * **Global road network analysis (urban science)**, refs [3],[4], 
  * **Global congestion analysis, accident analysis (transportation engineering)**, refs [5],[6],  
  * **Urban infrastructure evaluation (economics, sociology)**, refs [7],[8], please refer to [**takeaway-1/F1-30-30.txt**](https://github.com/jiang719/road-network-predictability/blob/master/takeaway-1/F1-30-30.txt) under this GitHub page to access these 30*30=900 values.  

with

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
    

## Reference
| Model | Authors | Publication | Venue |  
| :-----| :-----| :-----| :-----|
| Node2vec | Grover, A. and Leskovec, J. | node2vec: Scalable feature learning for networks. | SIGKDD, 2016 |
| Struc2vec | Ribeiro, L.F., Saverese, P.H. and Figueiredo, D.R. | struc2vec: Learning node representations from structural identity. | SIGKDD, 2017 |
| Spectral GCN | Kipf, T. N. and Welling, M. | Semi-supervised classification with graph convolutional networks. | ICLR, 2017 |
| GraphSAGE | Hamilton, W. L., Ying, R. and Leskovec, J. |  Inductive representation learning on large graphs. | NIPS, 2017 |
| Graph Attention Network | Velickovic, P., Cucurull, G., Casanova, A., Romero, A., Lio, P. and Bengio, Y.|  Graph attention networks. | ICLR, 2018 |
| Relational GCN | Schlichtkrull, M., Kipf, T.N., Bloem, P., Van Den Berg, R., Titov, I. and Welling, M. | Modeling relational data with graph convolutional networks. | The Semantic Web, ESWC 2018 |
| DistMult | Yang, B., Yih, W., He, X., Gao, J. and Deng, L. | Embedding entities and relations for learning and inference in knowledge bases. | ICLR, 2015 |
| Review | Zhou, J., Cui, G., Hu, S., Zhang, Z., Yang, C., Liu, Z., Wang, L., Li, C. and Sun, M.| Graph neural networks: A review of methods and applications. | AI Open, 2020 |

## License
MIT license

