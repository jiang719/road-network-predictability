# Network Predictability and Road Network Classification

This part is the implementation of road network classification and predictability analysis. We aggregate 11 measures to extract the topology features from a road network. These measures consist of the node degree distribution, road circuity and dendricity. Based on these measures, we study the relation between

* Network predictability vs. network types
* Network predictability vs. principle component of aggregated measures

## Requirements
* Window system
* python 3.6
* networkx 2.1

## Feature extraction
* Extract the features of the road networks from training set,
```
 python measures.py --mode train
```
* Extract the features of the road networks from test set,
```
 python measures.py --mode test
```
The results will be saved in the directory 'RoadClassification/results'

## Visualization
We summarize the visualization results as below. 

<p align="center">
  <img src="https://github.com/jiang719/road-network-predictability/blob/master/RoadClassification/figures/combine.png" width="800" height="550">
</p>

* To get F1 value vs. cities as Fig. (1), you can run the command
```
 python kmean_pca_analysis.py --mode f1_vs_city
```
* We classify the road networks into 4 types using k-mean clustering, and visualize them via PCA as Fig. (2). The command is followed,
```
 python kmean_pca_analysis.py --mode pca_visualize
```
* To get the center point of each road type as Fig. (3), you can run the command
```
 python kmean_pca_analysis.py --mode center
```
* To get the road type distribution in a city as Fig. (4), you can run the command
```
 python kmean_pca_analysis.py --mode city_ratio
```
* To get the f1 value vs. road network types as Fig. (5), you can run the command
```
 python kmean_pca_analysis.py --mode f1_vs_type
```
* To get the f1 value vs. PCA1 as Fig. (6), you can run the command
```
 python kmean_pca_analysis.py --mode f1_vs_PCA1
```
