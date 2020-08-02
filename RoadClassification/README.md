# Network Predictability and Road Network Classification

This part is the implementation of road network classification and predictability analysis. We aggregate 11 measures to extract the topology features from a road network. These measures consist of the node degree distribution, road circuity and dendricity. Based on these measures, we study the relation between

* Network predictability vs. network type
* Network predictability vs. principle component of aggregated measures

## Requirements
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
* We classify the road networks into 4 types using k-mean clustering, and visualize via PCA, and you can get the Fig. ()
```
 python kmean_pca_analysis.py --mode pca_visualize
```
* To see the ratio of different road types in a city, you can get the Fig. ()
```
 python kmean_pca_analysis.py --mode city_ratio
```
* To see the f1 value vs. road network types, you can get the Fig. ()
```
 python kmean_pca_analysis.py --mode f1_vs_type
```
