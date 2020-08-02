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
To see the ratio of different road types in a city, 
```
 python kmean_pca_analysis.py --mode city_ratio
```
