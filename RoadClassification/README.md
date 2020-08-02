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
* To see F1 value vs. cities, and you can get the Fig. (1)
```
 python kmean_pca_analysis.py --mode f1_vs_city
```
* We classify the road networks into 4 types using k-mean clustering, and visualize via PCA, and you can get the Fig. (2)
```
 python kmean_pca_analysis.py --mode pca_visualize
```
* To see the center point of each road type, refer to Fig. (3)
```
 python kmean_pca_analysis.py --mode center
```
* To see the road type distribution in a city, refer Fig. (4)
```
 python kmean_pca_analysis.py --mode city_ratio
```
* To find out the f1 value vs. road network types, you can get the Fig. (5)
```
 python kmean_pca_analysis.py --mode f1_vs_type
```
* To see the f1 value vs. PCA1, you can get the Fig. (6)
```
 python kmean_pca_analysis.py --mode f1_vs_PCA1
```