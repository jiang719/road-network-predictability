# Road network classification

This section is the implementation of road network classification and predictability analysis. We aggregate 11 measures to extract the topology features from a road network. These measures consist of the node degree distribution, road circuity and dendricity. Based on these measures, we study the relation between

* Network predictability vs. network types
* Network predictability vs. principle component of aggregated measures

## Feature extraction
* Extract the features of the road networks from training set,
```
 python measures.py --mode train
```
* Extract the features of the road networks from testing set,
```
 python measures.py --mode test
```

## Visualization

* To get F1 value VS cities, you can run the command
```
 python kmean_pca_analysis.py --mode f1_vs_city
```
* We classify the road networks into 4 types using k-mean clustering, and visualize them via PCA. The command is followed,
```
 python kmean_pca_analysis.py --mode pca_visualize
```
* To get the center point of each road type, you can run the command
```
 python kmean_pca_analysis.py --mode center
```
* To get the road type distribution in a city, you can run the command
```
 python kmean_pca_analysis.py --mode city_ratio
```
* To get the f1 value VS road network types, you can run the command
```
 python kmean_pca_analysis.py --mode f1_vs_type
```
* To get the f1 value VS PC1, you can run the command
```
 python kmean_pca_analysis.py --mode f1_vs_PCA1
```
