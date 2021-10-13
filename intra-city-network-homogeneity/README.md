# Road network prediction

This setion is the implementation of road network prediction. We try 6 different models to predict the missing roads based on the incomplete road network, including Node2vec, Struc2vec, GraphSage, Sprectral GCN, Relational GCN and Graph Attention Network. 

## Directory Structure
* models: implementation of 6 different road network prediction models
* trainer: code to train road network prediction models
* tester: code to test road network prediction models
* utils: code to for data loader and analyzing the best threshold
* figure1.defg_result: code to visualize the evaluation and comparison of different models
* shifted_result: RGCN models' result on the same cities but the city networks are shifted by a little distance.
* large_city_network_result: RGCN models' result on six cities with larger road network

## Training
* To train Node2vec + Distmult model
```
 cd trainer
 python node2vec_trainer.py
```
* To train Struc2vec + Distmult model
```
 python struc2vec_trainer.py
```
* To train Spectral gcn + Distmult model
```
 python spectral_gcn_trainer.py
```
* To train GraphSage + Distmult model
```
 python graph_sage_trainer.py
```
* To train Relational GCN + Distmult model
```
 python relational_gcn_trainer.py
```
* To train Graph Attention Network + Distmult model
```
 python gat_trainer.py
```
