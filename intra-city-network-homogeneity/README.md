# Road network prediction

This section pertains to the execution of road network prediction. We forecast the missing road segments within an incomplete road network by employing six distinct models, which include Node2vec, Struc2vec, Sprectral GCN, GraphSAGE, Graph Attention Network, and Relational GCN.

## Directory Structure
* models: implementation of the six road network prediction models
* trainer: code to train road network prediction models
* tester: code to test road network prediction models
* utils: code to for data loader and analyzing the best threshold
* figure1.defg_result: code to visualize the evaluation and comparison of different models
* shifted_result: Relational GCN model's result on the same cities where the urban networks are slightly displaced.
* large_city_network_result: Relational GCN model's result on the six cities with larger road networks (30*30km).

## Training
* To train Node2vec + DistMult
```
 cd trainer
 python node2vec_trainer.py
```
* To train Struc2vec + DistMult
```
 python struc2vec_trainer.py
```
* To train Spectral GCN + DistMult
```
 python spectral_gcn_trainer.py
```
* To train GraphSAGE + DistMult
```
 python graph_sage_trainer.py
```
* To train Graph Attention Network + DistMult
```
 python gat_trainer.py
```
* To train Relational GCN + DistMult
```
 python relational_gcn_trainer.py
```
