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
* To train Graph Attention Network + Distmult model
```
 python gat_trainer.py
```
* To train Relational GCN + DistMult
```
 python relational_gcn_trainer.py
```


## Reference
| Model | Authors | Publication | Venue |  
| :-----| :-----| :-----| :-----|
| Node2vec | Grover, A. and Leskovec, J. | node2vec: Scalable feature learning for networks. | SIGKDD, 2016 |
| Struc2vec | Ribeiro, L.F., Saverese, P.H. and Figueiredo, D.R. | struc2vec: Learning node representations from structural identity. | SIGKDD, 2017 |
| Spectral GCN | Kipf, T. N. and Welling, M. | Semi-supervised classifcation with graph convolutional networks. | ICLR, 2017 |
| GraphSAGE | Hamilton, W. L., Ying, R. and Leskovec, J. |  Inductive representation learning on large graphs. | NIPS, 2017 |
| Graph Attention Network | Velickovic, P., Cucurull, G., Casanova, A., Romero, A., Lio, P. and Bengio, Y.|  Graph attention networks. | ICLR, 2018 |
| Relational GCN | Schlichtkrull, M., Kipf, T.N., Bloem, P., Van Den Berg, R., Titov, I. and Welling, M. | Modeling relational data with graph convolutional networks. | The Semantic Web, ESWC 2018 |
| DistMult | Yang, B., Yih, W., He, X., Gao, J. and Deng, L. | Embedding entities and relations for learning and inference in knowledge bases. | ICLR, 2015 |

Readers can also refer to other GNN models summarized in the review: Zhou, J., Cui, G., Hu, S., Zhang, Z., Yang, C., Liu, Z., Wang, L., Li, C. and Sun, M., 2020. Graph neural networks: A review of methods and applications. AI Open.
