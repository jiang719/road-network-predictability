# Road Network Prediction

This setion is the implementation of road network prediction. We try 6 different models to predict the missing roads based on the incomplete road network, including Node2Vec, Struc2Vec, Graph-SAGE, SPrectral GCN, Relational GCN and Graph Attention Network. 

## Directory Structure
* GraphEmbedding: G


## Requirements
* Window system
* python 3.6
* networkx 2.1
* pytorch 1.0

## Model Comparison

After training 6 models for road network prediction, we test them on our test set. We use 2 metrics, true positive rate - false positive rate curve and precision recall curve, to evaluate and visualize the performance of each model. The results show that Relational GCN performs the best on road network prediction. Then we use 2 improvements (i.e. intersection and acute angle judgment) to get an improved Relational GCN.

<p align="center">
  <img src="https://github.com/jiang719/road-network-predictability/blob/master/RoadPrediction/figure1.b_c_result/figure1_b_total.png" width="450" height="400">
  <img src="https://github.com/jiang719/road-network-predictability/blob/master/RoadPrediction/figure1.b_c_result/figure1_c_total.png" width="450" height="400">
</p>
