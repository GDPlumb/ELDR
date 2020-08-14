# ELDR
Explaining Low Dimensional Representations

A common workflow in data exploration is to learn a low-dimensional representation of the data, identify groups of points in that representation, and examine the differences between the groups to determine what they represent. 
We treat this as an interpretable machine learning problem by leveraging the model that learned the low-dimensional representation to help identify the key differences between the groups. 
To solve this problem, we introduce a new type of explanation, a Global Counterfactual Explanation (GCE), and our algorithm, Transitive Global Translations (TGT), for computing GCEs. 
TGT identifies the differences between each pair of groups using compressed sensing but constrains those pairwise differences to be consistent among all of the groups.
Empirically, we demonstrate that TGT is able to identify explanations that accurately explain the model while being relatively sparse, and that these explanations match real patterns in the data.


This repo contains an implementation of TGT as well as all of the code to reproduce the results in the [paper](https://proceedings.icml.cc/book/2020/hash/ccbd8ca962b80445df1f7f38c57759f0).  
