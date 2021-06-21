# Hybrid Trace Clustering

This is the implementation of the master thesis conducted at Utrecht University in the fulfilment of the requirements Master of Science in Business Informatics. Master thesis can be reached at [Utrecht University thesis repository](/https://dspace.library.uu.nl/handle/1874/401960)

Hybrid trace clustering is a new approach to cluster traces (process instances) in order to derive more comprehensible models from an event log. Hybrid approach employs both similarity-based and model-driven methods to divide the original event log into homogeneous sub-logs.

The implementation is based on [PM4PY](/http://pm4py.org/), a python library that supports (state-of-the-art) process mining algorithms in python. This implementation is also compatible with version 2.2.1 of PM4PY. Use the following command to install required dependencies.

```
pip3 install -r requirements.txt
```


### Algorithm Inputs

- event_log: Original event log
- initial_f1_score: F1-score threshold that qualifies high quality clusters in the first step
- number_of_clusters: Number of clusters to be found
- minimum_cluster_size: Minimum number of traces in a cluster 
- neighbourhood_size: Number of neighbour(window) trace variants to be selected when building a new cluster
- distance_technique: The technique that calculates the distance between traces. Should be either 'BOA' or 'levenshtein'
- clustering_technique: Clustering technique to be used in the first step. Should be either 'K-means' or 'DBSCAN'
- discovery_technique: Process model discovery algorithm used in the algorithm. Should be either 'inductive miner' or 'heuristic miner'
- max_distance: The maximum distance two trace can have in order to be clustered together

### Algorithm Output
- A list of clusters(sub-logs) of the original event log

### Example

    import hybrid_trace_clustering
    import pm4py
    log = pm4py.xes_importer.apply("example_log.xes")
    clusters = hybrid_trace_clustering.algorithm.apply(log)
    
    