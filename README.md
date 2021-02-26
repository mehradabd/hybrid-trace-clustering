# Hybrid Trace Clustering

This is the implementation of the master thesis conducted at Utrecht University in the fulfilment of the requirements Master of Science in Business Informatics.

Hybrid trace clustering is a new approach to cluster traces (process instances) in order to derive more comprehensible models from an event log. Hybrid approach employs both similarity-based and model-driven methods to divide the original event log into homogeneous sub-logs.

The implementation is based on pm4py, a python library that supports (state-of-the-art) process mining algorithms in python. More info available on http://pm4py.org/


### Example

    import hybrid_trace_clustering
    import pm4py
    log = pm4py.xes_importer.apply("example_log.xes")
    clusters = hybrid_trace_clustering.algorithm.apply(log)
    
    