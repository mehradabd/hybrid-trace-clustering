import os
from pm4py.objects.log.importer.xes import importer as xes_importer
# from ClusterFlags import bag_of_activities
from pm4py.objects.log.util import get_log_representation as get_
from sklearn.decomposition import PCA
from pm4py.objects.log.log import EventLog, Trace
import matplotlib.pyplot as plt
# from kneed import KneeLocator
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def apply(log, number_of_clusters):
    kmeans = KMeans(
        init="random",
        n_clusters=number_of_clusters,
        n_init=10,
        max_iter=300,
        random_state=42
    )

    data, feature_names = get_.get_representation(log, str_ev_attr=['concept:name'], str_tr_attr=[],
                                                  num_ev_attr=[], num_tr_attr=[], str_evsucc_attr=[])

    pca = PCA(n_components=3)
    pca.fit(data)
    data3d = pca.transform(data)
    km = kmeans.fit(data3d)

    already_seen = {}
    labels = km.labels_
    cluster_list = []

    for i in range(len(log)):
        if not labels[i] in already_seen:
            already_seen[labels[i]] = len(list(already_seen.keys()))
            cluster_list.append(EventLog())
        trace = log[i]
        cluster_list[already_seen[labels[i]]].append(trace)

    return cluster_list
