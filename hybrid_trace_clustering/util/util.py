from hybrid_trace_clustering.distance_calculation.dist_calc import bag_of_activities_variant
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA


def trace_to_string(trace):
    control_flow = []
    for event in trace:
        control_flow.append(event['concept:name'])
    trace_string = ','.join(control_flow)
    return trace_string


def unique_activities(event_log):
    activities = []
    for trace in event_log:
        for event in trace:
            activity = event['concept:name']
            if activity not in activities:
                activities.append(activity)
    return activities


def find_nearest_neighbours(log, variant, variant_count_list, variant_trace_list, number_of_neighbours):
    variant_list = []
    accepted_variants = []
    # accepted_variants.append(variant)

    for item in variant_count_list:
        variant_list.append(item['variant'])

    lenght_of_array = bag_of_activities_variant(variant_list[0], log).size
    vec = np.zeros([len(variant_count_list), lenght_of_array])

    for index, i in enumerate(variant_list):
        vec[index] = bag_of_activities_variant(i, log)

    pca = PCA(n_components=2)
    pca.fit(vec)
    pca.transform(vec)

    neigh = NearestNeighbors(n_neighbors=number_of_neighbours)
    neigh.fit(vec)

    print(neigh.kneighbors(bag_of_activities_variant(variant, log), return_distance=True))
    print(vec.shape)

    neighbours = neigh.kneighbors(bag_of_activities_variant(variant, log), return_distance=False)
    for j in range(3):
        accepted_variants.append(variant_list[neighbours[0][j]])
    """
    for v in accepted_variants:
        print(f'v is {v}')
        cc = variant_trace_list[v]
        print(len(cc))
        # for index, variant_instance in enumerate(variant_trace_list[variant_flow]):
        for index, variant_instance in enumerate(cc):
            cluster.append(variant_instance)
            log.remove(variant_instance)
    """
    return accepted_variants
