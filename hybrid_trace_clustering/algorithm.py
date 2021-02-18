from scipy.spatial import distance
import string
from pm4py.statistics.traces.log import case_statistics
from pm4py.statistics.variants.log import get
from hybrid_trace_clustering.clustering_algo import dbscan_algo as dbscan_clustering
from hybrid_trace_clustering.clustering_algo import kmeans_algo as kmeans_clustering
from hybrid_trace_clustering.util.evaluation import cluster_evaluation
from hybrid_trace_clustering.distance_calculation.dist_calc import bag_of_activities, levenshtein
from pm4py.objects.log.log import EventLog as EvenLogClass

DEFAULT_DISCOVERY_TECHNIQUE = 'inductive miner'
DEFAULT_CLUSTERING_TECHNIQUE = 'K-means'
DEFAULT_NUMBER_OF_CLUSTERS = 4
DEFAULT_MAX_DISTANCE = 2
DEFAULT_DISTANCE_TECHNIQUE = 'levenshtein'
DEFAULT_F1_SCORE_THRESHOLD = 0.95
DEFAULT_MINIMUM_CLUSTER_SIZE = 300
DEFAULT_NEIGHBORHOOD_SIZE = 10


class EventLog(EvenLogClass):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def remove(self, x):
        self._list.remove(x)


def apply(event_log,
          initial_f1_score=DEFAULT_F1_SCORE_THRESHOLD,
          number_of_clusters=DEFAULT_NUMBER_OF_CLUSTERS,
          minimum_cluster_size=DEFAULT_MINIMUM_CLUSTER_SIZE,
          neighbourhood_size=DEFAULT_NEIGHBORHOOD_SIZE,
          distance_technique=DEFAULT_DISTANCE_TECHNIQUE,
          clustering_technique=DEFAULT_CLUSTERING_TECHNIQUE,
          discovery_technique=DEFAULT_DISCOVERY_TECHNIQUE,
          max_distance=DEFAULT_MAX_DISTANCE):
    """
    Apply hybrid trace clustering algorithm on an event log

    Parameters
    ----------
    event_log
        Original event log

    initial_f1_score
        F1-score threshold that qualifies high quality clusters in the first step

    number_of_clusters
        Number of clusters to be found

    minimum_cluster_size
        Minimum size of a cluster

    neighbourhood_size
        Number of neighbour trace variants to be selected when building a new cluster

    distance_technique
        The technique that calculates the distance between traces. Should be either 'BOA' or 'levenshtein'

    clustering_technique
        Clustering technique to be used in the first step. Should be either 'K-means' or 'DBSCAN'

    discovery_technique
        Process model discovery algorithm used in the algorithm. Should be either 'inductive miner' or 'heuristic miner'

    max_distance
        The maximum distance two trace can have in order to be clustered together

    Returns
    -------
    A list of event log clusters

    """

    finalized_clusters = []
    remaining_logs = EventLog()
    filtered_log = EventLog()

    if clustering_technique == 'DBSCAN':
        dbscan_parameters = {"dbscan_eps": 0.75}
        clusters = dbscan_clustering.apply(event_log, dbscan_parameters)

    if clustering_technique == 'K-means':
        clusters = kmeans_clustering.apply(event_log, number_of_clusters)

    i = 0
    flag_list = list(string.ascii_letters)[0:len(clusters)]

    print(f'number of clusters: {len(clusters)} \nsize of clusters: {[len(x) for x in clusters]}')

    for sublog in clusters:

        # net, im, fm = heuristics_miner.apply(sublog, parameters={"dependency_thresh": 0.99})

        # fitness = replay_fitness_evaluator.apply(sublog, net, im, fm,
        #                                         variant=replay_fitness_evaluator.Variants.TOKEN_BASED)
        # precision = precision_evaluator.apply(sublog, net, im, fm,
        #                                      variant=precision_evaluator.Variants.ETCONFORMANCE_TOKEN)
        # f1_score = 2 * (fitness["log_fitness"] * precision) / (fitness["log_fitness"] + precision)
        eval = cluster_evaluation(sublog, discovery_technique)
        fitness = eval[0]
        precision = eval[1]
        f1_score = eval[2]

        print(f'******** Cluster {clusters.index(sublog)} with size {len(sublog)} ********')
        print(f'Token-based Fitness is: {fitness}')
        print(f'Token-based precision is: {precision}')
        print(f'Token-based F1-score is: {f1_score}')

        if len(finalized_clusters) == number_of_clusters - 1:
            for trace in sublog:
                remaining_logs.append(trace)
        else:
            if f1_score >= initial_f1_score and len(sublog) > minimum_cluster_size:
                finalized_clusters.append(sublog)
            else:
                for trace in sublog:
                    trace.flag = flag_list[i]
                    remaining_logs.append(trace)
                i += 1

    if len(finalized_clusters) == number_of_clusters:
        return finalized_clusters

    if len(finalized_clusters) < number_of_clusters - 1:
        cluster_initialization, filtered_log = new_cluster(remaining_logs, neighbourhood_size, minimum_cluster_size,
                                                           distance_technique, discovery_technique, max_distance)
        finalized_clusters.append(cluster_initialization)

    while len(finalized_clusters) < number_of_clusters - 1:
        cluster_new, filtered_log = new_cluster(filtered_log, neighbourhood_size, minimum_cluster_size,
                                                distance_technique, discovery_technique, max_distance)
        finalized_clusters.append(cluster_new)
        print(f' lenght of cluster so far is: {len(finalized_clusters)}')

    if len(finalized_clusters) == (number_of_clusters - 1):  # and len(filtered_log) != 0:
        """ To distribute remaining traces among clusters
        for index, remained_trace in enumerate(filtered_log):
            finalized_clusters[index % number_of_clusters].append(remained_trace)
        """
        if len(filtered_log) == 0:
            filtered_log = remaining_logs
        print("Trash cluster created")
        finalized_clusters.append(filtered_log)  # Building a trash cluster for remaining traces

    return finalized_clusters


# ***********************************************************
def new_cluster(log, neighbourhood_size, minimum_cluster_size, distance_technique, discovery_technique, max_distance):

    print('***********New cluster initialization starts!*********\n')
    iteration = 0
    f1_score = 0
    if f1_score == 0:
        cluster = EventLog()
        # if iteration == 0:
        #    variants_count_list = case_statistics.get_variant_statistics(log)
        # else:
        #    variants_count_list = case_statistics.get_variant_statistics(log)
        #    random.shuffle(variants_count_list)
        variants_count_list = case_statistics.get_variant_statistics(log)
        variant_list = get.get_variants(log)
        frequent = variants_count_list[0]['variant']
        frequent_flag = variant_list[frequent][0].flag
        print(f'The most frequent variant is: {frequent} with flag: {frequent_flag}')

        """ Building a cluster using KNN (optional)

        neighbour_variants = find_nearest_neighbours(log, frequent, variants_count_list,
                                                     variant_list, neighbourhood_size)
        for neighbour in neighbour_variants:
            trace_list = variant_list[neighbour]
            print(len(trace_list))
            for index, variant_trace in enumerate(trace_list):
                cluster.append(variant_trace)
                log.remove(variant_trace)

        """

        # """ Building a cluster using the most frequent variants
        for trace in variant_list[frequent]:
            cluster.append(trace)
            log.remove(trace)
        # log = EventLog(filter(lambda x: x not in cluster, log))

        for neighbourhood, variant in enumerate(variants_count_list):
            if neighbourhood == 0:
                continue
            if neighbourhood < neighbourhood_size:
                variant_flow = variant['variant']
                neighbour_trace = variant_list[variant_flow][0]
                print("********** Flags! ************ ")
                print(neighbour_trace.flag)
                print(frequent_flag)
                if neighbour_trace.flag != frequent_flag:
                    if distance_technique == 'BOA':
                        frequent_trace = variant_list[frequent][0]
                        # neighbour_trace = variant_list[variant_flow][0]
                        similarity_distance = distance.euclidean(bag_of_activities(frequent_trace, log),
                                                                 bag_of_activities(neighbour_trace, log))

                    if distance_technique == 'levenshtein':
                        similarity_distance = levenshtein(frequent, variant_flow)

                    print(f'Distance with {variant_flow} is: {similarity_distance}')
                    if similarity_distance <= max_distance:
                        for trace in variant_list[variant_flow]:
                            cluster.append(trace)
                            log.remove(trace)
                    # log = EventLog(filter(lambda x: x not in cluster, log))
            else:
                break
                # """
        print(f'length of cluster: {len(cluster)}, log: {len(log)}')
        # net, im, fm = heuristics_miner.apply(cluster, parameters={"dependency_thresh": 0.99})

        # fitness = replay_fitness_evaluator.apply(cluster, net, im, fm,
        #                                     variant=replay_fitness_evaluator.Variants.TOKEN_BASED)
        # precision = precision_evaluator.apply(cluster, net, im, fm,
        #                                  variant=precision_evaluator.Variants.ETCONFORMANCE_TOKEN)
        # f1_score = 2 * (fitness["log_fitness"] * precision) / (fitness["log_fitness"] + precision)

        eval = cluster_evaluation(cluster, discovery_technique)
        fitness = eval[0]
        precision = eval[1]
        f1_score = eval[2]
        print(f'f1-score is: {f1_score}')
        iteration += 1

    trace_distribution(cluster, log, minimum_cluster_size, discovery_technique, f1_score)
    return cluster, log


# ***********************************************************

def trace_distribution(cluster, log, minimum_cluster_size, discovery_technique, score):
    print('***********Trace Distribution Starts!*********\n')
    print(f'length of cluster: {len(cluster)}, log: {len(log)}')

    variants_count_list = case_statistics.get_variant_statistics(log)
    # variants_count_list_sampled = sample(variants_count_list, int(len(variants_count_list) / 4))
    # variants_count_list = variants_count_list_sampled
    variant_trace_list = get.get_variants(log)

    # if discovery_technique == 'heuristic miner':
    #    net, im, fm = heuristics_miner.apply(cluster, parameters={"dependency_thresh": 0.99})

    # if discovery_technique == 'inductive miner':
    #    net, im, fm = inductive_miner.apply(cluster)

    # initial_fitness = replay_fitness_evaluator.apply(cluster, net, im, fm,
    #                                                 variant=replay_fitness_evaluator.Variants.TOKEN_BASED)
    # initial_precision = precision_evaluator.apply(cluster, net, im, fm,
    #                                              variant=precision_evaluator.Variants.ETCONFORMANCE_TOKEN)

    # current_f1_score_initial = 2 * (initial_fitness["log_fitness"] * initial_precision) / (
    #        initial_fitness["log_fitness"] + initial_precision)

    current_f1_score = score
    print(f'initial f1 is: {current_f1_score}')
    for variant in variants_count_list:
        variant_flow = variant['variant']
        trace = variant_trace_list[variant_flow][0]

        cluster.append(trace)
        # net, im, fm = inductive_miner.apply(cluster)
        # new_fitness = replay_fitness_evaluator.apply(cluster, net, im, fm,
        #                                             variant=replay_fitness_evaluator.Variants.TOKEN_BASED)
        # new_precision = precision_evaluator.apply(cluster, net, im, fm,
        #                                          variant=precision_evaluator.Variants.ETCONFORMANCE_TOKEN)
        # new_f1_score = 2 * (new_fitness["log_fitness"] * new_precision) / (new_fitness["log_fitness"] + new_precision)
        eval = cluster_evaluation(cluster, discovery_technique)
        new_f1_score = eval[2]
        # print(f'new fitness is: {initial_fitness}')
        print(f'current f1-score is: {current_f1_score}')
        print(f'new f1 is: {new_f1_score}')
        if current_f1_score <= new_f1_score:
            print(f'*****Improved the model!****: {trace}')
            cluster.remove(trace)

            """ Optional Use of KNN to find neighbours of qualified variant 

            temp_variant_list = case_statistics.get_variant_statistics(log)
            temp_variant_trace_list = get.get_variants(log)

            neighbour_variants = find_nearest_neighbours(log, variant_flow, temp_variant_list,
                                                         temp_variant_trace_list, 5)

            for v in neighbour_variants:
                print(f'v is {v}')
                cc = temp_variant_trace_list[v]
                print(len(cc))
            # for index, variant_instance in enumerate(variant_trace_list[variant_flow]):
                for index, variant_instance in enumerate(cc):
                    cluster.append(variant_instance)
                    log.remove(variant_instance)
            print(f'length of cluster: {len(cluster)}, log: {len(log)}')  # , sample: {len(sampled_log)}')
            net, im, fm = inductive_miner.apply(cluster)
            new_fitness = replay_fitness_evaluator.apply(cluster, net, im, fm,
                                                         variant=replay_fitness_evaluator.Variants.TOKEN_BASED)
            new_precision = precision_evaluator.apply(cluster, net, im, fm,
                                                      variant=precision_evaluator.Variants.ETCONFORMANCE_TOKEN)
            new_f1_score = 2 * (new_fitness["log_fitness"] * new_precision) / (
                        new_fitness["log_fitness"] + new_precision)
            current_f1_score = new_f1_score
            continue
            """

            # """ Without KNN
            current_f1_score = new_f1_score
            for variant_instance in variant_trace_list[variant_flow]:
                cluster.append(variant_instance)
                log.remove(variant_instance)
            print(f'length of cluster: {len(cluster)}, log: {len(log)}')
            continue
            # """

        if current_f1_score > new_f1_score:
            if new_f1_score >= 0.9 and abs(current_f1_score - new_f1_score) <= 0.05:
                print(f'$$$$Did not improve the model but a close trace!$$$$: {trace}')
                current_f1_score = new_f1_score
                for index, variant_instance in enumerate(variant_trace_list[variant_flow]):
                    if index > 0:
                        cluster.append(variant_instance)
                        log.remove(variant_instance)
                print(f'length of cluster: {len(cluster)}, log: {len(log)}')
                # continue
            else:
                # variants_count_list.remove(variant)
                cluster.remove(trace)
                if len(cluster) >= minimum_cluster_size:
                    print(f'did not improve the model and enough traces!: {trace}')
                    final_cluster = cluster
                    break
                else:
                    print(f'did not improve the model and not enough traces!: {trace}')
                    print(f'length of cluster: {len(cluster)}, log: {len(log)}')
                    # continue

    print(f'length of cluster: {len(cluster)}, log: {len(log)}')
