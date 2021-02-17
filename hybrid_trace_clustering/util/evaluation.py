from pm4py.algo.discovery.heuristics import algorithm as heuristics_miner
from pm4py.evaluation.replay_fitness import evaluator as replay_fitness_evaluator
from pm4py.evaluation.precision import evaluator as precision_evaluator
from pm4py.algo.discovery.inductive import algorithm as inductive_miner

def cluster_evaluation(sublog, discovery_technique):
    if discovery_technique == 'heuristic miner':
        net, im, fm = heuristics_miner.apply(sublog, parameters={"dependency_thresh": 0.99})

    if discovery_technique == 'inductive miner':
        net, im, fm = inductive_miner.apply(sublog)

    fitness = replay_fitness_evaluator.apply(sublog, net, im, fm,
                                             variant=replay_fitness_evaluator.Variants.TOKEN_BASED)
    precision = precision_evaluator.apply(sublog, net, im, fm,
                                          variant=precision_evaluator.Variants.ETCONFORMANCE_TOKEN)
    f1_score = 2 * (fitness["log_fitness"] * precision) / (fitness["log_fitness"] + precision)

    return fitness, precision, f1_score


def alignment_based_final_cluster_evaluation(clusters):
    f1_scores = {}
    weighted_sum_f1 = 0
    weighted_sum_fitness = 0
    weighted_sum_precision = 0
    cluster_lenghts = []
    lenghts = 0
    for index, cluster in enumerate(clusters):
        net, im, fm = inductive_miner.apply(cluster)
        fitness = replay_fitness_evaluator.apply(cluster, net, im, fm,
                                                 variant=replay_fitness_evaluator.Variants.ALIGNMENT_BASED)
        precision = precision_evaluator.apply(cluster, net, im, fm,
                                              variant=precision_evaluator.Variants.ALIGN_ETCONFORMANCE)
        f1_score = 2 * (fitness["averageFitness"] * precision) / (fitness["averageFitness"] + precision)
        f1_scores['cluster ' + str(len(cluster))] = f1_score
        cluster_lenghts.append(len(cluster))

        weighted_sum_f1 += len(cluster) * f1_score
        weighted_sum_fitness += len(cluster) * fitness["averageFitness"]
        weighted_sum_precision += len(cluster) * precision
        lenghts += len(cluster)

    if len(cluster_lenghts) < 4:
        cluster_lenghts.append(0)
    if len(cluster_lenghts) < 5:
        cluster_lenghts.append(0)

    weighted_average_fitness = weighted_sum_fitness / lenghts
    weighted_average_precision = weighted_sum_precision / lenghts
    weighted_average_f1 = weighted_sum_f1 / lenghts

    return weighted_average_f1, weighted_average_fitness, weighted_average_precision, f1_scores, cluster_lenghts