import string
import numpy as np
from pm4py.objects.log.util import get_log_representation as get_


def variant_to_string(variant1, variant2):
    var1 = []
    var2 = []
    variant1_separate = str()
    variant2_separate = str()

    for index, item in enumerate(variant1):
        if item != ',':
            variant1_separate = variant1_separate + item
        else:
            var1.append(variant1_separate)
            variant1_separate = str()
        if index == (len(variant1) - 1):
            var1.append(variant1_separate)

    for index, item in enumerate(variant2):
        if item != ',':
            variant2_separate = variant2_separate + item
        else:
            var2.append(variant2_separate)
            variant2_separate = str()
        if index == (len(variant2) - 1):
            var2.append(variant2_separate)

    listsum = sorted(list(set(var1 + var2)))
    alphabet = list(string.ascii_letters)[0:len(listsum)]

    str1 = [alphabet[listsum.index(item)] for item in var1]
    str2 = [alphabet[listsum.index(item)] for item in var2]
    str1 = ''.join(str1)
    str2 = ''.join(str2)

    return str1, str2

# ***********************************************************

def levenshtein(variant1, variant2):
    """
    Levenshtein distance function computes the syntactic distance between two trace variants, taken from following resource
    ***************************************************************************************/
    *    Title: Levenshtein Distance
    *    Author: AHMED FAWZY GAD
    *    Date: 20 March 2020
    *    Availability: https://blog.paperspace.com/implementing-levenshtein-distance-word-autocomplete-autocorrect/
    ***************************************************************************************/
    """

    variant1, variant2 = variant_to_string(variant1, variant2)

    distances = np.zeros((len(variant1) + 1, len(variant2) + 1))

    d1 = 0
    d2 = 0
    d3 = 0

    for v1 in range(len(variant1) + 1):
        distances[v1][0] = v1

    for v2 in range(len(variant2) + 1):
        distances[0][v2] = v2

    for v1 in range(1, len(variant1) + 1):
        for v2 in range(1, len(variant2) + 1):
            if (variant1[v1 - 1] == variant2[v2 - 1]):
                distances[v1][v2] = distances[v1 - 1][v2 - 1]
            else:
                d1 = distances[v1][v2 - 1]
                d2 = distances[v1 - 1][v2]
                d3 = distances[v1 - 1][v2 - 1]

                if (d1 <= d2 and d1 <= d3):
                    distances[v1][v2] = d1 + 1
                elif (d2 <= d1 and d2 <= d3):
                    distances[v1][v2] = d2 + 1
                else:
                    distances[v1][v2] = d3 + 1

    # print(distances, len(variant1), len(variant2))
    return distances[len(variant1)][len(variant2)]


# ***********************************************************

def bag_of_activities(trace, log):
    data = []
    dictionary = {}
    count = 0
    activity_frequency = {}
    feature_names = []
    values = []
    activities = get_.get_all_string_event_attribute_values(log, 'concept:name')
    for activity in activities:
        dictionary[activity] = count
        feature_names.append(activity)
        count = count + 1
    trace_rep = [0] * count
    for event in trace:
        values.append('event:concept:name@' + event['concept:name'])
    for value in values:
        if value in dictionary:
            if trace_rep[dictionary[value]] != 0:
                activity_frequency[value] = activity_frequency[value] + 1
                trace_rep[dictionary[value]] = activity_frequency[value]
            else:
                trace_rep[dictionary[value]] = 1
                activity_frequency[value] = 1
    data.append(trace_rep)
    data = np.asarray(data)
    return data

# ***********************************************************

def bag_of_activities_variant(variant, log):
    data = []
    dictionary = {}
    count = 0
    activity_frequency = {}
    feature_names = []
    values = []
    activities = get_.get_all_string_event_attribute_values(log, 'concept:name')
    for activity in activities:
        dictionary[activity] = count
        feature_names.append(activity)
        count = count + 1
    trace_rep = [0] * count
    for letter in variant:
        values.append('event:concept:name@' + letter)
    for value in values:
        if value in dictionary:
            if trace_rep[dictionary[value]] != 0:
                activity_frequency[value] = activity_frequency[value] + 1
                trace_rep[dictionary[value]] = activity_frequency[value]
            else:
                trace_rep[dictionary[value]] = 1
                activity_frequency[value] = 1
    data.append(trace_rep)
    data = np.asarray(data)
    return data