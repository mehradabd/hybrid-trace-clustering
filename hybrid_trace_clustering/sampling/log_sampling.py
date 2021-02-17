from pm4py.objects.log.log import EventLog
from pm4py.statistics.variants.log import get
from pm4py.statistics.traces.log import case_statistics
import random
from hybrid_trace_clustering.util.util import unique_activities


def variant_filter(log):
    new_log = EventLog()
    result = []
    variant_list = get.get_variants(log)
    variant_list_count = case_statistics.get_variant_statistics(log)
    sampled = random.sample(variant_list_count, 1000)
    vlist = [v['variant'] for v in variant_list_count]
    vlist_s = [v['variant'] for v in sampled]
    for v in vlist:
        if v in vlist_s:
            for trace in variant_list[v]:
                new_log.append(trace)

    new_len = len(case_statistics.get_variant_statistics(new_log))
    result.extend([new_len, len(new_log), len(unique_activities(new_log))])
    return new_log


def event_trace_sampling(log, k):
    new_log = EventLog()
    unique_ev = unique_activities(log)
    sampled = random.sample(unique_ev, k)
    for trace in log:
        valid = True
        eve_list = []
        for event in trace:
            if event['concept:name'] not in sampled:
                valid = False
                break
            eve_list.append(event['concept:name'])
        if valid:
            new_log.append(trace)
    print(f'lenght of sampled log is: {len(new_log)}')
    return new_log


def event_sampling(input_log, k):
    sampled_log = EventLog()
    sampled_log = input_log
    unique_event = unique_activities(sampled_log)
    sampled_events = random.sample(unique_event, k)
    for t in sampled_log:
        t[:] = [e for e in t if e['concept:name'] in sampled_events]
    sampled_log[:] = [t for t in sampled_log if len(t) != 0]
    print(f'lenght of sampled log is: {len(sampled_log)}')
    return sampled_log