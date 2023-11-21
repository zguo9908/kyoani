import math

import numpy as np
from sympy.abc import x,y
from sympy import *
from scipy.stats import ttest_ind
from scipy import stats

def getOptimalTime(m,p,bg):
   # print(f'current adjusted backrgound is {bg}')
    f = (1 - exp(-(1/m)*(x-bg)))*p
    k = np.linspace(2, 30, 1600) # range of x
    ml1 = dict()
    for i in k:
        ml1[i]= f.subs(x,i)/i
   # print(max(ml1, key=ml1.get))
    optimal_time = max(ml1, key=ml1.get) -bg
    return optimal_time

def calculate_padded_averages_and_std(data):
    print(data)
    max_length = max(len(sublist) for sublist in data)
    averages = [sum(sublist) / len(sublist) if len(sublist) > 0 else 0 for sublist in data]
    padded_data = [
        [entry if entry != 0 else avg for entry in sublist] + [avg] * (max_length - len(sublist))
        for sublist, avg in zip(data, averages)
    ]
    averages = [sum(entry) / len(padded_data) for entry in zip(*padded_data)]

    std_deviation = np.std(padded_data, axis=0, ddof=0)

    return averages, std_deviation, padded_data

def compare_lists_with_significance(list1, list2, alpha=0.05, alpha_high=0.001):
    # Find the minimum length of sublists
    min_length = min(len(sublist1) for sublist1 in list1) if list1 else 0
    min_length = min(min_length, min(len(sublist2) for sublist2 in list2) if list2 else 0)
    significant_locations = {'low': [], 'high': []}

    # Iterate over the entries at the same positions in all sublists
    sublist_length = len(list1[0])
    for i in range(min_length):
        data1 = [sublist[i] for sublist in list1]
        data2 = [sublist[i] for sublist in list2]

        # Perform an independent t-test between the corresponding data points
        t_stat, p_value = stats.ttest_ind(data1, data2)

        # Check if the p-value is less than the significance level (alpha)
        if p_value < alpha_high:
            significant_locations['high'].append(i)
        elif p_value < alpha:
            significant_locations['low'].append(i)

    return significant_locations

def find_last_not_nan(lst):
    for item in reversed(lst):
        if not math.isnan(item):
            return item
    return None  # Return None if all elements are NaN

def merge_lists_with_sources(list1, list2):
    # Merging lists and marking the source
    return [(x, 'List 1') if not math.isnan(x) else (y, 'List 2') for x, y in zip(list1, list2)]