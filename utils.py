import math
import os

import numpy as np
from sympy.abc import x,y
from sympy import *
from scipy.stats import ttest_ind, expon
from scipy import stats

def get_optimal_time(m,p,bg):
   # print(f'current adjusted backrgound is {bg}')
    f = (1 - exp(-(1/m)*(x-bg)))*p
    k = np.linspace(bg, 30, 1600) # range of x
    ml1 = dict()
    for i in k:
        total_time = i
        ml1[i]= f.subs(x,i)/total_time
   # print(max(ml1, key=ml1.get))
    optimal_time = max(ml1, key=ml1.get) -bg
    return optimal_time

def get_optimal_time_expon(time_array, m,p,bg):
    reward_pdf = expon.pdf(time_array, 0, m)*p
    reward_cdf = expon.cdf(time_array, 0, m)*p

    ps_reward_rate = np.zeros(len(time_array))
    gb_reward_rate = np.zeros(len(time_array))
    total_time = bg + time_array
    for i in range(0, len(time_array)):
        t = time_array[i]
        prob_rewarded_before_t = reward_cdf[i]

        ps_reward_rate[i] = prob_rewarded_before_t / t
        gb_reward_rate[i] = prob_rewarded_before_t / total_time[i]
    optimal_giveup_index_rou_g = np.argmax(gb_reward_rate)
    optimal_giveup_time_rou_g = time_array[optimal_giveup_index_rou_g]
    optimal_giveup_index_rou_l = np.argmax(ps_reward_rate)
    optimal_giveup_time_rou_l = time_array[optimal_giveup_index_rou_l]
    return gb_reward_rate, ps_reward_rate, optimal_giveup_time_rou_g, optimal_giveup_time_rou_l

def calculate_padded_averages_and_std(data):
    # print(data)
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

def merge_lists(list1, list2):
    return [x if not math.isnan(x) else y for x, y in zip(list1, list2)]


def set_analysis_path(has_block, task_params):
    if has_block:
        path = os.path.normpath(r'D:\behavior_data') + "\\" + "blocks" + "\\" + task_params
    else:
        path = os.path.normpath(r'D:\behavior_data') + "\\" + "no_blocks" + "\\" + task_params
    os.chdir(path)
    return path