import json
import os
import seaborn as sns
from matplotlib import pyplot as plt
from statsmodels.formula.api import mixedlm

import plots
import utils
from animal import Animal
from session import Session
import ruptures as rpt
from tqdm import tqdm
from scipy.stats import ttest_ind
from scipy import stats
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLM
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.neighbors import KernelDensity

global groupings
groupings = ['timescape', 'sex', 'single_housed']

class BehaviorAnalysis:
    def get_exp_config(self):
        """Get experiment config from json"""
        with open(f'{self.exp_name}.json', 'r') as json_file:
            config_data = json.load(json_file)
        return config_data

    def __init__(self, exp_name, optimal_wait, param_dict, task_type, has_block, task_params):
        self.exp_name = exp_name
        self.animal_assignment = self.get_exp_config()
        self.task_type = task_type
        self.task_params = task_params
        self.has_block = has_block
        self.param_dict = param_dict
        self.optimal_wait = [] # a list of times for every session's optimal wait time
        self.path, _ = utils.set_analysis_path(self.has_block, self.task_params)
        self.animal_list = os.listdir()
        self.animal_num = len(self.animal_list)
        self.mice = [] # this stores the animal object
        self.long_mice_list = []
        self.short_mice_list = []
        self.block_diff = []
        self.stable_block_diff = []

        self.long_session_mean = []
        self.short_session_mean = []
        self.long_session_nonimpulsive_mean = []
        self.short_session_nonimpulsive_mean = []
        self.long_consumption_length = []
        self.short_consumption_length = []
        self.long_bg_repeat = []
        self.short_bg_repeat = []
        self.long_impulsive_perc = []
        self.short_impulsive_perc = []
        self.long_bg_repeat_times = []
        self.short_bg_repeat_times = []
        self.all_licks_by_session_l = []
        self.all_licks_by_session_s = []
        self.long_missing_perc = []
        self.short_missing_perc = []
        self.long_mean_reward_rate = []
        self.short_mean_reward_rate = []
        self.long_adjusted_optimal = []
        self.short_adjusted_optimal = []

        self.bg_length_s = []
        self.bg_length_l = []

    def process_all_animals(self, animals):
        animal_num = len(animals)
        for i in tqdm(range(animal_num)):
            animal = animals[i]
            curr_default = self.animal_assignment.get(animal, [])[0].get("timescape", {}).get("default", [])[0]
            print(curr_default)
            curr_change = self.animal_assignment.get(animal, [])[0].get("timescape", {}).get("change", [])[0]

            curr_sex = self.animal_assignment.get(animal, [])[0].get("sex", {})[0]
            curr_single_housing = self.animal_assignment.get(animal, [])[0].get("single_housed", {})[0]
            curr_single_housed = True if curr_single_housing == "T" else False
            curr_animal = Animal(animal, curr_default, curr_change, curr_sex, curr_single_housed,
                                 self.task_params)

            self.mice.append(curr_animal)
            print(self.path)

            if os.name == 'nt':
                default_path = self.path + "\\" + animal + "\\" + 'default'
            else:
                default_path = self.path + "/" + animal + "/" + 'default'
            # default_path = self.path + "\\" + animal + "\\" + 'default'
            # print(f'Trying to change to directory: {default_path}')
            os.chdir(default_path)
            # print(f'Current working directory: {os.getcwd()}')
            default_session_list = os.listdir()
            default_sessions = [session for session in default_session_list if self.task_type in session]
            curr_animal.default_sessions = default_sessions
            curr_animal.default_session_num = len(default_sessions)
            curr_animal.reverse_index = len(default_sessions)
            # print(curr_animal.reverse_index)
            curr_animal.allSession(default_path, 'default', self.has_block)
            print(f'processed all default {curr_animal.default_session_num} sessions for mice {animal}')

            if os.name == 'nt':
                change_path = self.path + "\\" + animal + "\\" + 'change'
            else:
                change_path = self.path + "/" + animal + "/" + 'change'
            # print(f'Trying to change to directory: {change_path}')
            if os.path.exists(change_path):
                os.chdir(change_path)
                # print(f'Current working directory: {os.getcwd()}')
                change_session_list = os.listdir()
                # filter all the items that are regular
                change_sessions = [session for session in change_session_list if self.task_type in session]
                curr_animal.change_sessions = change_sessions
                curr_animal.change_session_num = len(change_sessions)

                curr_animal.allSession(change_path, 'change', self.has_block)
                print(f'processed all change sessions for mice {animal}')
            else:
                print("only default")
                curr_animal.change_session_num = 0

            curr_animal.getMovingAvg(window_size=8)
            curr_animal.getBlockWaiting()
            curr_animal.getAdjustedOptimal()
            curr_animal.find_significance_from_optimal(0.05)
        return self.mice

    # test the difference between statictics of different blocks
    def test_block_diff(self):
        # make loop
        for i in range(len(self.mice)):
            t_stat, p_value = ttest_ind(self.mice[i].stable_s, self.mice[i].stable_l)
            self.stable_block_diff.append(p_value)
            t_stat, p_value = ttest_ind(self.mice[i].holding_s_mean, self.mice[i].holding_l_mean)
            self.block_diff.append(p_value)
        print("p-vals for different blocks are")
        print(self.block_diff)
        print(self.stable_block_diff)

    def organize_mice_data(self, grouping_criteria, default_only, num_before_transition):
        grouped_data = {}
        for i in range(len(self.mice)):
            mouse = self.mice[i].name
            if grouping_criteria != 'timescape':
                group_key = self.animal_assignment.get(mouse, [])[0].get(grouping_criteria, {})[0]
            else:
                group_key = self.animal_assignment.get(mouse, [])[0].get("timescape", {}).get("default", [])[0]
         #   print(f'current group key is {group_key} under {grouping_criteria}')
            if group_key not in grouped_data:
                grouped_data[group_key] = {
                    'mice_list': [],
                    'optimal_time': [],
                    'session_mean': [],
                    'session_nonimpulsive_mean': [],
                    'consumption_length': [],
                    'mean_reward_rate': [],
                    'bg_repeat': [],
                    'impulsive_perc': [],
                    'all_licks_by_session': [],
                    'bg_repeat_times': [],
                    'bg_length': [],
                    'missing_perc': [],
                    'adjusted_optimal':[]
                }
            grouped_data[group_key]['mice_list'].append(mouse)
            if len(grouped_data[group_key]['optimal_time']) < len(self.mice[i].optimal_wait):
                grouped_data[group_key]['optimal_time'] = self.mice[i].optimal_wait

            if default_only:
                # print(self.mice[i].name)
                grouped_data[group_key]['consumption_length'].append(
                    self.mice[i].mean_consumption_length[:self.mice[i].default_session_num])
                grouped_data[group_key]['mean_reward_rate'].append(
                    self.mice[i].mean_session_reward_rate[:self.mice[i].default_session_num])
                grouped_data[group_key]['adjusted_optimal'].append(self.mice[i].session_adjusted_optimal
                                                                   [:self.mice[i].default_session_num])

                if self.mice[i].default == 'long':
                    grouped_data[group_key]['session_mean'].append(self.mice[i].holding_l_mean
                                                                   [:self.mice[i].default_session_num])
                    grouped_data[group_key]['session_nonimpulsive_mean'].append(self.mice[i].non_reflexive_l_mean
                                                                                [:self.mice[i].default_session_num])
                    grouped_data[group_key]['bg_repeat'].append(self.mice[i].bg_restart_l
                                                                [:self.mice[i].default_session_num])
                    grouped_data[group_key]['impulsive_perc'].append(self.mice[i].reflex_lick_perc_l
                                                                     [:self.mice[i].default_session_num])
                    grouped_data[group_key]['all_licks_by_session'].append(self.mice[i].all_holding_l_by_session
                                                                           [:self.mice[i].default_session_num])
                    grouped_data[group_key]['bg_repeat_times'].append(self.mice[i].bg_restart_licks_l
                                                                      [:self.mice[i].default_session_num])
                    grouped_data[group_key]['bg_length'].append(self.mice[i].mean_background_length_l
                                                                [:self.mice[i].default_session_num])
                    grouped_data[group_key]['missing_perc'].append(self.mice[i].miss_perc_l
                                                                   [:self.mice[i].default_session_num])
                elif self.mice[i].default == 'short':
                    grouped_data[group_key]['session_mean'].append(self.mice[i].holding_s_mean
                                                                   [:self.mice[i].default_session_num])
                    grouped_data[group_key]['session_nonimpulsive_mean'].append(self.mice[i].non_reflexive_s_mean
                                                                                [:self.mice[i].default_session_num])
                    grouped_data[group_key]['bg_repeat'].append(self.mice[i].bg_restart_s
                                                                [:self.mice[i].default_session_num])
                    grouped_data[group_key]['impulsive_perc'].append(self.mice[i].reflex_lick_perc_s
                                                                     [:self.mice[i].default_session_num])
                    grouped_data[group_key]['all_licks_by_session'].append(self.mice[i].all_holding_s_by_session
                                                                           [:self.mice[i].default_session_num])
                    grouped_data[group_key]['bg_repeat_times'].append(self.mice[i].bg_restart_licks_s
                                                                      [:self.mice[i].default_session_num])
                    grouped_data[group_key]['bg_length'].append(self.mice[i].mean_background_length_s
                                                                [:self.mice[i].default_session_num])
                    grouped_data[group_key]['missing_perc'].append(self.mice[i].miss_perc_s
                                                                   [:self.mice[i].default_session_num])
            else:
                num_session = -(num_before_transition + self.mice[i].change_session_num)
                list_pairs = [
                    (self.mice[i].holding_l_mean[num_session:], self.mice[i].holding_s_mean[num_session:]),
                    (self.mice[i].non_reflexive_l_mean[num_session:],
                     self.mice[i].non_reflexive_s_mean[num_session:]),
                    (self.mice[i].bg_restart_l[num_session:], self.mice[i].bg_restart_s[num_session:]),
                    (self.mice[i].reflex_lick_perc_l[num_session:], self.mice[i].reflex_lick_perc_s[num_session:]),
                    (self.mice[i].bg_restart_licks_l[num_session:], self.mice[i].bg_restart_licks_s[num_session:]),
                    (self.mice[i].mean_background_length_l[num_session:],
                     self.mice[i].mean_background_length_s[num_session:]),
                    (self.mice[i].miss_perc_l[num_session:], self.mice[i].miss_perc_s[num_session:])]
                # if self.mice[i].default == "long":
                merged_lists = [utils.merge_lists(list1, list2) for list1, list2 in list_pairs]
                grouped_data[group_key]['session_mean'].append(merged_lists[0])
                grouped_data[group_key]['session_nonimpulsive_mean'].append(merged_lists[1])
                grouped_data[group_key]['consumption_length'].append(
                    self.mice[i].mean_consumption_length[-(self.mice[i].change_session_num
                                                           + num_before_transition):])
                grouped_data[group_key]['adjusted_optimal'].append(self.mice[i].session_adjusted_optimal
                                            [-(self.mice[i].change_session_num + num_before_transition):])
                grouped_data[group_key]['mean_reward_rate'].append(
                    self.mice[i].mean_session_reward_rate[-(self.mice[i].change_session_num
                                                            + num_before_transition):])
                grouped_data[group_key]['bg_repeat'].append(merged_lists[2])
                grouped_data[group_key]['impulsive_perc'].append(merged_lists[3])
                grouped_data[group_key]['all_licks_by_session'].append(self.mice[i].all_holding_l_by_session
                                                                       + self.mice[i].all_holding_s_by_session)
                grouped_data[group_key]['bg_repeat_times'].append(merged_lists[4])
                grouped_data[group_key]['bg_length'].append(merged_lists[5])
                grouped_data[group_key]['missing_perc'].append(merged_lists[6])
        return grouped_data

    def get_groups(self, default_only, num_before_transition, has_single_housing):

        groups_by_timescape = self.organize_mice_data(groupings[0], default_only, num_before_transition)
        groups_by_sex = self.organize_mice_data(groupings[1], default_only, num_before_transition)
        groups_by_housing = self.organize_mice_data(groupings[2], default_only, num_before_transition)

        def process_groups(groups, group_name, attributes):
            variables = {}
            categories = utils.get_categories(group_name)

            for category in categories:
                for attribute in attributes:
                    variables[f'{group_name}_{category}_{attribute}'] = groups[category][attribute]
            return variables

        time_attributes = ['mice_list', 'optimal_time', 'session_mean', 'session_nonimpulsive_mean', 'consumption_length',
                           'mean_reward_rate', 'bg_repeat', 'impulsive_perc', 'all_licks_by_session',
                           'bg_repeat_times', 'bg_length', 'missing_perc', 'adjusted_optimal']
        sex_attributes = ['mice_list', 'optimal_time', 'session_mean', 'session_nonimpulsive_mean', 'consumption_length',
                          'mean_reward_rate', 'bg_repeat', 'impulsive_perc', 'all_licks_by_session',
                          'bg_repeat_times', 'bg_length', 'missing_perc', 'adjusted_optimal']
        housing_attributes = ['mice_list', 'session_mean', 'session_nonimpulsive_mean', 'consumption_length',
                              'mean_reward_rate', 'bg_repeat', 'impulsive_perc', 'all_licks_by_session',
                              'bg_repeat_times', 'bg_length', 'missing_perc', 'adjusted_optimal']

        time_variables = process_groups(groups_by_timescape, 'timescape', time_attributes)
        sex_variables = process_groups(groups_by_sex, 'sex', sex_attributes)
        housing_variables = process_groups(groups_by_housing, 'single_housed', housing_attributes) \
            if has_single_housing else None

        variables = [time_variables, sex_variables, housing_variables]
        return variables

    def find_group_diff(self, has_block, task_params, default_only, *args):
        path, user = utils.set_plotting_path(has_block, task_params)
        os.chdir(path)
        print(f'plotting and saving in {path}')
        if len(args) > 0:
            num_before_transition = args[0]
        else:
            num_before_transition = -1
        has_single_housing, groupings_in_use = utils.get_single_housing(task_params)
        variables = self.get_groups(default_only, num_before_transition, has_single_housing)
        variables = [var for var in variables if var is not None]

        plots.plot_all_animal_waiting(variables[0]['timescape_long_mice_list'],
                                      variables[0]['timescape_long_session_mean'],
                                      variables[0]['timescape_short_mice_list'],
                                      variables[0]['timescape_short_session_mean'])

        plot_patch = False if default_only else True
        for i in range(len(groupings_in_use)):
            categories = utils.get_categories(groupings[i])
            if user == 'ziyi':
                curr_path = path + "\\" + groupings[i] + "\\" + "default "+str(default_only)
            else:
                curr_path = path + "/" + groupings[i] + "/" + "default "+str(default_only)

            if not os.path.exists(curr_path):
                os.makedirs(curr_path)
            os.chdir((curr_path))
           # print(categories)
            plots.plot_group_diff(variables[i][f'{groupings[i]}_{categories[0]}_missing_perc'],
                                  variables[i][f'{groupings[i]}_{categories[1]}_missing_perc'],
                                  categories, 'perc', plot_patch, False, False,
                                  num_before_transition=num_before_transition)
            plt.savefig(f'default only {default_only} missing percentages by {groupings[i]}.svg')
            plt.close()

            g1_bg_length_mean, g2_bg_length_mean = plots.plot_group_diff(
                variables[i][f'{groupings[i]}_{categories[0]}_bg_length'],
                variables[i][f'{groupings[i]}_{categories[1]}_bg_length'],
                categories, 'time', plot_patch, False, False, num_before_transition=num_before_transition)
            plt.savefig(f'default only {default_only} background lengths by {groupings[i]}.svg')
            plt.close()

            plots.plot_group_diff(
                variables[i][f'{groupings[i]}_{categories[0]}_bg_repeat_times'],
                variables[i][f'{groupings[i]}_{categories[1]}_bg_repeat_times'],
                categories, 'count', plot_patch, False, False, num_before_transition=num_before_transition)
            plt.savefig(f'default only {default_only} repeat trigger times by {groupings[i]}.svg')
            plt.close()

            plots.plot_group_diff(variables[i][f'{groupings[i]}_{categories[0]}_mean_reward_rate'],
                                 variables[i][f'{groupings[i]}_{categories[1]}_mean_reward_rate'],
                                  categories, 'rate', plot_patch,
                                 True, False, num_before_transition=num_before_transition)
            plt.savefig(f'default only {default_only} mean reward rate by {groupings[i]}.svg')
            plt.close()

           # impuslive perc
            plots.plot_group_diff(variables[i][f'{groupings[i]}_{categories[0]}_impulsive_perc'],
                                  variables[i][f'{groupings[i]}_{categories[1]}_impulsive_perc'],
                                  categories, 'perc', plot_patch,
                                 True, False, num_before_transition=num_before_transition)
            plt.savefig(f'default only {default_only} impulsive licking percentage by {groupings[i]}.svg')
            plt.close()

            # bg repeats plot
            plots.plot_group_diff(variables[i][f'{groupings[i]}_{categories[0]}_bg_repeat_times'],
                                  variables[i][f'{groupings[i]}_{categories[1]}_bg_repeat_times'],
                                  categories, 'count', plot_patch,
                                 False, False, num_before_transition=num_before_transition)
            plt.savefig(f'default only {default_only} bg repeats for long vs short cohorts by {groupings[i]}.svg')
            plt.close()

           # print(variables[i][f'{groupings[i]}_{categories[0]}_adjusted_optimal'])

            g1_adjusted_optimal, g2_adjusted_optimal = \
                plots.plot_group_diff(variables[i][f'{groupings[i]}_{categories[0]}_adjusted_optimal'],
                                      variables[i][f'{groupings[i]}_{categories[1]}_adjusted_optimal'],
                                      categories, 'time', plot_patch, False, False,
                                      num_before_transition = num_before_transition)
            plt.savefig(f'default only {default_only} adjusted optimal by {groupings[i]}.svg')
            plt.close()

            g1_com_averages, g2_com_averages = \
                plots.plot_group_diff(variables[i][f'{groupings[i]}_{categories[0]}_consumption_length'],
                                      variables[i][f'{groupings[i]}_{categories[1]}_consumption_length'],
                                      categories, 'time', plot_patch, False, False,
                                      num_before_transition = num_before_transition)
            plt.savefig(f'default only {default_only} consumption times long vs short cohorts by {groupings[i]}.svg')
            plt.close()

            # session plots
            if groupings[i] == 'timescape':

                long_session_mean, short_session_mean = plots.plot_group_diff(
                    variables[i][f'{groupings[i]}_{categories[0]}_session_mean'],
                    variables[i][f'{groupings[i]}_{categories[1]}_session_mean'],
                                      categories, 'time', plot_patch, True, True,
                                      opt_long=variables[i][f'{groupings[i]}_{categories[0]}_optimal_time'],
                                      opt_short=variables[i][f'{groupings[i]}_{categories[1]}_optimal_time'],
                                      adjusted_short = g2_adjusted_optimal,
                                      adjusted_long=g1_adjusted_optimal, num_before_transition=num_before_transition)

                self.plot_last_n_differences(10, True,
                                             False, variables[0]['timescape_long_mice_list'],
                                             variables[0]['timescape_short_mice_list'],
                                             variables[i][f'{groupings[i]}_{categories[0]}_adjusted_optimal'],
                                             variables[i][f'{groupings[i]}_{categories[1]}_adjusted_optimal'],
                                             variables[i][f'{groupings[i]}_{categories[0]}_session_mean'],
                                             variables[i][f'{groupings[i]}_{categories[1]}_session_mean'])

                # self.plot_last_n_differences(10, True,
                #                              True, variables[0]['timescape_long_mice_list'],
                #                              variables[0]['timescape_short_mice_list'],
                #
                #                              g1_adjusted_optimal,
                #                              g2_adjusted_optimal,
                #                              long_session_mean,
                #                              short_session_mean
                #                              )
            else:
                plots.plot_group_diff(variables[i][f'{groupings[i]}_{categories[0]}_session_mean'],
                                      variables[i][f'{groupings[i]}_{categories[1]}_session_mean'],
                                      categories, 'time', plot_patch, True, True,
                                      num_before_transition=num_before_transition)
            plt.savefig(f'default only {default_only} session average for long vs short cohorts by {groupings[i]}.svg')
            plt.close()

            # non_impulsive licks
            if groupings[i] == 'timescape':
                plots.plot_group_diff(variables[i][f'{groupings[i]}_{categories[0]}_session_nonimpulsive_mean'],
                                      variables[i][f'{groupings[i]}_{categories[1]}_session_nonimpulsive_mean'],
                                      categories, 'time', plot_patch, True, True,
                                      opt_long=variables[i][f'{groupings[i]}_{categories[0]}_optimal_time'],
                                      opt_short=variables[i][f'{groupings[i]}_{categories[1]}_optimal_time'],
                                      adjusted_short=g2_adjusted_optimal,
                                      adjusted_long=g1_adjusted_optimal, num_before_transition=num_before_transition)
            else:
                plots.plot_group_diff(variables[i][f'{groupings[i]}_{categories[0]}_session_nonimpulsive_mean'],
                                      variables[i][f'{groupings[i]}_{categories[1]}_session_nonimpulsive_mean'],
                                      categories, 'time', plot_patch, True, True,
                                      num_before_transition=num_before_transition)
            plt.savefig(f'default only {default_only} '
                        f'non impulsive session licks average for long vs short cohorts by {groupings[i]}.svg')
            plt.close()

        self.plot_group_pde(variables, has_block, task_params)


    def plot_group_pde(self, variables,has_block,task_params):
        # Define the cohorts (e.g., 'cohort_s' and 'cohort_l')
        path, user = utils.set_plotting_path(has_block, task_params)
        os.chdir(path)
        has_single_housing, groupings_in_use = utils.get_single_housing(task_params)
        for i in range(len(groupings_in_use)):
            categories = utils.get_categories(groupings[i])

            combined_data = [variables[i][f'{groupings[i]}_{categories[0]}_all_licks_by_session'],
                             variables[i][f'{groupings[i]}_{categories[1]}_all_licks_by_session']]
            max_sessions = max(max(len(data) for data in cohort) for cohort in combined_data)
            print(f'number of max session {max_sessions}')
            fig, axes = plt.subplots(max_sessions, 1, figsize=(4, 100))

            for session in range(max_sessions):
                ax = axes[session]

                for cohort_index, cohort in enumerate(categories):
                    # Collect licking data for the current session and cohort
                    licking_data = []
                    for animal_data in combined_data[cohort_index]:
                        if session < len(animal_data):
                            licking_data.extend(animal_data[session])

                    # Plot the KDE for the current cohort in the same subplot
                    if cohort == categories[0]:
                        sns.kdeplot(licking_data, label=f'{categories[0]} Cohort_kde', color='blue',
                                    ax=ax, common_norm=False, bw_adjust=.4)
                        sns.histplot(licking_data, kde=False, label=f'{categories[0]} Cohort_hist',
                                     color='lightblue', stat="density",
                                     ax=ax, bins=50)
                    else:
                        sns.kdeplot(licking_data, label=f'{categories[1]} Cohort_kde', color='red',
                                    ax=ax, common_norm=False, bw_adjust=.4)
                        sns.histplot(licking_data, kde=False, label=f'{categories[1]} Cohort_hist',
                                     color='lightcoral', stat="density",
                                     ax=ax, bins=50)

                ax.set_title(f'Session {session + 1}')
                ax.set_ylabel('Density')

            # Add a common legend to the last subplot
            axes[-1].set_xlabel('Licking Time')
            axes[-1].legend()

            plt.tight_layout()
            plt.savefig(f'PDE for cohorts across sessions by {groupings[i]}.svg')
            plt.close()

    def plot_last_n_differences(self, n, default_only, cohort_avg, long_list, short_list,
                                long_adjusted_optimal, short_adjusted_optimal, long_session_mean,
                                short_session_mean):
        fig, ax = plt.subplots(figsize=(9,7))
        # print(long_list)
        alpha = 0.05

        if default_only:
            if cohort_avg: # t test
                adjusted_optimal_long = long_adjusted_optimal[-n:]
                adjusted_optimal_short = short_adjusted_optimal[-n:]

                short_mean = short_session_mean[-n:]
                long_mean = long_session_mean[-n:]

                # Compare short_mean and long_mean
                t_statistic, p_value_short_long = stats.ttest_ind(short_mean, long_mean)
                if p_value_short_long < alpha:
                    print("Significant difference between short_mean and long_mean")
                else:
                    print("No significant difference between short_mean and long_mean")

                # Compare adjusted_optimal_long and long_mean
                t_statistic, p_value_adj_long = stats.ttest_ind(adjusted_optimal_long, long_mean)
                if p_value_adj_long < alpha:
                    print("Significant difference between adjusted_optimal_long and long_mean")
                else:
                    print("No significant difference between adjusted_optimal_long and long_mean")

                # Compare short_mean and adjusted_optimal_short
                t_statistic, p_value_short_adj = stats.ttest_ind(short_mean, adjusted_optimal_short)
                if p_value_short_adj < alpha:
                    print("Significant difference between short_mean and adjusted_optimal_short")
                else:
                    print("No significant difference between short_mean and adjusted_optimal_short")
            else: # anova
                last_n_entries_long = [sublist[-n:] for sublist in long_session_mean]
                last_n_entries_short = [sublist[-n:] for sublist in short_session_mean]

                last_n_adjust_long = [sublist[-n:] for sublist in long_adjusted_optimal]
                last_n_adjust_short = [sublist[-n:] for sublist in short_adjusted_optimal]

                # print(results)
                combined_data = []
                for animal_sessions, animal_id in zip(last_n_entries_long, long_list):
                    combined_data.extend([(session, animal_id, 'Group 1') for session in animal_sessions])
                for animal_sessions, animal_id in zip(last_n_entries_short, short_list):
                    combined_data.extend([(session, animal_id, 'Group 2') for session in animal_sessions])

                df = pd.DataFrame(combined_data, columns=['session', 'animal', 'group'])
                print(df)
                model_formula = 'session ~ group'
                mixedlm_model = mixedlm(model_formula, groups="animal", data=df).fit(alpha = 0.05)
                print(mixedlm_model.summary())
                p_value_short_long = mixedlm_model.pvalues['group[T.Group 2]']
                print(p_value_short_long)
                if p_value_short_long < alpha:
                    print("Significant difference between short_mean and long_mean")
                else:
                    print("No significant difference between short_mean and long_mean")

                model_formula_adjust_session = 'time ~ group'
                combined_data_short = []
                for animal_sessions, animal_id in zip(last_n_entries_short, short_list):
                    combined_data_short.extend([(session, animal_id, 'session') for session in animal_sessions])
                for adjust_short, animal_id in zip(last_n_adjust_short, short_list):
                    combined_data_short.extend([(adjust, animal_id, 'adjust') for adjust in adjust_short])

                df_short = pd.DataFrame(combined_data_short, columns=['time', 'animal', 'group'])
                print(df_short)
                mixedlm_model_short = mixedlm(model_formula_adjust_session, groups="animal",
                                              data=df_short).fit(alpha=0.05)

                # Print the summary of the model
                print(mixedlm_model_short.summary())
                p_value_short_adj = mixedlm_model_short.pvalues['group[T.session]']
                print(p_value_short_adj)
                if p_value_short_adj < alpha:
                    print("Significant difference between short_mean and adjusted_optimal_short")
                else:
                    print("No significant difference between short_mean and adjusted_optimal_short")

                combined_data_long = []
                for animal_sessions, animal_id in zip(last_n_entries_long, long_list):
                    combined_data_long.extend([(session, animal_id, 'session') for session in animal_sessions])
                for adjust_long, animal_id in zip(last_n_adjust_long, long_list):
                    combined_data_long.extend([(adjust, animal_id, 'adjust') for adjust in adjust_long])

                df_long = pd.DataFrame(combined_data_long, columns=['time', 'animal', 'group'])
                print(df_long)
                mixedlm_model_long = mixedlm(model_formula_adjust_session, groups="animal",
                                             data=df_long).fit(alpha=0.05)

                # Print the summary of the model
                print(mixedlm_model_long.summary())
                # Print the summary of the model
                p_value_adj_long = mixedlm_model_long.pvalues['group[T.session]']
                # print(p_value_long_adj)
                if p_value_adj_long < alpha:
                    print("Significant difference between long_mean and adjusted_optimal_long")
                else:
                    print("No significant difference between long_mean and adjusted_optimal_long")

                # Flatten
                long_mean = [item for sublist in last_n_entries_long for item in sublist]
                short_mean = [item for sublist in last_n_entries_short for item in sublist]
                adjusted_optimal_long = [item for sublist in last_n_adjust_long for item in sublist]
                adjusted_optimal_short = [item for sublist in last_n_adjust_short for item in sublist]

            # Calculate standard deviations
            short_std = np.std(short_mean)
            long_std = np.std(long_mean)
            short_adj_std = np.std(adjusted_optimal_short)
            long_adj_std = np.std(adjusted_optimal_long)

            # Plotting

            labels = ['s_mean', 's_adjusted_opt', 'l_mean', 'l_adjusted_opt']
            short_means = [np.mean(short_mean), np.mean(adjusted_optimal_short), np.nan, np.nan]
            long_means = [np.nan, np.nan, np.mean(long_mean), np.mean(adjusted_optimal_long)]

            short_stds = [short_std, short_adj_std, np.nan, np.nan]
            long_stds = [np.nan, np.nan, long_std, long_adj_std]

            colors = ['blue', 'lightblue', 'red', 'lightcoral']

            plt.bar(labels, short_means, yerr=short_stds, color=colors, alpha=0.5, label='Short', capsize=5)
            plt.bar(labels, long_means, yerr=long_stds, color=colors, alpha=0.5, label='Long', capsize=5)

            for i, (short_data, long_data) in enumerate(zip(short_mean, long_mean)):
                plt.scatter([labels[0], labels[2]], [short_data, long_data], color='gray', marker='o', alpha=0.5)

            for i, (short_data, long_data) in enumerate(zip(adjusted_optimal_short, adjusted_optimal_long)):
                plt.scatter([labels[1], labels[3]], [short_data, long_data], color='gray', marker='o', alpha=0.5)

            if p_value_short_long < alpha:
                # Calculate position for the bracket and asterisk
                x_short = labels.index('s_mean')
                x_long = labels.index('l_mean')
                x_center = (x_short + x_long) / 2
                y_max = max(short_means[x_short], long_means[x_long])

                # Plot bracket
                plt.plot([x_short, x_long], [y_max+2, y_max+2], color='black', linewidth=2)

                # Plot asterisk
                if p_value_short_long < 0.001:
                    plt.text(x_center, y_max + 2.2, '***', fontsize=12, ha='center')

                else:
                    plt.text(x_center, y_max + 2.2, '*', fontsize=12, ha='center')

            if p_value_adj_long < alpha:
                # Calculate position for the bracket and asterisk
                x_short = labels.index('l_mean')
                x_long = labels.index('l_adjusted_opt')
                x_center = (x_short + x_long) / 2
                y_max = max(long_means[x_short], adjusted_optimal_long[x_long])

                # Plot bracket
                plt.plot([x_short, x_long], [y_max+2, y_max+2], color='black', linewidth=2)

                # Plot asterisk
                if p_value_adj_long < 0.001:

                    plt.text(x_center, y_max + 2.2, '***', fontsize=12, ha='center')
                else:
                    plt.text(x_center, y_max + 2.2, '*', fontsize=12, ha='center')

            if p_value_short_adj < alpha:
                # Calculate position for the bracket and asterisk
                x_short = labels.index('s_mean')
                x_long = labels.index('s_adjusted_opt')
                x_center = (x_short + x_long) / 2
                y_max = max(short_means[x_short], adjusted_optimal_short[x_long])

                # Plot bracket
                plt.plot([x_short, x_long], [y_max+2, y_max+2], color='black', linewidth=2)

                if p_value_short_adj < 0.001:

                    plt.text(x_center, y_max + 2.2, '***', fontsize=12, ha='center')
                else:
                    plt.text(x_center, y_max + 2.2, '*', fontsize=12, ha='center')
            plt.xlabel('Variables')
            plt.ylabel('Mean')
            plt.savefig(f'default only {default_only} grouped {cohort_avg} last {n} differences of statistics.svg')
            plt.close()

