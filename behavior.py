import json
import math
import os

import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
from pkg_resources import resource_string

import plots
import utils
from animal import Animal
from session import Session
import ruptures as rpt
from tqdm import tqdm
from scipy.stats import ttest_ind
from scipy import stats
from sklearn.neighbors import KernelDensity

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
        self.optimal_wait = optimal_wait
        utils.set_analysis_path(self.has_block, self.task_params)
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

    def allAnimal(self, animals):
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
                                 self.task_params, self.optimal_wait)

            self.mice.append(curr_animal)
            default_path = self.path + "\\" + animal + "\\" + 'default'
            # print(f'Trying to change to directory: {default_path}')
            os.chdir(default_path)
            # print(f'Current working directory: {os.getcwd()}')
            default_session_list = os.listdir()
            # filter all the items that are regular
            default_sessions = [session for session in default_session_list if self.task_type in session]
            curr_animal.default_sessions = default_sessions
            curr_animal.default_session_num = len(default_sessions)
            curr_animal.reverse_index = len(default_sessions)
            # print(curr_animal.reverse_index)
            curr_animal.allSession(default_path, 'default', self.has_block)
            print(f'processing all default sessions for mice {animal}')

            change_path = self.path + "\\" + animal + "\\" + 'change'
            # print(f'Trying to change to directory: {change_path}')
            os.chdir(change_path)
            # print(f'Current working directory: {os.getcwd()}')
            change_session_list = os.listdir()
            # filter all the items that are regular
            change_sessions = [session for session in change_session_list if self.task_type in session]
            curr_animal.change_sessions = change_sessions
            curr_animal.change_session_num = len(change_sessions)
            # print(f'change session numer is {curr_animal.change_session_num}')

            curr_animal.allSession(change_path, 'change', self.has_block)
            print(f'processing all change sessions for mice {animal}')
            curr_animal.getMovingAvg(window_size=8)
            curr_animal.getBlockWaiting()
            curr_animal.getAdjustedOptimal()

        return self.mice

    # test the difference between statictics of different blocks
    def testBlockDiff(self):
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
        if default_only:
            for i in range(len(self.mice)):
                mouse = self.mice[i].name
                if grouping_criteria == 'timescape':
                    group_key = self.animal_assignment[mouse][grouping_criteria][0]
                else:
                    group_key = self.animal_assignment[mouse][grouping_criteria]['default'][0]
                print(f'current group key is {group_key}')
                if group_key not in grouped_data:
                    grouped_data[group_key] = {
                        'mice_list': [],
                        'session_mean': [],
                        'session_nonimpulsive_mean': [],
                        'consumption_length': [],
                        'mean_reward_rate': [],
                        'bg_repeat': [],
                        'impulsive_perc': [],
                        'all_licks_by_session': [],
                        'bg_repeat_times': [],
                        'bg_length': [],
                        'missing_perc': []
                    }
                if default_only:
                    grouped_data[group_key]['mice_list'].append(mouse)
                    grouped_data[group_key]['session_mean'].append(self.mice[i].holding_mean)
                    grouped_data[group_key]['session_nonimpulsive_mean'].append(self.mice[i].non_reflexive_mean)
                    grouped_data[group_key]['consumption_length'].append(
                        self.mice[i].mean_consumption_length[self.mice[i].default_session_num:])
                    grouped_data[group_key]['mean_reward_rate'].append(
                        self.mice[i].mean_session_reward_rate[self.mice[i].default_session_num:])
                    grouped_data[group_key]['bg_repeat'].append(self.mice[i].bg_restart)
                    grouped_data[group_key]['impulsive_perc'].append(self.mice[i].reflex_lick_perc)
                    grouped_data[group_key]['all_licks_by_session'].append(self.mice[i].all_holding_by_session)
                    grouped_data[group_key]['bg_repeat_times'].append(self.mice[i].bg_restart_licks)
                    grouped_data[group_key]['bg_length'].append(self.mice[i].mean_background_length)
                    grouped_data[group_key]['missing_perc'].append(self.mice[i].miss_perc)
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
        else:
            # reverse_session.append(self.mice[i].reverse_index)
            for i in range(len(self.mice)):
                mouse = self.mice[i].name
             #   print(f'{mouse} get reversed at {self.mice[i].reverse_index}')
                num_session = -(num_before_transition + self.mice[i].change_session_num)

                list_pairs = [
                    (self.mice[i].holding_l_mean[num_session:], self.mice[i].holding_s_mean[num_session:]),
                    (self.mice[i].non_reflexive_l_mean[num_session:], self.mice[i].non_reflexive_s_mean[num_session:]),
                    (self.mice[i].bg_restart_l[num_session:], self.mice[i].bg_restart_s[num_session:]),
                    (self.mice[i].reflex_lick_perc_l[num_session:], self.mice[i].reflex_lick_perc_s[num_session:]),
                    (self.mice[i].bg_restart_licks_l[num_session:], self.mice[i].bg_restart_licks_s[num_session:]),
                    (self.mice[i].mean_background_length_l[num_session:],
                     self.mice[i].mean_background_length_s[num_session:]),
                    (self.mice[i].miss_perc_l[num_session:], self.mice[i].miss_perc_s[num_session:])]

                if self.mice[i].default == "long":
                    merged_lists = [utils.merge_lists_with_sources(list1, list2) for list1, list2 in list_pairs]
                    self.long_mice_list.append(mouse)
                    self.long_session_mean.append([x[0] for x in merged_lists[0]])
                    self.long_session_nonimpulsive_mean.append([x[0] for x in merged_lists[1]])
                    self.long_consumption_length.append(
                        self.mice[i].mean_consumption_length[-(self.mice[i].change_session_num + num_before_transition):])
                    self.long_mean_reward_rate.append(
                        self.mice[i].mean_session_reward_rate[-(self.mice[i].change_session_num + num_before_transition):])
                    self.long_bg_repeat.append([x[0] for x in merged_lists[2]])
                    self.long_impulsive_perc.append([x[0] for x in merged_lists[3]])
                    self.all_licks_by_session_l.append(
                        self.mice[i].all_holding_l_by_session + self.mice[i].all_holding_s_by_session)
                    self.long_bg_repeat_times.append([x[0] for x in merged_lists[4]])
                    self.bg_length_l.append([x[0] for x in merged_lists[5]])
                    self.long_missing_perc.append([x[0] for x in merged_lists[6]])
                    self.long_adjusted_optimal.append(
                        self.mice[i].session_adjusted_optimal[-(self.mice[i].change_session_num + num_before_transition):])
        return grouped_data
    #
    # self.grouped_by_training_program = organize_mice_data('timescape')
    # self.grouped_by_sex = self.organize_mice_data('sex')
    # self.grouped_by_housing_status = self.organize_mice_data('housing_status')
    def PlotCohortDiff(self, default_only, *args):
        path = os.path.normpath(r'D:\figures\behplots') + "\\" + "no_blocks" + "\\" + self.task_params
        os.chdir(path)
        print(f'plotting and saving in {path}')
        # reverse_session = []

        if len(args) > 0:
            num_before_transition = args[0]
        else:
            num_before_transition = -1
        grouped_by_training_program = self.organize_mice_data("timescape", default_only, num_before_transition)
        print(grouped_by_training_program) #empty?

        if default_only:
            for i in range(len(self.mice)):
                mouse = self.mice[i].name
                if self.mice[i].default == "long":
                    self.long_mice_list.append(mouse)
                    self.long_session_mean.append(self.mice[i].holding_l_mean)
                    self.long_session_nonimpulsive_mean.append(self.mice[i].non_reflexive_l_mean)
                    self.long_consumption_length.append(self.mice[i].mean_consumption_length[self.mice[i].default_session_num:])
                    self.long_mean_reward_rate.append(self.mice[i].mean_session_reward_rate[self.mice[i].default_session_num:])
                    self.long_bg_repeat.append(self.mice[i].bg_restart_l)
                    self.long_impulsive_perc.append(self.mice[i].reflex_lick_perc_l)
                    self.all_licks_by_session_l.append(self.mice[i].all_holding_l_by_session)
                    self.long_bg_repeat_times.append(self.mice[i].bg_restart_licks_l)
                    self.bg_length_l.append(self.mice[i].mean_background_length_l)
                    self.long_missing_perc.append(self.mice[i].miss_perc_l)
                    self.long_adjusted_optimal.append(self.mice[i].session_adjusted_optimal[self.mice[i].default_session_num:])
                else:
                    self.short_mice_list.append(mouse)
                    self.short_session_mean.append(self.mice[i].holding_s_mean)
                    self.short_session_nonimpulsive_mean.append(self.mice[i].non_reflexive_s_mean)
                    self.short_consumption_length.append(self.mice[i].mean_consumption_length[self.mice[i].default_session_num:])
                    self.short_mean_reward_rate.append(self.mice[i].mean_session_reward_rate[self.mice[i].default_session_num:])
                    self.short_bg_repeat.append(self.mice[i].bg_restart_s)
                    self.short_impulsive_perc.append(self.mice[i].reflex_lick_perc_s)
                    self.all_licks_by_session_s.append(self.mice[i].all_holding_s_by_session)
                    self.short_bg_repeat_times.append(self.mice[i].bg_restart_licks_s)
                    self.bg_length_s.append(self.mice[i].mean_background_length_s)
                    self.short_missing_perc.append(self.mice[i].miss_perc_s)
                    self.short_adjusted_optimal.append(self.mice[i].session_adjusted_optimal[self.mice[i].default_session_num:])

        else:
            for i in range(len(self.mice)):
                mouse = self.mice[i].name
                # reverse_session.append(self.mice[i].reverse_index)
                print(f'{mouse} get reversed at {self.mice[i].reverse_index}')
                num_session = -(num_before_transition + self.mice[i].change_session_num)

                list_pairs = [
                    (self.mice[i].holding_l_mean[num_session:], self.mice[i].holding_s_mean[num_session:]),
                    (self.mice[i].non_reflexive_l_mean[num_session:], self.mice[i].non_reflexive_s_mean[num_session:]),
                    (self.mice[i].bg_restart_l[num_session:], self.mice[i].bg_restart_s[num_session:]),
                    (self.mice[i].reflex_lick_perc_l[num_session:], self.mice[i].reflex_lick_perc_s[num_session:]),
                    (self.mice[i].bg_restart_licks_l[num_session:], self.mice[i].bg_restart_licks_s[num_session:]),
                    (self.mice[i].mean_background_length_l[num_session:], self.mice[i].mean_background_length_s[num_session:]),
                    (self.mice[i].miss_perc_l[num_session:], self.mice[i].miss_perc_s[num_session:])]

                if self.mice[i].default == "long":
                    merged_lists = [utils.merge_lists_with_sources(list1, list2) for list1, list2 in list_pairs]
                    self.long_mice_list.append(mouse)
                    self.long_session_mean.append([x[0] for x in merged_lists[0]])
                    self.long_session_nonimpulsive_mean.append([x[0] for x in merged_lists[1]])
                    self.long_consumption_length.append(self.mice[i].mean_consumption_length[-(self.mice[i].change_session_num+num_before_transition):])
                    self.long_mean_reward_rate.append(self.mice[i].mean_session_reward_rate[-(self.mice[i].change_session_num+num_before_transition):])
                    self.long_bg_repeat.append([x[0] for x in merged_lists[2]])
                    self.long_impulsive_perc.append([x[0] for x in merged_lists[3]])
                    self.all_licks_by_session_l.append(self.mice[i].all_holding_l_by_session + self.mice[i].all_holding_s_by_session)
                    self.long_bg_repeat_times.append([x[0] for x in merged_lists[4]])
                    self.bg_length_l.append([x[0] for x in merged_lists[5]])
                    self.long_missing_perc.append([x[0] for x in merged_lists[6]])
                    self.long_adjusted_optimal.append(self.mice[i].session_adjusted_optimal[-(self.mice[i].change_session_num+num_before_transition):])
                else:
                    merged_lists = [utils.merge_lists_with_sources(list1, list2) for list2, list1 in list_pairs]
                    self.short_mice_list.append(mouse)
                    self.short_session_mean.append([x[0] for x in merged_lists[0]])
                    self.short_session_nonimpulsive_mean.append([x[0] for x in merged_lists[1]])
                    self.short_consumption_length.append(self.mice[i].mean_consumption_length[-(self.mice[i].change_session_num+num_before_transition):])
                    self.short_mean_reward_rate.append(self.mice[i].mean_session_reward_rate[-(self.mice[i].change_session_num+num_before_transition):])
                    self.short_bg_repeat.append([x[0] for x in merged_lists[2]])
                    self.short_impulsive_perc.append([x[0] for x in merged_lists[3]])
                    self.all_licks_by_session_s.append(self.mice[i].all_holding_s_by_session + self.mice[i].all_holding_l_by_session)
                    self.short_bg_repeat_times.append([x[0] for x in merged_lists[4]])
                    self.bg_length_s.append([x[0] for x in merged_lists[5]])
                    self.short_missing_perc.append([x[0] for x in merged_lists[6]])
                    self.short_adjusted_optimal.append(self.mice[i].session_adjusted_optimal[-(self.mice[i].change_session_num+num_before_transition):])


        plots.plotAllAnimalWaiting(self.long_mice_list, self.long_session_mean,
                                   self.short_mice_list, self.short_session_mean)

        plot_patch = False if default_only else True

        plots.plotCohortDiff(self.long_missing_perc, self.short_missing_perc, 'perc', plot_patch, False, False,
                             num_before_transition=num_before_transition)
        plt.savefig('missing percentages.svg')
        plt.close()

        long_bg_length_mean, short_bg_length_mean = plots.plotCohortDiff(self.bg_length_l, self.bg_length_s,
                                                                         'time', plot_patch, False, False,
                                                                         num_before_transition=num_before_transition)
        plt.savefig('background lengths.svg')
        plt.close()

        plots.plotCohortDiff(self.long_bg_repeat_times, self.short_bg_repeat_times, 'count', plot_patch,
                             False, False, num_before_transition=num_before_transition)
        plt.savefig('repeat trigger times.svg')
        plt.close()

        plots.plotCohortDiff(self.long_mean_reward_rate, self.short_mean_reward_rate, 'rate', plot_patch,
                             True, False, num_before_transition=num_before_transition)
        plt.savefig('mean reward rate.svg')
        plt.close()

       # impuslive perc
        plots.plotCohortDiff(self.long_impulsive_perc, self.short_impulsive_perc, 'perc', plot_patch,
                             True, False, num_before_transition=num_before_transition)
        plt.savefig('impulsive licking percentage.svg')
        plt.close()

        # bg repeats plot
        plots.plotCohortDiff(self.long_bg_repeat_times, self.short_bg_repeat_times, 'count', plot_patch,
                             False, False, num_before_transition=num_before_transition)
        plt.savefig('bg repeats for long vs short cohorts.svg')
        plt.close()

        long_adjusted_optimal, short_adjusted_optimal = plots.plotCohortDiff(self.long_adjusted_optimal,
                                                                             self.short_adjusted_optimal, 'time',
                                                                             plot_patch, False, False,
                                                                             num_before_transition = num_before_transition)

        long_com_averages, short_com_averages = plots.plotCohortDiff(self.long_consumption_length,
                                                                     self.short_consumption_length, 'time',
                                                                     plot_patch, False, False,
                                                                     num_before_transition=num_before_transition)
        plt.savefig('consumption times long vs short cohorts.svg')
        plt.close()

        # session plots
        plots.plotCohortDiff(self.long_session_mean, self.short_session_mean, 'time', plot_patch, True, True,
                             opt_long=self.optimal_wait[1], opt_short=self.optimal_wait[0],
                             adjusted_short=short_adjusted_optimal,
                             adjusted_long=long_adjusted_optimal, num_before_transition=num_before_transition)
        plt.savefig('session average for long vs short cohorts.svg')
        plt.close()

        # non_impulsive licks
        plots.plotCohortDiff(self.long_session_nonimpulsive_mean, self.short_session_nonimpulsive_mean,
                             'time', plot_patch, True, True, opt_long=self.
                             optimal_wait[1], opt_short=self.optimal_wait[0], adjusted_short=short_adjusted_optimal,
                             adjusted_long=long_adjusted_optimal, num_before_transition=num_before_transition)
        plt.savefig('non impulsive session licks average for long vs short cohorts.svg')
        plt.close()

    # def PlotAllAnimal(self):


    def PlotCohortSessionPDEDiff(self):
        # Define the cohorts (e.g., 'cohort_s' and 'cohort_l')
        cohorts = ['cohort_s', 'cohort_l']
        combined_data = [self.all_licks_by_session_s, self.all_licks_by_session_l]
        max_sessions = max(len(session_data) for session_data in combined_data)
        max_sessions = max(max(len(data) for data in cohort) for cohort in combined_data)
        print(f'number of max session {max_sessions}')
        fig, axes = plt.subplots(max_sessions, 1, figsize=(4,100))

        for session in range(max_sessions):
            ax = axes[session]

            for cohort_index, cohort in enumerate(cohorts):
                # Collect licking data for the current session and cohort
                licking_data = []
                for animal_data in combined_data[cohort_index]:
                    if session < len(animal_data):
                        licking_data.extend(animal_data[session])

                # Plot the KDE for the current cohort in the same subplot
                if cohort == 'cohort_s':
                    sns.kdeplot(licking_data, label='Short Cohort_kde', color='blue', ax=ax, common_norm=False, bw_adjust=.4)
                    sns.histplot(licking_data, kde=False, label='Short Cohort_hist', color='lightblue', stat="density",
                                 ax=ax, bins=50)
                else:
                    sns.kdeplot(licking_data, label='Long Cohort_kde', color='red', ax=ax, common_norm=False, bw_adjust=.4)
                    sns.histplot(licking_data, kde=False, label='Long Cohort_hist', color='lightcoral', stat="density",
                                 ax=ax, bins=50)

            ax.set_title(f'Session {session + 1}')
            ax.set_ylabel('Density')

        # Add a common legend to the last subplot
        axes[-1].set_xlabel('Licking Time')
        axes[-1].legend()

        plt.tight_layout()
        plt.savefig('PDE for cohorts across sessions.svg')
        plt.close()
