import json
import math
import os

import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
from pkg_resources import resource_string

import utils
from animal import Animal
from session import Session
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
        self.exp_config = self.get_exp_config()
        self.animal_assignment = self.exp_config['timescape']
        self.task_type = task_type
        self.task_params = task_params
        self.has_block = has_block
        self.param_dict = param_dict
        self.optimal_wait = optimal_wait
        if self.has_block:
            self.path = os.path.normpath(r'D:\behavior_data') + "\\" + "blocks" + "\\" + task_params
        else:
            self.path = os.path.normpath(r'D:\behavior_data') + "\\" + "no_blocks" + "\\" + task_params
        # print(self.path)
        os.chdir(self.path)
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

        self.bg_length_s = []
        self.bg_length_l = []
    def getStableTimes(self, mouse):
        for j in range(len(mouse.moving_average_s_var)):
            if not math.isnan(mouse.moving_average_s_var[j]) and mouse.moving_average_s_var[j] < 1:
                mouse.stable_s.append(mouse.moving_average_s[j])

        for k in range(len(mouse.moving_average_l_var)):
            if not math.isnan(mouse.moving_average_l_var[k]) and mouse.moving_average_l_var[k] < 1:
                mouse.stable_l.append(mouse.moving_average_l[k])
        print(len(mouse.stable_l))

    def allAnimal(self, animals):
        animal_num = len(animals)
        for i in range(animal_num):
            animal = animals[i]
            curr_animal = Animal(animal, self.task_params)
            self.mice.append(curr_animal)
            curr_path = self.path + "\\" + animal
            os.chdir(curr_path)
            session_list = os.listdir()
            # filter all the items that are regular
            sessions = [session for session in session_list if self.task_type in session]
            curr_animal.sessions = sessions

            curr_animal.allSession(curr_path, self.has_block)
            print(f'processing all sessions for mice {animal}')
            curr_animal.getMovingAvg(window_size=8)
            curr_animal.getBlockWaiting()
            self.getStableTimes(curr_animal)
            # self.mice[i].stable_s = [s for s in self.mice[i].moving_average_s if s > self.optimal_wait[0]]
            # self.mice[i].stable_l = [l for l in self.mice[i].moving_average_l if l > self.optimal_wait[1]]
            print(len(curr_animal.stable_s))
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

    def PlotCohortDiff(self):
        path = os.path.normpath(r'D:\figures\behplots') + "\\" + "no_blocks" + "\\" + self.task_params
        os.chdir(path)
        print(f'plotting and saving in {path}')
        for i in range(len(self.mice)):
            mouse = self.mice[i].name
            if self.animal_assignment[self.mice[i].name]['default'][0] == "long":
                self.long_mice_list.append(mouse)
                self.long_session_mean.append(self.mice[i].holding_l_mean)
                self.long_session_nonimpulsive_mean.append(self.mice[i].non_reflexive_l_mean)
                self.long_consumption_length.append(self.mice[i].mean_consumption_length)
                self.long_bg_repeat.append(self.mice[i].bg_restart_l)
                self.long_impulsive_perc.append(self.mice[i].reflex_lick_perc_l)
                self.all_licks_by_session_l.append(self.mice[i].all_holding_l_by_session)
                self.long_bg_repeat_times.append(self.mice[i].bg_restart_licks_l)
                self.bg_length_l.append(self.mice[i].mean_background_length_l)
                self.long_missing_perc.append(self.mice[i].miss_perc_l)
            else:
                self.short_mice_list.append(mouse)
                self.short_session_mean.append(self.mice[i].holding_s_mean)
                self.short_session_nonimpulsive_mean.append(self.mice[i].non_reflexive_s_mean)
                self.short_consumption_length.append(self.mice[i].mean_consumption_length)
                self.short_bg_repeat.append(self.mice[i].bg_restart_s)
                self.short_impulsive_perc.append(self.mice[i].reflex_lick_perc_s)
                self.all_licks_by_session_s.append(self.mice[i].all_holding_s_by_session)
                self.short_bg_repeat_times.append(self.mice[i].bg_restart_licks_s)
                self.bg_length_s.append(self.mice[i].mean_background_length_s)
                self.short_missing_perc.append(self.mice[i].miss_perc_s)
        fig, ax = plt.subplots()
        # Iterate through each sublist and plot it as a line
        for mice, animal_sessions in zip(self.long_mice_list, self.long_session_mean):
           # print(animal_sessions)
            x = list(range(1, len(animal_sessions) + 1))  # Generate x values (1, 2, 3, ...)
            y = animal_sessions
            ax.plot(x, y, marker='o', label=mice)
        for mice, animal_sessions in zip(self.short_mice_list, self.short_session_mean):
           # print(animal_sessions)
            x = list(range(1, len(animal_sessions) + 1))  # Generate x values (1, 2, 3, ...)
            y = animal_sessions
            ax.plot(x, y, marker='o', label=mice)
        # Customize the plot
        ax.set_xlabel('sessions')
        ax.set_ylabel('mean waiting time')
        ax.legend()
        plt.savefig('all animal waiting.svg')

        # bg repeats times
        long_miss_perc_mean, long_miss_perc_std, padded_long = calculate_padded_averages_and_std(
            self.long_missing_perc)
        short_miss_perc_mean, short_miss_perc_std, padded_short = calculate_padded_averages_and_std(
            self.short_missing_perc)

        fig, ax = plt.subplots()
        # Plot the line graph for long sessions
        x = list(range(1, len(long_miss_perc_mean) + 1))
        y = long_miss_perc_mean
        ax.plot(x, y, marker='o', label='Average_long', color='red')

        # Shade the area around the line plot to represent the standard deviation for long sessions
        ax.fill_between(x, [mean - std for mean, std in zip(y, long_miss_perc_std)],
                        [mean + std for mean, std in zip(y, long_miss_perc_std)], alpha=0.5,
                        label='Standard Deviation_long',
                        color='#FFAAAA')
        # Plot the line graph for short sessions
        x = list(range(1, len(short_miss_perc_mean) + 1))
        y = short_miss_perc_mean
        ax.plot(x, y, marker='o', label='Average_short', color='blue')

        # Shade the area around the line plot to represent the standard deviation for short sessions
        ax.fill_between(x, [mean - std for mean, std in zip(y, short_miss_perc_std)],
                        [mean + std for mean, std in zip(y, short_miss_perc_std)], alpha=0.5,
                        label='Standard Deviation_short',
                        color='lightblue')
        ax.set_xlabel('session #')
        ax.set_ylabel('%')
        ax.legend()
        plt.savefig('missing percentages.svg')
        plt.close()

        # bg repeats times
        long_bg_length_mean, long_bg_length_std, padded_long = calculate_padded_averages_and_std(
            self.bg_length_l)
        short_bg_length_mean, short_bg_length_std, padded_short = calculate_padded_averages_and_std(
            self.bg_length_s)

        fig, ax = plt.subplots()
        # Plot the line graph for long sessions
        x = list(range(1, len(long_bg_length_mean) + 1))
        y = long_bg_length_mean
        ax.plot(x, y, marker='o', label='Average_long', color='red')

        # Shade the area around the line plot to represent the standard deviation for long sessions
        ax.fill_between(x, [mean - std for mean, std in zip(y, long_bg_length_std)],
                        [mean + std for mean, std in zip(y, long_bg_length_std)], alpha=0.5,
                        label='Standard Deviation_long',
                        color='#FFAAAA')
        # Plot the line graph for short sessions
        x = list(range(1, len(short_bg_length_mean) + 1))
        y = short_bg_length_mean
        ax.plot(x, y, marker='o', label='Average_short', color='blue')

        # Shade the area around the line plot to represent the standard deviation for short sessions
        ax.fill_between(x, [mean - std for mean, std in zip(y, short_bg_length_std)],
                        [mean + std for mean, std in zip(y, short_bg_length_std)], alpha=0.5,
                        label='Standard Deviation_short',
                        color='lightblue')
        ax.set_xlabel('session #')
        ax.set_ylabel('time(s)')
        ax.legend()
        plt.savefig('background lengths.svg')
        plt.close()

        # bg repeats times
        long_bg_repeat_time_averages, long_bg_repeat_time_std, padded_long = calculate_padded_averages_and_std(
            self.long_bg_repeat_times)
        short_bg_repeat_time_averages, short_bg_repeat_time_std, padded_short = calculate_padded_averages_and_std(
            self.short_bg_repeat_times)

        fig, ax = plt.subplots()
        # Plot the line graph for long sessions
        x = list(range(1, len(long_bg_repeat_time_averages) + 1))
        y = long_bg_repeat_time_averages
        ax.plot(x, y, marker='o', label='Average_long', color='red')

        # Shade the area around the line plot to represent the standard deviation for long sessions
        ax.fill_between(x, [mean - std for mean, std in zip(y, long_bg_repeat_time_std)],
                        [mean + std for mean, std in zip(y, long_bg_repeat_time_std)], alpha=0.5,
                        label='Standard Deviation_long',
                        color='#FFAAAA')
        # Plot the line graph for short sessions
        x = list(range(1, len(short_bg_repeat_time_averages) + 1))
        y = short_bg_repeat_time_averages
        ax.plot(x, y, marker='o', label='Average_short', color='blue')

        # Shade the area around the line plot to represent the standard deviation for short sessions
        ax.fill_between(x, [mean - std for mean, std in zip(y, short_bg_repeat_time_std)],
                        [mean + std for mean, std in zip(y, short_bg_repeat_time_std)], alpha=0.5,
                        label='Standard Deviation_short',
                        color='lightblue')
        ax.set_xlabel('session #')
        ax.set_ylabel('number of repeat trigger')
        ax.legend()
        plt.savefig('repeat trigger times.svg')
        plt.close()

       # impuslive perc
        long_impusive_perc_averages, long_impusive_perc_std, padded_long = calculate_padded_averages_and_std(self.long_impulsive_perc)
        short_impusive_perc_averages, short_impusive_perc_std, padded_short = calculate_padded_averages_and_std(self.short_impulsive_perc)

        fig, ax = plt.subplots()
        # Plot the line graph for long sessions
        x = list(range(1, len(long_impusive_perc_averages) + 1))
        y = long_impusive_perc_averages
        ax.plot(x, y, marker='o', label='Average_long', color='red')

        # Shade the area around the line plot to represent the standard deviation for long sessions
        ax.fill_between(x, [mean - std for mean, std in zip(y, long_impusive_perc_std)],
                        [mean + std for mean, std in zip(y, long_impusive_perc_std)], alpha=0.5,
                        label='Standard Deviation_long',
                        color='#FFAAAA')
        # Plot the line graph for short sessions
        x = list(range(1, len(short_impusive_perc_averages) + 1))
        y = short_impusive_perc_averages
        ax.plot(x, y, marker='o', label='Average_short', color='blue')

        # Shade the area around the line plot to represent the standard deviation for short sessions
        ax.fill_between(x, [mean - std for mean, std in zip(y, short_impusive_perc_std)],
                        [mean + std for mean, std in zip(y, short_impusive_perc_std)], alpha=0.5,
                        label='Standard Deviation_short',
                        color='lightblue')
        ax.set_xlabel('session #')
        ax.set_ylabel('percentage')
        ax.legend()
        plt.savefig('impulsive licking percentage.svg')
        plt.close()

        # bg repeats plot
        long_repeat_averages, long_std_repeat, padded_long = calculate_padded_averages_and_std(self.long_bg_repeat)
        short_repeat_averages, short_std_repeat, padded_short = calculate_padded_averages_and_std(self.short_bg_repeat)

        fig, ax = plt.subplots()
        # Plot the line graph for long sessions
        x = list(range(1, len(long_repeat_averages) + 1))
        y = long_repeat_averages
        ax.plot(x, y, marker='o', label='Average_long', color='red')

        # Shade the area around the line plot to represent the standard deviation for long sessions
        ax.fill_between(x, [mean - std for mean, std in zip(y, long_std_repeat)],
                        [mean + std for mean, std in zip(y, long_std_repeat)], alpha=0.5,
                        label='Standard Deviation_long',
                        color='#FFAAAA')
        # Plot the line graph for short sessions
        x = list(range(1, len(short_repeat_averages) + 1))
        y = short_repeat_averages
        ax.plot(x, y, marker='o', label='Average_short', color='blue')

        # Shade the area around the line plot to represent the standard deviation for short sessions
        ax.fill_between(x, [mean - std for mean, std in zip(y, short_std_repeat)],
                        [mean + std for mean, std in zip(y, short_std_repeat)], alpha=0.5,
                        label='Standard Deviation_short',
                        color='lightblue')
        ax.set_xlabel('session #')
        ax.set_ylabel('percentage')
        ax.legend()
        plt.savefig('bg repeats for long vs short cohorts.svg')
        plt.close()

        long_com_averages, long_com_std, padded_long = calculate_padded_averages_and_std(self.long_consumption_length)
        short_com_averages, short_com_std, padded_short = calculate_padded_averages_and_std(
            self.short_consumption_length)
        fig, ax = plt.subplots()
        # Plot the line graph for long sessions
        x = list(range(1, len(long_com_averages) + 1))
        y = long_com_averages
        ax.plot(x, y, marker='o', label='Average_long', color='red')

        # Shade the area around the line plot to represent the standard deviation for long sessions
        ax.fill_between(x, [mean - std for mean, std in zip(y, long_com_std)],
                        [mean + std for mean, std in zip(y, long_com_std)], alpha=0.5,
                        label='Standard Deviation_long',
                        color='#FFAAAA')
        # Plot the line graph for short sessions
        x = list(range(1, len(short_com_averages) + 1))
        y = short_com_averages
        ax.plot(x, y, marker='o', label='Average_short', color='blue')

        # Shade the area around the line plot to represent the standard deviation for short sessions
        ax.fill_between(x, [mean - std for mean, std in zip(y, short_com_std)],
                        [mean + std for mean, std in zip(y, short_com_std)], alpha=0.5,
                        label='Standard Deviation_short',
                        color='lightblue')

        ax.set_xlabel('session #')
        ax.set_ylabel('time (s)')
        ax.legend()
        plt.savefig('consumption times long vs short cohorts.svg')
        plt.close()

        adjusted_long_optimal = utils.getOptimalTime(self.param_dict['m2'],self.param_dict['p2'],
                                                     long_bg_length_mean[-1] + long_com_averages[-1])
        adjusted_short_optimal = utils.getOptimalTime(self.param_dict['m1'],self.param_dict['p1'],
                                                     short_bg_length_mean[-1] + short_com_averages[-1])

        # session plots
        long_session_averages, long_std_session, padded_long = calculate_padded_averages_and_std(self.long_session_mean)
        short_session_averages, short_std_session, padded_short = calculate_padded_averages_and_std(self.short_session_mean)

        fig, ax = plt.subplots()
        # Plot the line graph for long sessions
        x = list(range(1, len(long_session_averages) + 1))
        y = long_session_averages
        ax.plot(x, y, marker='o', label='Average_long', color='red')
        ax.axhline(y=self.optimal_wait[1], color='r', linestyle='--', label='Optimal_long')
        ax.axhline(y=adjusted_long_optimal, color='lightcoral', linestyle='--', label='adjusted_optimal_long')


        # Shade the area around the line plot to represent the standard deviation for long sessions
        ax.fill_between(x, [mean - std for mean, std in zip(y, long_std_session)],
                        [mean + std for mean, std in zip(y, long_std_session)], alpha=0.5,
                        label='Standard Deviation_long',
                        color='#FFAAAA')
        # Plot the line graph for short sessions
        x = list(range(1, len(short_session_averages) + 1))
        y = short_session_averages
        ax.plot(x, y, marker='o', label='Average_short', color='blue')
        ax.axhline(y=self.optimal_wait[0], color='b', linestyle='--', label='Optimal_short')
        ax.axhline(y=adjusted_short_optimal, color='lightblue', linestyle='--', label='adjusted_optimal_short')

        # Shade the area around the line plot to represent the standard deviation for short sessions
        ax.fill_between(x, [mean - std for mean, std in zip(y, short_std_session)],
                        [mean + std for mean, std in zip(y, short_std_session)], alpha=0.5,
                        label='Standard Deviation_short',
                        color='lightblue')
        sig = compare_lists_with_significance(padded_long, padded_short)
        for i in sig['high']:
            ax.annotate('***', xy=(x[i], y[i]), xytext=(x[i], 0), textcoords='data',
                        arrowprops=dict(arrowstyle="->"), rotation='vertical')

        for i in sig['low']:
            ax.annotate('*', xy=(x[i], y[i]), xytext=(x[i], 0), textcoords='data',
                        arrowprops=dict(arrowstyle="->"), rotation='vertical')

        ax.set_xlabel('session #')
        ax.set_ylabel('wait time (s)')
        ax.legend()
        plt.savefig('session average for long vs short cohorts.svg')
        plt.close()

        # non_impulsive licks
        long_averages, long_std_deviation, padded_long = calculate_padded_averages_and_std(self.long_session_nonimpulsive_mean)
        short_averages, short_std_deviation, padded_short = calculate_padded_averages_and_std(self.short_session_nonimpulsive_mean)
        fig, ax = plt.subplots()
        # Plot the line graph for long sessions
        x = list(range(1, len(long_averages) + 1))
        y = long_averages
        ax.plot(x, y, marker='o', label='Average_long', color='red')
        ax.axhline(y=self.optimal_wait[1], color='r', linestyle='--', label='Optimal_l')
        ax.axhline(y=adjusted_long_optimal, color='lightcoral', linestyle='--', label='adjusted_optimal_long')
        # Shade the area around the line plot to represent the standard deviation for long sessions
        ax.fill_between(x, [mean - std for mean, std in zip(y, long_std_deviation)],
                        [mean + std for mean, std in zip(y, long_std_deviation)], alpha=0.5, label='Standard Deviation_long',
                        color='#FFAAAA')
        # Plot the line graph for short sessions
        x = list(range(1, len(short_averages) + 1))
        y = short_averages
        ax.plot(x, y, marker='o', label='Average_short', color='blue')
        ax.axhline(y=self.optimal_wait[0], color='b', linestyle='--', label='Optimal_s')
        ax.axhline(y=adjusted_short_optimal, color='lightblue', linestyle='--', label='adjusted_optimal_short')
        # Shade the area around the line plot to represent the standard deviation for short sessions
        ax.fill_between(x, [mean - std for mean, std in zip(y, short_std_deviation)],
                        [mean + std for mean, std in zip(y, short_std_deviation)], alpha=0.5, label='Standard Deviation_short',
                        color='lightblue')

        non_impuslive_sig = compare_lists_with_significance(padded_long, padded_short)
        for i in non_impuslive_sig['high']:
            ax.annotate('***', xy=(x[i], y[i]), xytext=(x[i], 0), textcoords='data',
                        arrowprops=dict(arrowstyle="->"), rotation='vertical')

        for i in non_impuslive_sig['low']:
            ax.annotate('*', xy=(x[i], y[i]), xytext=(x[i], 0), textcoords='data',
                        arrowprops=dict(arrowstyle="->"), rotation='vertical')

        ax.set_xlabel('session #')
        ax.set_ylabel('wait time (s)')
        ax.legend()
        plt.savefig('non impulsive session licks average for long vs short cohorts.svg')
        plt.close()



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
                                 ax=ax, bins=25)
                else:
                    sns.kdeplot(licking_data, label='Long Cohort_kde', color='red', ax=ax, common_norm=False, bw_adjust=.4)
                    sns.histplot(licking_data, kde=False, label='Long Cohort_hist', color='lightcoral', stat="density",
                                 ax=ax, bins=25)

            ax.set_title(f'Session {session + 1}')
            ax.set_ylabel('Density')

        # Add a common legend to the last subplot
        axes[-1].set_xlabel('Licking Time')
        axes[-1].legend()

        plt.tight_layout()
        plt.savefig('PDE for cohorts across sessions.svg')
        plt.close()




def calculate_padded_averages_and_std(data):
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


