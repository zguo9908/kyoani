import json
import math
import os

import numpy as np
from matplotlib import pyplot as plt
from pkg_resources import resource_string

from animal import Animal
from session import Session
from scipy.stats import ttest_ind


class BehaviorAnalysis:
    def get_exp_config(self):
        """Get experiment config from json"""
        with open(f'{self.exp_name}.json', 'r') as json_file:
            config_data = json.load(json_file)
        return config_data

    def __init__(self, exp_name, optimal_wait, task_type, has_block, task_params):
        self.exp_name = exp_name
        self.exp_config = self.get_exp_config()
        self.animal_assignment = self.exp_config['timescape']
        self.task_type = task_type
        self.task_params = task_params
        self.has_block = has_block
        self.optimal_wait = optimal_wait
        if self.has_block:
            self.path = os.path.normpath(r'D:\behavior_data') + "\\" + "blocks" + "\\" + task_params
        else:
            self.path = os.path.normpath(r'D:\behavior_data') + "\\" + "no_blocks" + "\\" + task_params
        print(self.path)
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
            else:
                self.short_mice_list.append(mouse)
                self.short_session_mean.append(self.mice[i].holding_s_mean)
                self.short_session_nonimpulsive_mean.append(self.mice[i].non_reflexive_s_mean)
                self.short_consumption_length.append(self.mice[i].mean_consumption_length)
                self.short_bg_repeat.append(self.mice[i].bg_restart_s)
                self.short_impulsive_perc.append(self.mice[i].reflex_lick_perc_s)

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

        # bg repeats plot
        long_impusive_perc_averages, long_impusive_perc_std = calculate_padded_averages_and_std(self.long_impulsive_perc)
        short_impusive_perc_averages, short_impusive_perc_std = calculate_padded_averages_and_std(self.short_impulsive_perc)

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
        ax.set_ylabel('wait times')
        ax.legend()
        plt.savefig('impulsive licking percentage.svg')
        plt.close()

        # bg repeats plot
        long_repeat_averages, long_std_repeat = calculate_padded_averages_and_std(self.long_bg_repeat)
        short_repeat_averages, short_std_repeat = calculate_padded_averages_and_std(self.short_bg_repeat)

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
        ax.set_ylabel('wait times')
        ax.legend()
        plt.savefig('bg repeats for long vs short cohorts.svg')
        plt.close()

        # session plots
        long_session_averages, long_std_session = calculate_padded_averages_and_std(self.long_session_mean)
        short_session_averages, short_std_session = calculate_padded_averages_and_std(self.short_session_mean)

        fig, ax = plt.subplots()
        # Plot the line graph for long sessions
        x = list(range(1, len(long_session_averages) + 1))
        y = long_session_averages
        ax.plot(x, y, marker='o', label='Average_long', color='red')

        # Shade the area around the line plot to represent the standard deviation for long sessions
        ax.fill_between(x, [mean - std for mean, std in zip(y, long_std_session)],
                        [mean + std for mean, std in zip(y, long_std_session)], alpha=0.5,
                        label='Standard Deviation_long',
                        color='#FFAAAA')
        # Plot the line graph for short sessions
        x = list(range(1, len(short_session_averages) + 1))
        y = short_session_averages
        ax.plot(x, y, marker='o', label='Average_short', color='blue')

        # Shade the area around the line plot to represent the standard deviation for short sessions
        ax.fill_between(x, [mean - std for mean, std in zip(y, short_std_session)],
                        [mean + std for mean, std in zip(y, short_std_session)], alpha=0.5,
                        label='Standard Deviation_short',
                        color='lightblue')
        ax.set_xlabel('session #')
        ax.set_ylabel('wait times')
        ax.legend()
        plt.savefig('session average for long vs short cohorts.svg')
        plt.close()

        # non_impulsive licks
        long_averages, long_std_deviation = calculate_padded_averages_and_std(self.long_session_nonimpulsive_mean)
        short_averages, short_std_deviation = calculate_padded_averages_and_std(self.short_session_nonimpulsive_mean)
        fig, ax = plt.subplots()
        # Plot the line graph for long sessions
        x = list(range(1, len(long_averages) + 1))
        y = long_averages
        ax.plot(x, y, marker='o', label='Average_long', color='red')

        # Shade the area around the line plot to represent the standard deviation for long sessions
        ax.fill_between(x, [mean - std for mean, std in zip(y, long_std_deviation)],
                        [mean + std for mean, std in zip(y, long_std_deviation)], alpha=0.5, label='Standard Deviation_long',
                        color='#FFAAAA')
        # Plot the line graph for short sessions
        x = list(range(1, len(short_averages) + 1))
        y = short_averages
        ax.plot(x, y, marker='o', label='Average_short', color='blue')

        # Shade the area around the line plot to represent the standard deviation for short sessions
        ax.fill_between(x, [mean - std for mean, std in zip(y, short_std_deviation)],
                        [mean + std for mean, std in zip(y, short_std_deviation)], alpha=0.5, label='Standard Deviation_short',
                        color='lightblue')

        ax.set_xlabel('session #')
        ax.set_ylabel('wait times')
        ax.legend()
        plt.savefig('non impulsive session licks average for long vs short cohorts.svg')
        plt.close()

        # non_impulsive licks
        long_com_averages, long_com_std = calculate_padded_averages_and_std(self.long_consumption_length)
        short_com_averages, short_com_std = calculate_padded_averages_and_std(self.short_consumption_length)
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
        ax.set_ylabel('wait times')
        ax.legend()
        plt.savefig('consumption times long vs short cohorts.svg')
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

    return averages, std_deviation