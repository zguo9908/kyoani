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
            else:
                self.short_mice_list.append(mouse)
                self.short_session_mean.append(self.mice[i].holding_s_mean)

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

        # Step 1: Pad the sublists with the average of each sublist
        max_length_long = max(len(animal_sessions) for animal_sessions in self.long_session_mean)
        long_session_averages = [sum(animal_sessions) / len(animal_sessions) if len(animal_sessions) > 0 else 0 for animal_sessions in self.long_session_mean]
        padded_data = [[entry if entry != 0 else avg for entry in sublist] + [avg] * (max_length_long - len(sublist)) for
                       sublist, avg in zip(self.long_session_mean, long_session_averages)]

        # Step 2: Calculate the average for each position in the padded data
        long_session_averages = [sum(entry) / len(padded_data) for entry in zip(*padded_data)]

        # Step 3: Create a figure and axis
        fig, ax = plt.subplots()

        # Plot the line graph
        x = list(range(1, len(padded_data[0]) + 1))
        y = long_session_averages
        ax.plot(x, y, marker='o', label='Average_long', color = 'red')

        # Step 4: Shade the area around the line plot to represent the standard deviation
        std_deviation = np.std(padded_data, axis=0, ddof=0)
        ax.fill_between(x, [mean - std for mean, std in zip(y, std_deviation)],
                        [mean + std for mean, std in zip(y, std_deviation)], alpha=0.5, label='Standard Deviation_long',
                        color='#FFAAAA')

        max_length_short = max(len(animal_sessions) for animal_sessions in self.short_session_mean)
        short_session_averages = [sum(animal_sessions) / len(animal_sessions) if len(animal_sessions) > 0 else 0 for animal_sessions in self.short_session_mean]
        padded_data = [[entry if entry != 0 else avg for entry in sublist] + [avg] * (max_length_short - len(sublist)) for
                       sublist, avg in zip(self.short_session_mean, short_session_averages)]

        # Step 2: Calculate the average for each position in the padded data
        short_session_averages = [sum(entry) / len(padded_data) for entry in zip(*padded_data)]

        # Plot the line graph
        x = list(range(1, len(padded_data[0]) + 1))
        y = short_session_averages
        ax.plot(x, y, marker='o', label='Average_short', color='blue')

        # Step 4: Shade the area around the line plot to represent the standard deviation
        std_deviation = np.std(padded_data, axis=0, ddof=0)
        ax.fill_between(x, [mean - std for mean, std in zip(y, std_deviation)],
                        [mean + std for mean, std in zip(y, std_deviation)], alpha=0.5, label='Standard Deviation_short',
                        color='lightblue')

        # Customize the plot
        ax.set_xlabel('session #')
        ax.set_ylabel('wait times')
        #ax.set_title('Line Plot with Standard Deviation')
        ax.legend()

        # Show the plot
        plt.savefig('padded session average for long vs short cohorts.svg')

        # test the effect of switching block to see if how covert changes are observed by animals
        # def testBlockSwitch(self):
        #     for i in range(self.animal_num):
