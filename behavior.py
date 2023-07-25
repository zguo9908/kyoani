import math
import os

from animal import Animal
from session import Session
from scipy.stats import ttest_ind
class BehaviorAnalysis:

    def __init__(self, optimal_wait, task_type, task_params):
        self.task_type = task_type
        self.task_params = task_params
        self.optimal_wait = optimal_wait
        self.path = os.path.normpath(r'D:\behavior_data') + "\\" + task_params
        print(self.path)
        os.chdir(self.path)
        self.animal_list = os.listdir()
        self.animal_num = len(self.animal_list)
        self.mice = [] # this stores the animal object

        self.block_diff = []
        self.stable_block_diff = []

    def getStableTimes(self, mouse):
        for j in range(len(mouse.moving_average_s_var)):
            if not math.isnan(mouse.moving_average_s_var[j]) and mouse.moving_average_s_var[j]< 1:
                mouse.stable_s.append(mouse.moving_average_s[j])

        for k in range(len(mouse.moving_average_l_var)):
            if not math.isnan(mouse.moving_average_l_var[k]) and mouse.moving_average_l_var[k] < 1:
                mouse.stable_l.append(mouse.moving_average_l[k])
        print(len(mouse.stable_l))

    def allAnimal(self):
        for i in range(self.animal_num):
            animal = self.animal_list[i]
            curr_animal = Animal(animal, self.task_params)
            self.mice.append(curr_animal)
            curr_path = self.path + "\\" + animal
            os.chdir(curr_path)
            session_list = os.listdir()
            # filter all the items that are regular
            sessions = [session for session in session_list if self.task_type in session]
            curr_animal.sessions = sessions

            curr_animal.allSession(curr_path)
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
        for i in range(self.animal_num):
            t_stat, p_value = ttest_ind(self.mice[i].stable_s, self.mice[i].stable_l)
            self.stable_block_diff.append(p_value)
            t_stat, p_value = ttest_ind(self.mice[i].blk_holding_s, self.mice[i].blk_holding_l)
            self.block_diff.append(p_value)
        print("p-vals for different blocks are")
        print(self.block_diff)
        print(self.stable_block_diff)


    # # test the effect of switching block to see if how covert changes are observed by animals
    # def testBlockSwitch(self):
    #     for i in range(self.animal_num):
