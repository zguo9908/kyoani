# class Animal holds relevant information about a mouse
import os
from session import Session

class Animal:
    def __init__(self, name):
        self.holding_l_by_block =[]
        self.holding_s_by_block = []
        self.session_num = None
        self.name = name
        self.sessions = []

        self.holding_s_blk = []
        self.holding_l_blk = []
        self.opt_diff_s = []
        self.opt_diff_l = []

        self.perc_rewarded = []
        self.perc_rewarded_s = []  # across days
        self.perc_rewarded_l = []

        self.blk_holding_s = []
        self.blk_holding_l = []

        self.all_holding_s = []
        self.all_holding_l = []

        self.lick_prob_s = []
        self.lick_prob_l = []

        self.holding_perf_s = []
        self.holding_perf_l = []

        self.blk_miss_perc_s = []
        self.blk_miss_perc_l = []

        self.moving_average_s = [] #waiting time across all trials smoothed by a window size
        self.moving_average_l = []

        self.stable_s = [] #times were performance was stable (mean across a window
        self.stable_l = []


    def allSession(self, path):
        self.session_num = len(self.sessions)
        # print(self.session_num)
        for j in range(self.session_num):
            curr_session_path = path + '\\' + self.sessions[j]
            os.chdir(curr_session_path)
            file_path = curr_session_path + '\\' + os.listdir()[0]
            curr_session = Session(self, file_path)
            curr_session.parseSessionStats()
            print(f'parsing animal {self.name} stats')
            curr_session.updateSessionStats()
            print(f'updating animal {self.name} stats')


    # this function will take moving average across windows of trials
    def getMovingAvg(self, window_size):
        curr_all_s = self.all_holding_s
        curr_all_l = self.all_holding_l
        i = 0
        # Initialize an empty list to store moving averages
        while i < len(curr_all_s) - window_size + 1:
            window = curr_all_s[i: i + window_size]
            window_average = round(sum(window) / window_size, 2)
            #         print(window_average)
            self.moving_averages_s.append(window_average)
            i += 1
        i = 0
        while i < len(curr_all_l) - window_size + 1:
            window = curr_all_l[i: i + window_size]
            window_average = round(sum(window) / window_size, 2)
            self.moving_averages_l.append(window_average)
            i += 1

    # this function will compute block waiting time average as training goes
    def getBlockWaiting(self):
        curr_all_s = self.blk_holding_s
        # self.holding_s_by_block = []
        curr_all_l = self.blk_holding_l
        # self.holding_l_by_block = []
        for j in range(len(curr_all_s)):
            sublist = curr_all_s[:j + 1]
            mean = sum(sublist) / len(sublist)
            self.holding_s_by_block.append(mean)
        for k in range(len(curr_all_l)):
            sublist = curr_all_l[:k + 1]
            mean = sum(sublist) / len(sublist)
            self.holding_l_by_block.append(mean)

