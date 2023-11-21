# class Animal holds relevant information about a mouse
import os
import statistics

from session import Session

class ShortProperties:
    def __init__(self):
        self.holding_by_block = []
        self.holding_mean = []
        self.opt_diff = []
        self.perc_rewarded = []
        self.holding = []
        self.all_holding = []
        self.all_holding_list = []

        self.holding_std = []
        self.holding_sem = []
        self.holding_median = []
        self.holding_q25 = []
        self.holding_q75 = []

        self.lick_prob = []
        self.holding_perf = []

        self.miss_perc = []

        self.reflex_lick_perc = []
        self.non_reflexive = []
        self.non_reflexive_mean = []
        self.non_reflexive_std = []

        self.moving_average = []  # waiting time across all trials smoothed by a window size
        self.moving_average_var = []

        self.stable = []  # times were performance was stable (mean across a window

        self.bg_restart = []

        self.bg_restart_all = []

        self.bg_restart_licks = []
        self.all_holding_good = []
        self.holding_mean_good = []
        self.opt_diff_good = []
        self.perc_rewarded_good = []
        self.prob_at_lick_good = []
        self.all_holding_index = []
        self.all_holding_by_session = []
        self.mean_background_length_from_consumption = []
        self.mean_background_lick_from_consumption = []
        self.perc_bout_into_background = []
        self.mean_background_length = []

class LongProperties:
    def __init__(self):
        self.holding_by_block = []
        self.holding_mean = []
        self.opt_diff = []
        self.perc_rewarded = []
        self.holding = []
        self.all_holding = []
        self.all_holding_list = []

        self.holding_std = []
        self.holding_sem = []
        self.holding_median = []
        self.holding_q25 = []
        self.holding_q75 = []

        self.lick_prob = []
        self.holding_perf = []

        self.miss_perc = []

        self.reflex_lick_perc = []
        self.non_reflexive = []
        self.non_reflexive_mean = []
        self.non_reflexive_std = []

        self.moving_average = []  # waiting time across all trials smoothed by a window size
        self.moving_average_var = []

        self.stable = []  # times were performance was stable (mean across a window

        self.bg_restart = []

        self.bg_restart_all = []

        self.bg_restart_licks = []
        self.all_holding_good = []
        self.holding_mean_good = []
        self.opt_diff_good = []
        self.perc_rewarded_good = []
        self.prob_at_lick_good = []
        self.all_holding_index = []
        self.all_holding_by_session = []
        self.mean_background_length_from_consumption = []
        self.mean_background_lick_from_consumption = []
        self.perc_bout_into_background = []
        self.mean_background_length = []

class Animal:
    def __init__(self, name, default, change, task_params, optimal_wait):
        self.holding_l_by_block = []
        self.holding_s_by_block = []
        self.session_num = None
        self.default_session_num = None
        self.change_session_num = None
        self.name = name
        self.default = default
        self.change = change
        self.task_params = task_params
        self.optimal_wait = optimal_wait
        self.sessions = []
        self.default_sessions = []
        self.change_sessions = []

        self.short_properties = ShortProperties()
        self.long_properties = LongProperties()

        # mean of holding during blocks or sessions of a timescape type
        self.holding_s_mean = []
        self.holding_l_mean = []
        self.opt_diff_s = []
        self.opt_diff_l = []

        self.perc_rewarded = []
        self.perc_rewarded_s = []  # across days
        self.perc_rewarded_l = []

        # all holding times.
        self.holding_s = []
        self.holding_l = []

        self.all_holding_s = []
        self.all_holding_l = []
        self.all_holding_s_list = []
        self.all_holding_l_list = []

        self.holding_s_std = []
        self.holding_l_std = []
        self.holding_s_sem = []
        self.holding_l_sem = []
        self.holding_s_median = []
        self.holding_l_median = []
        self.holding_s_q25 = []
        self.holding_s_q75 = []
        self.holding_l_q25 = []
        self.holding_l_q75 = []

        self.lick_prob_s = []
        self.lick_prob_l = []

        self.holding_perf_s = []
        self.holding_perf_l = []

        self.miss_perc_s = []
        self.miss_perc_l = []

        self.reflex_lick_perc_s = []
        self.reflex_lick_perc_l = []
        self.non_reflexive_l_mean = []
        self.non_reflexive_l_std = []
        self.non_reflexive_s_mean = []
        self.non_reflexive_s_std = []

        self.sl_blk_start_var = []
        self.ls_blk_start_var = []
        self.sl_blk_end_var = []
        self.ls_blk_end_var = []

        self.moving_average_s = []  # waiting time across all trials smoothed by a window size
        self.moving_average_l = []
        self.moving_average_s_var = []
        self.moving_average_l_var = []

        self.stable_s = []  # times were performance was stable (mean across a window
        self.stable_l = []

        self.blk_start_var = []
        self.blk_end_var = []
        self.blk_start_slope = []
        self.blk_end_slope = []

        self.bg_restart_s = []
        self.bg_restart_l = []
        self.bg_restart_s_all = []
        self.bg_restart_l_all = []
        self.bg_restart_licks_s = []
        self.bg_restart_licks_l = []
        self.session_list = []

        self.all_holding_s_good = []
        self.holding_s_mean_good = []

        self.all_holding_l_good = []
        self.holding_l_mean_good = []

        self.opt_diff_s_good = []
        self.opt_diff_l_good = []

        self.perc_rewarded_s_good = []
        self.perc_rewarded_l_good = []

        self.prob_at_lick_s_good = []
        self.prob_at_lick_l_good = []

        self.all_holding_s_index = []
        self.all_holding_l_index = []
        self.all_holding_s_by_session = []
        self.all_holding_l_by_session = []

        self.sessoion_index = []

        self.mean_consumption_length = []
        self.mean_consumption_licks = []

        self.mean_background_length_from_consumption_s = []
        self.mean_background_lick_from_consumption_s = []
        self.perc_bout_into_background_s = []
        self.mean_background_length_from_consumption_l = []
        self.mean_background_lick_from_consumption_l = []
        self.perc_bout_into_background_l = []
        self.mean_background_length_s = []
        self.mean_background_length_l = []

        self.reverse_index = -1

    def allSession(self, path, stage, has_block):
        os.chdir(path)
        if stage == 'default':
            self.default_session_num = len(self.default_sessions)
            print(f'number of sessions processing {self.default_session_num}')
            curr_session_num = self.default_session_num
            curr_sessions = self.default_sessions
        else:
            self.change_session_num = len(self.change_sessions)
            print(f'number of sessions processing {self.default_session_num}')
            curr_session_num = self.change_session_num
            curr_sessions = self.change_sessions


        for j in range(curr_session_num):
            curr_session_path = path + '\\' + curr_sessions[j]
            os.chdir(curr_session_path)
            file_path = curr_session_path + '\\' + os.listdir()[0]
            curr_session = Session(self, file_path, has_block, self.task_params, self.optimal_wait)
            curr_session.parseSessionStats()
            curr_session.updateSessionStats()
            # self.session_index.append(self.all)
            self.session_list.append(curr_session)

    # print(f'std for l session {self.holding_l_std}')
    # print(self.holding_s_std)


    # this function will take moving average across windows of trials
    def getMovingAvg(self, window_size):
        curr_all_s = self.all_holding_s
        curr_all_l = self.all_holding_l
        i = 0
        # Initialize an empty list to store moving averages
        while i < len(curr_all_s) - window_size + 1:
            window = curr_all_s[i: i + window_size]
            window_average = round(sum(window) / window_size, 2)
            self.moving_average_s.append(window_average)
            self.moving_average_s_var.append(statistics.variance(window))
            i += 1
        i = 0
        while i < len(curr_all_l) - window_size + 1:
            window = curr_all_l[i: i + window_size]
            window_average = round(sum(window) / window_size, 2)
            self.moving_average_l.append(window_average)
            self.moving_average_l_var.append(statistics.variance(window))
            i += 1

    # this function will compute block waiting time average as training goes
    def getBlockWaiting(self):
        curr_all_s = self.holding_s_mean
        # self.holding_s_by_block = []
        curr_all_l = self.holding_l_mean
        # self.holding_l_by_block = []
        for j in range(len(curr_all_s)):
            sublist = curr_all_s[:j + 1]
            mean = sum(sublist) / len(sublist)
            self.holding_s_by_block.append(mean)
        for k in range(len(curr_all_l)):
            sublist = curr_all_l[:k + 1]
            mean = sum(sublist) / len(sublist)
            self.holding_l_by_block.append(mean)
