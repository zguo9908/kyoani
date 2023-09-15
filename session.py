# single session data parsing tool
import math
import os
import statistics
from statistics import mean

import numpy
import numpy as np
import pandas as pd

#from animal import Animal

class Session:
    def __init__(self, animal, file_path, task_params):
        # within session lists

        self.animal = animal
        self.file_path = file_path
        self.task_params = task_params

        if task_params == "curr_params":
            self.optimal_wait = [1.74, 3.45]
        elif task_params == "old_params":
            self.optimal_wait = [1.52, 2.93]

        self.block_type = []
        self.valid_block_type = []
        self.holding = []  # avg time of holding licks for all blocks
        self.holding_s = []  # avg time of holding licks during s blocks
        self.holding_l = []  # avg time of holding licks during l blocks

        self.opt_diff = []
        self.opt_diff_s = []  # avg diff of holding licks and optimal lick time of s block
        self.opt_diff_l = []  # avg diff of holding licks and optimal lick time of l block

        self.perc_rewarded = []
        self.perc_rewarded_s = []
        self.perc_rewarded_l = []

        self.prob_at_lick_s = []
        self.prob_at_lick_l = []

        self.blk_start_var = []  # variance of waiting time for the last 10 trials before block switch
        self.blk_end_var = []  # variance of waiting time for the first 10 trials after block switch

        self.session_reward_rate = []
        self.session_holding_times = []

        self.blk_window = 5

        self.bg_repeats_s = []
        self.bg_repeats_l = []

        self.blk_start_slope = []
        self.blk_end_slope = []

        self.animal.holding_diff = []

    def parseSessionStats(self):
        session_data = pd.read_csv(self.file_path, skiprows=3)
        session_data["session_time"] = session_data["session_time"] - session_data["session_time"][0]

        # find block switches

        session_data['blk_change'] = session_data.loc[:, 'block_num'].diff()
        session_data['total_volume_received'] = session_data['total_reward']*5
        session_blks = session_data[session_data['blk_change'] != 0]
        self.session_reward_rate = session_data['total_volume_received'].div(session_data['curr_trial_time'])

        for k in range(1, session_blks.shape[0] - 1):
            #             print(f'{i}+{j}+{k}')
            curr_blk = session_data[session_data.block_num == k]

            # find number of trials in this block
            blk_trial_num = max(curr_blk.session_trial_num) - min(curr_blk.session_trial_num) + 1

            blk_rewarded_trial = curr_blk.loc[(curr_blk['key'] == 'reward') & (curr_blk['value'] == 1)]
            blk_rewarded_trial_num = len(blk_rewarded_trial)
            perc_rewarded_perf = blk_rewarded_trial_num / blk_trial_num
            self.perc_rewarded.append(perc_rewarded_perf)

            curr_licks_during_wait = curr_blk.loc[(curr_blk['key'] == 'lick') & (curr_blk['state'] == 'in_wait')]
            curr_blk_mean_reward_time = int(curr_blk.mean_reward_time.iloc[0])
            prob_at_lick = curr_blk['curr_reward_prob']
            prob_at_lick = prob_at_lick.astype('float')
            #             prob_at_lick = prob_at_lick.rename(columns=lambda x: x+1)

            blk_cp = curr_blk.copy()
            blk_cp['next_state'] = blk_cp['state'].shift(-1)

            curr_rewarded_licks = blk_cp.loc[(blk_cp['key'] == 'lick') &
                                             (blk_cp['state'] == 'in_wait') &
                                             (blk_cp['next_state'] == 'in_consumption')]
            miss_trials = blk_cp.loc[(blk_cp['state'] == 'in_wait') &
                                     (blk_cp['next_state'] == 'trial_ends') &
                                     (blk_cp['key'] == 'wait')]

            lick_at_bg = blk_cp.loc[((blk_cp['state'] == 'in_background') &
                                     (blk_cp['key'] == 'lick') &
                                     (blk_cp['value'] == 1))]

            trials_lick_at_bg = lick_at_bg.drop_duplicates(subset=['session_trial_num'])
            blk_bg_repeat_perc = len(trials_lick_at_bg) / blk_trial_num
            # print(f'number of trials with repeat bg triggers{len(trials_lick_at_bg)}')
            #             print(len(miss_trials))
            blk_missed_perc = len(miss_trials) / blk_trial_num
            repeat_counts = lick_at_bg['session_trial_num'].value_counts()
            # print(repeat_counts)
            counts_column = repeat_counts.values
            # Calculate the average
            average_repeats = counts_column.mean()
            print("Average number of repeats: ", average_repeats)

            exclude_values = trials_lick_at_bg['session_trial_num']
            # print(len(exclude_values))
            good_trials = blk_cp[~blk_cp['session_trial_num'].isin(exclude_values)]
            good_trials_num = len(good_trials.drop_duplicates(subset=['session_trial_num']))
            # print("number of good trials: ", len(good_trials))
            # good_trials = blk_cp[exclude_condition].drop_duplicates(subset=['session_trial_num'])

            # print(len(good_trials))
            # print(len(trials_lick_at_bg))
            print(blk_trial_num == good_trials_num + len(trials_lick_at_bg))

            prob_rewarded_licks = blk_cp['curr_reward_prob']
            prob_rewarded_licks = prob_rewarded_licks.astype('float')
            mean_prob_at_lick = prob_rewarded_licks.mean()
            #             print(mean_prob_at_lick )

            if curr_blk_mean_reward_time == 3:
                curr_opt_wait = self.optimal_wait[1]
                self.block_type.append('l')
            elif curr_blk_mean_reward_time == 1:
                curr_opt_wait = self.optimal_wait[0]
                self.block_type.append('s')
            else:
                break

            if 'curr_wait_time' in curr_licks_during_wait:
                licks = curr_licks_during_wait.curr_wait_time
                #                 all_holding_diff = licks - curr_opt_wait
                #                 all_holding_perf = 1-abs(all_holding_diff)/curr_opt_wait
                self.session_holding_times.append(licks)
                # variance and linear fit slope for first and last 10 trials in each block.
                if len(licks) >= 10 and blk_trial_num >= 10:
                    self.valid_block_type.append(self.block_type[k-1])
                    blk_start_trials = licks[:self.blk_window]
                    blk_end_trials = licks[-self.blk_window:]
                    # print(len(licks))
                    # print(len(blk_start_trials))
                    self.blk_start_var.append(statistics.variance(blk_start_trials))
                    self.blk_end_var.append(statistics.variance(blk_end_trials))

                    self.blk_start_slope.append(np.polyfit(range(0, len(blk_start_trials)), blk_start_trials, 1))
                    self.blk_end_slope.append(np.polyfit(range(0, len(blk_end_trials)), blk_end_trials, 1))

                # block mean analysis
                lick_mean = licks.mean()
                #                 holding.append(lick_mean)
                lick_diff = curr_licks_during_wait.curr_wait_time - curr_opt_wait
                mean_lick_diff = lick_diff.mean()
                self.opt_diff.append(mean_lick_diff)

                # performance for holding lick
                holding_perf = 1 - abs(mean_lick_diff) / curr_opt_wait

                if self.block_type[k - 1] == 's':
                    self.animal.all_holding_s.extend(licks)
                    self.holding_s.append(lick_mean)
                    self.opt_diff_s.append(mean_lick_diff)
                    self.perc_rewarded_s.append(perc_rewarded_perf)
                    self.prob_at_lick_s.append(mean_prob_at_lick)
                    self.bg_repeats_s.append(blk_bg_repeat_perc)
                    if not math.isnan(holding_perf):
                        self.animal.holding_perf_s.append(holding_perf)
                    else:
                        self.animal.holding_perf_s.append(np.nan)
                    if not math.isnan(blk_missed_perc):
                        self.animal.blk_miss_perc_s.append(blk_missed_perc)
                    else:
                        self.animal.blk_miss_perc_s.append(np.nan)
                else:
                    self.animal.all_holding_l.extend(licks)
                    self.holding_l.append(lick_mean)
                    self.opt_diff_l.append(mean_lick_diff)
                    self.perc_rewarded_l.append(perc_rewarded_perf)
                    self.prob_at_lick_l.append(mean_prob_at_lick)
                    self.bg_repeats_l.append(blk_bg_repeat_perc)

                    if not math.isnan(holding_perf):
                        self.animal.holding_perf_l.append(holding_perf)
                    else:
                        self.animal.holding_perf_l.append(np.nan)
                    if not math.isnan(blk_missed_perc):
                        self.animal.blk_miss_perc_l.append(blk_missed_perc)
                    else:
                        self.animal.blk_miss_perc_l.append(np.nan)

            else:
                break


    def updateSessionStats(self):
        if len(self.holding_s) > 0:
            self.animal.blk_holding_s.extend(x for x in self.holding_s if not math.isnan(x))
            self.animal.holding_s_blk.append(mean(self.holding_s))
        else:
            self.animal.holding_s_blk.append(np.nan)
        if len(self.holding_l) > 0:
            self.animal.blk_holding_l.extend(x for x in self.holding_l if not math.isnan(x))
            self.animal.holding_l_blk.append(mean(self.holding_l))
        else:
            self.animal.holding_l_blk.append(np.nan)

        if not np.isnan(self.animal.holding_l_blk).any() and not np.isnan(self.animal.holding_s_blk).any():
            self.animal.holding_diff.append(self.animal.holding_s_blk[-1] - self.animal.holding_l_blk[-1])
        self.animal.bg_restart_s_blk.extend(self.bg_repeats_s)
        self.animal.bg_restart_l_blk.extend(self.bg_repeats_l)
        self.animal.bg_restart_s.append(mean(self.bg_repeats_s))
        self.animal.bg_restart_l.append(mean(self.bg_repeats_l))



        if len(self.opt_diff_s) > 0:
            self.animal.opt_diff_s.append(mean(self.opt_diff_s))
        else:
            self.animal.opt_diff_s.append(np.nan)
        if len(self.opt_diff_l) > 0:
            self.animal.opt_diff_l.append(mean(self.opt_diff_l))
        else:
            self.animal.opt_diff_l.append(np.nan)

        if len(self.perc_rewarded_s) > 0:
            self.animal.perc_rewarded_s.append(mean(self.perc_rewarded_s))
        else:
            self.animal.perc_rewarded_s.append(np.nan)
        if len(self.perc_rewarded_l) > 0:
            self.animal.perc_rewarded_l.append(mean(self.perc_rewarded_l))
        else:
            self.animal.perc_rewarded_l.append(np.nan)

        if len(self.prob_at_lick_s) > 0:
            self.animal.lick_prob_s.append(mean(self.prob_at_lick_s))
        else:
            self.animal.lick_prob_s.append(np.nan)

        if len(self.prob_at_lick_l) > 0:
            self.animal.lick_prob_l.append(mean(self.prob_at_lick_l))
        else:
            self.animal.lick_prob_l.append(np.nan)

        if len(self.blk_start_var) > 0:
            # print(f"number of start var {self.blk_start_var}")
            valid_start_var = [x for x in self.blk_start_var if not math.isnan(x)]
            # print(np.count_nonzero(np.isnan(self.blk_start_var)))
            valid_end_var = [x for x in self.blk_end_var if not math.isnan(x)]
            # print(np.count_nonzero(np.isnan(self.blk_end_var)))
            self.animal.blk_start_var.extend(valid_start_var)
            self.animal.blk_end_var.extend(x for x in self.blk_end_var if not math.isnan(x))
            self.animal.blk_start_slope.append(self.blk_start_slope)
            self.animal.blk_end_slope.append(self.blk_end_slope)
            num_block = len(self.block_type)
            # print(f"number of valid start var {len(valid_start_var)}")
            # print(f"number of valid blocks {len(self.valid_block_type)}")
            print(self.valid_block_type)
            for i in range(0, len(self.valid_block_type)-1):
                # print(f"i is now {i}")
                # interested in start var for all blks except the first one
                if i >= 1 and self.block_type[i] == "s":
                    # print(f"{i} block l-s start")
                    self.animal.ls_blk_start_var.append(valid_start_var[i])
                elif i >= 1 and self.block_type[i] == "l":
                    # print(f"{i} block s-l start")
                    self.animal.sl_blk_start_var.append(valid_start_var[i])

                if i < num_block-1 and self.block_type[i] == "s":
                    # print(f"{i} block s-l end")
                    self.animal.sl_blk_end_var.append(valid_end_var[i])
                elif i < num_block-1 and self.block_type[i] == "l":
                    # print(f"{i} block l-s end")
                    self.animal.ls_blk_end_var.append(valid_end_var[i])




