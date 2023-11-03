# single session data parsing tool
import math
import os
import statistics
from statistics import mean
from statistics import median
from scipy.stats import sem
import numpy
import numpy as np
import pandas as pd


def is_float(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def getRewardedPerc(df, trial_num):
    rewarded_trial = df.loc[(df['key'] == 'reward') & (df['value'] == 1)]
    perc_rewarded_perf = len(rewarded_trial) / trial_num
    return perc_rewarded_perf

# a lick bout based analysis of consumption licks
def getConsumptionStats(all_df, bout_interval, min_lick_count_consumption):
    df = all_df.loc[(all_df['key'] == 'lick')]
    df = df.sort_values(by='session_time')

    # print(df.head())
    df['time_diff'] = df['session_time'].diff()
    # Define start of a bout (time difference > 0.3 seconds)
    bout_start = df['time_diff'] > bout_interval
    # Create a bout number
    df['bout_number'] = bout_start.cumsum()

    # Calculate the start and end times of each bout
    bout_indices = df.groupby('bout_number').agg(session_time_first=('session_time', 'first'),
                                                 session_time_last=('session_time', 'last'),
                                                 state_first=('state', 'first')).reset_index()

    # Filter bouts that meet the lick time difference criterion and start in "in_consumption"
    lick_bouts = bout_indices[bout_indices['state_first'] == 'in_consumption']

    # Initialize variables to store results
    consumption_lengths = []
    background_lengths = []
    lick_counts_consumption = []
    lick_counts_background = []
    perc_bout_into_background = np.nan
    if not lick_bouts.empty:
        start_time = lick_bouts['session_time_first']
        end_time = lick_bouts['session_time_last']
        # Set a threshold for minimum consumption duration or number of licks
        for bout_number, start_time, end_time in zip(lick_bouts['bout_number'], start_time, end_time):
            # Calculate the length of consumption lick bout
            consumption_length = end_time - start_time

            # Calculate the length of licks that span into the next background state
            next_background = df[(df['state'] == 'in_background') & (df['session_time'] <= end_time) & (df['session_time'] > start_time)]
            if not next_background.empty:
                next_background_length = next_background['session_time'].iloc[-1] - \
                                         next_background['session_time'].iloc[-0]
            else:
                next_background_length = 0

            # Calculate the number of licks for the entire consumption period
            lick_count_consumption = df[(df['session_time'] >= start_time) & (df['session_time'] <= end_time)
                                        & (df['value'] == 1)]['session_time'].count()

            # Calculate the number of licks in the next background state
            lick_count_background = next_background[next_background['value'] == 1]['session_time'].count()

            # Check if the bout meets the minimum duration or lick count criteria for consumption
            if lick_count_consumption >= min_lick_count_consumption:
                consumption_lengths.append(consumption_length)
                background_lengths.append(next_background_length)
                lick_counts_consumption.append(lick_count_consumption)
                lick_counts_background.append(lick_count_background)

        # Add the calculated values to the lick_bouts DataFrame
        # lick_bouts['consumption_length'] = consumption_lengths
        # lick_bouts['background_length'] = background_lengths
        # lick_bouts['lick_count_consumption'] = lick_counts_consumption
        # lick_bouts['lick_count_background'] = lick_counts_background
        true_count = sum(background_lengths)
        perc_bout_into_background = true_count/len(consumption_lengths)
        # # Display the lick bouts and associated metrics
        # print("Lick Bouts:")
        # print(lick_bouts)
    else:
        print("No 'in_consumption' bouts found.")
    # Add the calculated values to the lick_bouts DataFrame
    print(f'consumption lengths {consumption_lengths}')
    print(f'lick length in the background during bouts {background_lengths}')
    print(f'lick counts in the background during bouts {lick_counts_background}')

    mean_consumption_length = mean(consumption_lengths) if len(consumption_lengths) >0 else np.nan
    mean_consumption_licks = mean(lick_counts_consumption) if len(lick_counts_consumption)>0 else np.nan
    mean_background_length = mean(background_lengths) if len(background_lengths)>0 else np.nan
    mean_background_licks = mean(lick_counts_background) if len(lick_counts_background)>0 else np.nan
    # Display the lick bouts and associated metrics
    # lick_bouts['consumption_length'] = consumption_lengths
    # lick_bouts['background_length'] = background_lengths
    # lick_bouts['lick_count_consumption'] = lick_counts_consumption
    # lick_bouts['lick_count_background'] = lick_counts_background
    # print("Lick Bouts:")
    # print(lick_bouts)

    return consumption_lengths, background_lengths, lick_counts_consumption, lick_counts_background, \
           mean_consumption_length, mean_consumption_licks, mean_background_length, mean_background_licks,\
           perc_bout_into_background

def getMissedTrials(df, trial_num):
    miss_trials = df.loc[(df['state'] == 'in_wait') &
                         (df['next_state'] == 'trial_ends') &
                         (df['key'] == 'wait')]
    miss_trials_num = len(miss_trials)
    missed_perc = miss_trials_num/trial_num
    return missed_perc, miss_trials

def getProbRewarded(df):
    #prob_rewarded_licks = df[df['curr_reward_prob'].notna()]

    df['curr_reward_prob'] = pd.to_numeric(df['curr_reward_prob'], errors='coerce')

    # Create a boolean column indicating which values could be converted to float
    prob_rewarded_licks = ~df['curr_reward_prob'].isna()
    prob_rewarded_licks = prob_rewarded_licks.astype('float')
    mean_prob_at_lick = prob_rewarded_licks.mean()
    return mean_prob_at_lick

def getBackgroundLicks(df, trial_num):
    lick_at_bg = df.loc[((df['state'] == 'in_background') &
                             (df['key'] == 'lick') &
                             (df['value'] == 1))]

    trials_lick_at_bg = lick_at_bg.drop_duplicates(subset=['session_trial_num'])
    repeat_counts = lick_at_bg['session_trial_num'].value_counts()
    counts_column = repeat_counts.values
    # Calculate the average
    average_repeats = counts_column.mean()
    #print("Average number of repeats: ", average_repeats)
    perc_repeat = len(trials_lick_at_bg)/trial_num
    exclude_values = trials_lick_at_bg['session_trial_num']
    # print(len(exclude_values))
    good_trials = df[~df['session_trial_num'].isin(exclude_values)]
    good_trials_num = len(good_trials.drop_duplicates(subset=['session_trial_num']))
    trials_good = good_trials.drop_duplicates(subset=['session_trial_num'])
    # print("number of good trials: ", len(good_trials))
    # good_trials = blk_cp[exclude_condition].drop_duplicates(subset=['session_trial_num'])

    # print(len(good_trials))
    # print(len(trials_lick_at_bg))
    return perc_repeat, trials_lick_at_bg, trials_good, good_trials_num, average_repeats

def getLickBoutsIntoBg(df, interval):
    licks = df.loc[df['key'] == 'lick']
    licks['time_diff'] = licks['session_time'].diff()
    bout_start = licks['time_diff'] > interval
    licks['bout_number'] = bout_start.cumsum()
    bout_indices = df.groupby('bout_number').agg({'session_time': ['first', 'last'], 'state': 'first'}).reset_index()
    bout_indices.columns = bout_indices.columns.droplevel()
    bout_indices.rename(columns={'first': 'start_time', 'last': 'end_time'}, inplace=True)

    # Filter bouts that meet the lick time difference criterion and start in "in_consumption"
    lick_bouts = bout_indices[bout_indices['state'] == 'in_consumption']

    # Count the number of instances where "in_background" state is included in each lick bout during "in_consumption"
    count_instances = 0
    for bout_number, start_time, end_time in zip(lick_bouts['bout_number'], lick_bouts['start_time'],
                                                 lick_bouts['end_time']):
        background_length = \
        df[(df['state'] == 'in_background') & (df['session_time'] >= start_time) & (df['session_time'] <= end_time)][
            'session_time'].sum()
        if background_length > 0:
            count_instances += 1


    # Display the lick bouts and count of instances
    print("Lick Bouts:")
    print(lick_bouts)
    print("\nNumber of Instances with 'in_background' state included in 'in_consumption' bout:", count_instances)
    return lick_bouts, count_instances


class Session:
    def __init__(self, animal, file_path, has_block, task_params):
        # within session lists

        self.animal = animal
        self.file_path = file_path
        self.has_block = has_block
        self.task_params = task_params
        self.lickbout = 0.3
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

        # only for sessions with blocks
        self.blk_start_var = []  # variance of waiting time for the last 10 trials before block switch
        self.blk_end_var = []  # variance of waiting time for the first 10 trials after block switch
        self.blk_window = 5
        self.blk_start_slope = []
        self.blk_end_slope = []

        self.session_reward_rate = []
        self.session_holding_times = []

        self.bg_repeats_s = []
        self.bg_repeats_l = []

        self.animal.holding_diff = []

        self.animal_all_holding_s_good = []
        self.holding_s_good = []
        self.opt_diff_s_good = []
        self.perc_rewarded_s_good = []
        self.prob_at_lick_s_good = []

        self.non_reflexive_s = []
        self.non_reflexive_l = []

        self.animal_all_holding_l_good = []
        self.holding_l_good = []
        self.opt_diff_l_good = []
        self.perc_rewarded_l_good = []
        self.prob_at_lick_l_good = []
        self.reflex_length = 0.5
        # number of bg repeats
        self.bg_repeats_s_licks = []
        self.bg_repeats_l_licks = []


    def processSelectedTrials(self, trial_num, df, curr_opt_wait, timescape_type):
        # this function will process selected trials dataframe
        # within block or within a no-block session
        # can be all trials or just the good trials.

        # outputs the miss trials percentage, miss trials,
        # mean prob at lick,
        blk_missed_perc, miss_trials = getMissedTrials(df, trial_num)
        print(self.file_path)
        mean_prob_at_lick = getProbRewarded(df)
        perc_rewarded_perf = getRewardedPerc(df, trial_num)
        curr_licks_during_wait = df.loc[
            (df['key'] == 'lick') & (df['state'] == 'in_wait')]
        licks = curr_licks_during_wait.curr_wait_time
        lick_mean, mean_lick_diff = self.getLickStats(licks, curr_opt_wait)
        consumption_lengths, background_lengths, lick_counts_consumption, lick_counts_background, \
        mean_consumption_length, mean_consumption_licks, mean_background_length, mean_background_licks, \
        perc_bout_into_background = getConsumptionStats(df, self.lickbout, 1)

        return blk_missed_perc, miss_trials, mean_prob_at_lick, licks, lick_mean, \
               mean_lick_diff, perc_rewarded_perf,  mean_consumption_length, mean_consumption_licks, \
               mean_background_length, mean_background_licks, perc_bout_into_background

    def parseSessionStats(self):
        session_data = pd.read_csv(self.file_path, skiprows=3)
        session_data["session_time"] = session_data["session_time"] - session_data["session_time"][0]

        # find block switches
        session_data['blk_change'] = session_data.loc[:, 'block_num'].diff()
        if self.has_block:
            session_data['total_volume_received'] = session_data['total_reward']*5
        else:
            session_data['total_volume_received'] = session_data['total_reward']*10
        self.session_reward_rate = session_data['total_volume_received']/(session_data['session_time'])
        # print(session_data['total_volume_received'].iloc[-1])
        # print(session_data['session_time'].iloc[-1])
        # print(f'mean of session reward rate is {self.session_reward_rate.mean()}')

        if self.has_block:
            print("processing session with blocks")
            session_blks = session_data[session_data['blk_change'] != 0]
            for k in range(1, session_blks.shape[0] - 1):
                #             print(f'{i}+{j}+{k}')
                curr_blk = session_data[session_data.block_num == k]

                # find number of trials in this block
                blk_trial_num = max(curr_blk.session_trial_num) - min(curr_blk.session_trial_num) + 1

                # perc_rewarded_perf = getRewardedPerc(curr_blk, blk_trial_num)
                curr_blk_mean_reward_time = int(curr_blk.mean_reward_time.iloc[0])
                curr_opt_wait, timescape_type = self.getTimescapeType(curr_blk_mean_reward_time)
                blk_cp = curr_blk.copy()
                blk_cp['next_state'] = blk_cp['state'].shift(-1)

                blk_bg_repeat_perc, trials_lick_at_bg, trials_good, good_trials_num, average_repeats = \
                    getBackgroundLicks(blk_cp, blk_trial_num)
                # print(blk_trial_num == good_trials_num + len(trials_lick_at_bg))

                blk_missed_perc, miss_trials, mean_prob_at_lick, licks, lick_mean, \
                mean_lick_diff, perc_rewarded_perf, mean_consumption_length, mean_consumption_licks, \
                mean_background_length, mean_background_licks,perc_bout_into_background = \
                                                                    self.processSelectedTrials(blk_trial_num, blk_cp,
                                                                                curr_opt_wait, timescape_type)

                blk_missed_perc_good, miss_trials_good, mean_prob_at_lick_good, licks_good, lick_mean_good, \
                mean_lick_diff_good, perc_rewarded_perf_good, mean_consumption_length_good, mean_consumption_licks_good, \
                mean_background_length_good, mean_background_licks_good, perc_bout_into_background_good =\
                                                                    self.processSelectedTrials(good_trials_num,
                                                                         trials_good, curr_opt_wait, timescape_type)

                self.block_type.append(timescape_type)
                self.perc_rewarded.append(perc_rewarded_perf)

                self.session_holding_times.append(licks)
                self.animal.mean_consumption_length.append(mean_consumption_length)
                self.animal.mean_consumption_licks.append(mean_consumption_licks)
                if len(licks) >= 10 and blk_trial_num >= 10:
                    self.valid_block_type.append(self.block_type[k-1])
                    self.blkVarianceAnalysis(licks)

                self.opt_diff.append(mean_lick_diff)

                if self.block_type[k - 1] == 's':
                    self.animal.all_holding_s.extend(licks)
                    self.animal.all_holding_s_index.append(len(self.animal.all_holding_s))
                    self.holding_s.append(lick_mean) # block mean
                    self.opt_diff_s.append(mean_lick_diff)
                    self.perc_rewarded_s.append(perc_rewarded_perf)
                    self.prob_at_lick_s.append(mean_prob_at_lick)
                    self.bg_repeats_s.append(blk_bg_repeat_perc)
                    print(f"bg repeat perc {self.bg_repeats_s}")

                    self.animal_all_holding_s_good.extend(licks_good)
                    self.holding_s_good.append(lick_mean_good)
                    self.opt_diff_s_good.append(mean_lick_diff_good)
                    self.perc_rewarded_s_good.append(perc_rewarded_perf_good)
                    self.prob_at_lick_s_good.append(mean_prob_at_lick_good)
                    print(f"s good holding number {len(self.holding_s_good)}")
                    if not math.isnan(blk_missed_perc):
                        self.animal.miss_perc_s.append(blk_missed_perc)
                    else:
                        self.animal.miss_perc_s.append(np.nan)
                else:
                    self.animal.all_holding_l.extend(licks)
                    self.animal.all_holding_l_index.append(len(self.animal.all_holding_l))
                    self.holding_l.append(lick_mean)
                    self.opt_diff_l.append(mean_lick_diff)
                    self.perc_rewarded_l.append(perc_rewarded_perf)
                    self.prob_at_lick_l.append(mean_prob_at_lick)
                    self.bg_repeats_l.append(blk_bg_repeat_perc)

                    self.animal_all_holding_l_good.extend(licks_good)
                    self.holding_l_good.append(lick_mean_good)
                    self.opt_diff_l_good.append(mean_lick_diff_good)
                    self.perc_rewarded_l_good.append(perc_rewarded_perf_good)
                    self.prob_at_lick_l_good.append(mean_prob_at_lick_good)
                    if not math.isnan(blk_missed_perc):
                        self.animal.miss_perc_l.append(blk_missed_perc)
                    else:
                        self.animal.miss_perc_l.append(np.nan)

        else:
            session_data_cp = session_data.copy()
            session_data_cp['next_state'] = session_data_cp['state'].shift(-1)

            session_trial_num = max(session_data_cp.session_trial_num) - min(session_data_cp.session_trial_num) + 1
            # session_missed_perc, miss_trials = getMissedTrials(session_data_cp, session_trial_num)

            session_repeat_perc, trials_lick_at_bg, trials_good, good_trials_num, average_repeats = \
                getBackgroundLicks(session_data_cp, session_trial_num)

            session_mean_reward_time = int(session_data.mean_reward_time.iloc[0])
            curr_opt_wait, timescape_type = self.getTimescapeType(session_mean_reward_time)

            session_missed_perc, miss_trials, mean_prob_at_lick, licks, lick_mean, \
            mean_lick_diff, perc_rewarded_perf, mean_consumption_length, mean_consumption_licks, mean_background_length, \
            mean_background_licks, perc_bout_into_background = self.processSelectedTrials( session_trial_num, session_data_cp,
                                                                            curr_opt_wait, timescape_type)

            session_missed_perc_good, miss_trials_good, mean_prob_at_lick_good, licks_good, lick_mean_good, \
            mean_lick_diff_good, perc_rewarded_perf_good, mean_consumption_length_good, mean_consumption_licks_good, \
            mean_background_length_good, mean_background_licks_good, perc_bout_into_background_good = \
                                                        self.processSelectedTrials(good_trials_num, trials_good, curr_opt_wait, timescape_type)
            self.animal.mean_consumption_length.append(mean_consumption_length)
            self.animal.mean_consumption_licks.append(mean_consumption_licks)

           # print(len(licks_good))
            if timescape_type == 's':
                self.animal.all_holding_s.extend(licks)
                self.animal.all_holding_s_list.append(licks)
                self.animal.all_holding_s_index.append(len(self.animal.all_holding_s))
                self.animal.mean_background_length_from_consumption_s.append(mean_background_length)
                self.animal.mean_background_lick_from_consumption_s.append(mean_background_licks)
                self.animal.perc_bout_into_background_s.append(perc_bout_into_background_good)
                self.animal.reflex_lick_perc_s.append(len(licks[licks <= self.reflex_length])/len(licks))
                self.non_reflexive_s.append((licks[licks > self.reflex_length]))
               # print(f'std of current licking is {np.std(licks)}')
                self.animal.holding_s_std.append(np.std(licks))
                self.animal.holding_s_sem.append(sem(licks))
                self.animal.holding_s_q25.append(licks.quantile(0.25))
                self.animal.holding_s_q75.append(licks.quantile(0.75))

                self.holding_s.append(lick_mean)
                self.opt_diff_s.append(mean_lick_diff)
                self.perc_rewarded_s.append(perc_rewarded_perf)
                self.prob_at_lick_s.append(mean_prob_at_lick)
                self.bg_repeats_s.append(session_repeat_perc)
                self.bg_repeats_s_licks.append(average_repeats)

                self.animal_all_holding_s_good.extend(licks_good)
                self.holding_s_good.append(lick_mean_good)
                self.opt_diff_s_good.append(mean_lick_diff_good)
                self.perc_rewarded_s_good.append(perc_rewarded_perf_good)
                self.prob_at_lick_s_good.append(mean_prob_at_lick_good)
                # print(f"s good holding number {len(self.holding_s_good)}")
                if not math.isnan(session_missed_perc):
                    self.animal.miss_perc_s.append(session_missed_perc)
                else:
                    self.animal.miss_perc_s.append(np.nan)
            else:
                self.animal.all_holding_l.extend(licks)
                self.animal.all_holding_l_list.append(licks)
                self.animal.all_holding_l_index.append(len(self.animal.all_holding_l))
                self.animal.mean_background_length_from_consumption_l.append(mean_background_length)
                self.animal.mean_background_lick_from_consumption_l.append(mean_background_licks)
                self.animal.perc_bout_into_background_l.append(perc_bout_into_background_good)
                self.animal.reflex_lick_perc_l.append(len(licks[licks <= self.reflex_length]) / len(licks))
                self.non_reflexive_l.append(licks[licks > self.reflex_length])
                # print(self.non_reflexive_l)
                self.animal.holding_l_std.append(np.std(licks))
                self.animal.holding_l_sem.append(sem(licks))
                self.animal.holding_l_q25.append(licks.quantile(0.25))
                self.animal.holding_l_q75.append(licks.quantile(0.75))
                self.holding_l.append(lick_mean)
                self.opt_diff_l.append(mean_lick_diff)
                self.perc_rewarded_l.append(perc_rewarded_perf)
                self.prob_at_lick_l.append(mean_prob_at_lick)
                self.bg_repeats_l.append(session_repeat_perc)
                self.bg_repeats_l_licks.append(average_repeats)
                self.animal_all_holding_l_good.extend(licks_good)
                self.holding_l_good.append(lick_mean_good)
                self.opt_diff_l_good.append(mean_lick_diff_good)
                self.perc_rewarded_l_good.append(perc_rewarded_perf_good)
                self.prob_at_lick_l_good.append(mean_prob_at_lick_good)

                if not math.isnan(session_missed_perc):
                    self.animal.miss_perc_l.append(session_missed_perc)
                else:
                    self.animal.miss_perc_l.append(np.nan)
    def getTimescapeType(self, mean_reward_time):
        if mean_reward_time == 3:
            opt_wait = self.optimal_wait[1]
            timescape_type = 'l'
        elif mean_reward_time == 1:
            opt_wait = self.optimal_wait[0]
            timescape_type = 's'
        else:
            raise Warning("wrong mean reward time")
        return opt_wait, timescape_type

    def blkVarianceAnalysis(self, licks):
        blk_start_trials = licks[:self.blk_window]
        blk_end_trials = licks[-self.blk_window:]

        self.blk_start_var.append(statistics.variance(blk_start_trials))
        self.blk_end_var.append(statistics.variance(blk_end_trials))

        self.blk_start_slope.append(np.polyfit(range(0, len(blk_start_trials)), blk_start_trials, 1))
        self.blk_end_slope.append(np.polyfit(range(0, len(blk_end_trials)), blk_end_trials, 1))

    def getLickStats(self, licks, opt):
        lick_mean = licks.mean()
        lick_diff = licks - opt
        mean_lick_diff = lick_diff.mean()
        return lick_mean, mean_lick_diff

    def updateSessionStats(self):
        if len(self.holding_s) > 0:
            non_nan_values = [x for x in self.holding_s if not math.isnan(x)]
            self.animal.holding_s.extend(non_nan_values)
            self.animal.all_holding_s_by_session.append(non_nan_values)
            self.animal.holding_s_mean.append(mean(self.holding_s))
            self.animal.holding_s_median.append(median(self.holding_s))
            if len(self.holding_s_good) > 0:
                self.animal.all_holding_s_good.extend(x for x in self.holding_s_good if not math.isnan(x))
                self.animal.holding_s_mean_good.append(mean(self.holding_s_good))
                # print(f"mean good trial licking {self.animal.holding_s_mean_good}")
        else:
            self.animal.holding_s_mean.append(np.nan)
        if len(self.holding_l) > 0:
            non_nan_values = [x for x in self.holding_l if not math.isnan(x)]
            # Calculate the standard deviation of non-NaN values
            self.animal.holding_l.extend(non_nan_values)
            self.animal.all_holding_l_by_session.append(non_nan_values)
            self.animal.holding_l_mean.append(mean(self.holding_l))
            self.animal.holding_l_median.append(median(self.holding_l))
            if len(self.holding_l_good) > 0:
                self.animal.all_holding_l_good.extend(x for x in self.holding_l_good if not math.isnan(x))
                self.animal.holding_l_mean_good.append(mean(self.holding_l_good))
        else:
            self.animal.holding_l_mean.append(np.nan)

        if not np.isnan(self.animal.holding_l_mean).any() and not np.isnan(self.animal.holding_s_mean).any():
            self.animal.holding_diff.append(self.animal.holding_s_mean[-1] - self.animal.holding_l_mean[-1])

        self.animal.bg_restart_s_all.extend(self.bg_repeats_s)
        self.animal.bg_restart_l_all.extend(self.bg_repeats_l)

        non_reflexive_s_mean = np.mean(self.non_reflexive_s) if len(self.non_reflexive_s) > 0 else np.nan
        non_reflexive_l_mean = np.mean(self.non_reflexive_l) if len(self.non_reflexive_l) > 0 else np.nan
        non_reflexive_s_std = np.std(self.non_reflexive_s) if len(self.non_reflexive_s) > 0 else np.nan
        non_reflexive_l_std = np.std(self.non_reflexive_l) if len(self.non_reflexive_l) > 0 else np.nan
        self.animal.non_reflexive_s_mean.append(non_reflexive_s_mean)
        self.animal.non_reflexive_s_std.append(non_reflexive_s_std)
        self.animal.non_reflexive_l_mean.append(non_reflexive_l_mean)
        self.animal.non_reflexive_l_std.append(non_reflexive_l_std)

        self.animal.bg_restart_s.append(mean(self.bg_repeats_s) if len(self.bg_repeats_s) > 0 else np.nan)
        self.animal.bg_restart_l.append(mean(self.bg_repeats_l) if len(self.bg_repeats_l) > 0 else np.nan)
        self.animal.bg_restart_licks_s.append(mean(self.bg_repeats_s_licks) if len(self.bg_repeats_s_licks) >0 else np.nan)
        self.animal.bg_restart_licks_l.append(mean(self.bg_repeats_l_licks) if len(self.bg_repeats_l_licks) >0 else np.nan)


        self.animal.opt_diff_s.append(mean(self.opt_diff_s) if len(self.opt_diff_s) > 0 else np.nan)
        self.animal.opt_diff_s_good.append(mean(self.opt_diff_s_good) if len(self.opt_diff_s) > 0 else np.nan)
        self.animal.opt_diff_l.append(mean(self.opt_diff_l) if len(self.opt_diff_l) > 0 else np.nan)
        self.animal.opt_diff_l_good.append(mean(self.opt_diff_l_good) if len(self.opt_diff_l) > 0 else np.nan)

        self.animal.perc_rewarded_s.append(mean(self.perc_rewarded_s) if len(self.perc_rewarded_s) > 0 else np.nan)
        self.animal.perc_rewarded_s_good.append(mean(self.perc_rewarded_s_good) if len(self.perc_rewarded_s_good)>0 else np.nan)
        self.animal.perc_rewarded_l.append(mean(self.perc_rewarded_l) if len(self.perc_rewarded_l) > 0 else np.nan)
        self.animal.perc_rewarded_l_good.append(mean(self.perc_rewarded_l_good) if len(self.perc_rewarded_l_good)>0 else np.nan)

        self.animal.lick_prob_s.append(mean(self.prob_at_lick_s) if len(self.prob_at_lick_s) > 0 else np.nan)
        self.animal.prob_at_lick_s_good.append(mean(self.prob_at_lick_s_good)if len(self.prob_at_lick_s_good) > 0 else np.nan)
        self.animal.lick_prob_l.append(mean(self.prob_at_lick_l) if len(self.prob_at_lick_l) > 0 else np.nan)
        self.animal.prob_at_lick_l_good.append(mean(self.prob_at_lick_l_good) if len(self.prob_at_lick_l_good) > 0 else np.nan)

        # only for tasks with blocks
        if self.has_block:
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




