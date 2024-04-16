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
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm

import utils


def generate_binary_reward_list(total_trials, rewarded_trials):
    binary_rewards = [0] * total_trials  # Initialize a list of zeros for all trials
    for trial_number in rewarded_trials:
        if trial_number <= total_trials:
            binary_rewards[trial_number - 1] = 1  # Mark rewarded trials as 1
    # print(f'rewarded trials are {rewarded_trials}')
    # print(f'binary values are {binary_rewards}')
    return binary_rewards

def getRewardedPerc(df, trial_num):
    # print(df)
    rewarded_trial = df.loc[(df['key'] == 'reward') & (df['value'] == 1) & (df['state'].shift(1) == 'in_wait')
                            & (df['key'].shift(1) == 'lick') & (df['value'].shift(1) == 1)]

    trials_rewarded = rewarded_trial['session_trial_num'].tolist()

    loc_trials_rewarded = generate_binary_reward_list(trial_num, trials_rewarded)
    perc_rewarded_perf = len(rewarded_trial) / trial_num

    return perc_rewarded_perf, trials_rewarded, loc_trials_rewarded

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
            next_background = df[(df['state'] == 'in_background') & (df['session_time'] <= end_time) &
                                 (df['session_time'] > start_time)]
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

        true_count = sum(background_lengths)
        perc_bout_into_background = true_count/len(consumption_lengths)

    mean_consumption_length = mean(consumption_lengths) if len(consumption_lengths) > 0 else np.nan
    mean_consumption_licks = mean(lick_counts_consumption) if len(lick_counts_consumption) > 0 else np.nan
    mean_next_background_length = mean(background_lengths) if len(background_lengths) > 0 else np.nan
    mean_background_licks = mean(lick_counts_background) if len(lick_counts_background) > 0 else np.nan

    return consumption_lengths, background_lengths, lick_counts_consumption, lick_counts_background, \
           mean_consumption_length, mean_consumption_licks, mean_next_background_length, mean_background_licks,\
           perc_bout_into_background

def getMissedTrials(df, trial_num):
    miss_trials = df.loc[(df['state'] == 'in_wait') &
                         (df['next_state'] == 'trial_ends') &
                         (df['key'] == 'wait')]
    trials_missed = miss_trials['session_trial_num'].tolist()
    # print(f'trials missed are {trials_missed}')
    loc_trials_missed = generate_binary_reward_list(trial_num, trials_missed)
    miss_trials_num = len(miss_trials)
    missed_perc = miss_trials_num/trial_num
    if sum(loc_trials_missed) ==0:
        print('no missed trials!')
    return missed_perc, miss_trials, trials_missed, loc_trials_missed

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

    # Initialize variables to store background state durations
    background_durations = []
    # Initialize variables to track the start time and duration of the current background chunk
    current_background_start_time = None
    current_background_duration = 0
    # Iterate through the DataFrame
    for index, row in df.iterrows():
        if row['state'] == 'in_background':
            # If this is the start of a new background chunk, record the start time
            if current_background_start_time is None:
                current_background_start_time = row['session_time']
        elif row['state'] == 'in_wait':
            # If this is the end of a background chunk (followed by in_wait), calculate the duration
            if current_background_start_time is not None:
                current_background_duration = row['session_time'] - current_background_start_time
                background_durations.append(current_background_duration)
                # Reset the tracking variables for the next chunk
                current_background_start_time = None
                current_background_duration = 0

    # Check if any background state chunks were found
    if background_durations:
        # Calculate the mean duration of background state chunks
        mean_background_length = sum(background_durations) / len(background_durations)
        # print("Background state durations (in seconds):")
        # for duration in background_durations:
        #     print(duration)
        print(f"Mean background state duration: {mean_background_length} seconds")
    else:
        print("No background state chunks found.")

    return perc_repeat, trials_lick_at_bg, trials_good, good_trials_num, average_repeats, mean_background_length

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

def get_prev_lick_impact(lick_time, num_prev_licks=3):
    lick_df = pd.DataFrame(lick_time, columns=['lick_time'])

    # Dynamically creating individual lagged features
    for i in range(1, num_prev_licks + 1):
        lick_df[f'prev_lick_{i}'] = lick_df['lick_time'].shift(i)

    # Dropping rows with NaN values
    lick_df.dropna(inplace=True)

    # Preparing the dataset for linear regression
    feature_cols = [f'prev_lick_{i}' for i in range(1, num_prev_licks + 1)]
    X = lick_df[feature_cols]
    y = lick_df['lick_time']

    # Building and training the linear regression model
    model = LinearRegression()
    model.fit(X, y)

    # Evaluating the model on the entire dataset
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)

    # Returning model coefficients and MSE
    return model.coef_, mse

def get_regression_results(lick_time, num_prev_licks=3):
    lick_df = pd.DataFrame(lick_time, columns=['lick_time'])

    # Creating lagged features
    for i in range(1, num_prev_licks + 1):
        lick_df[f'prev_lick_{i}'] = lick_df['lick_time'].shift(i)

    lick_df.dropna(inplace=True)

    # Preparing the dataset for linear regression
    X = lick_df[[f'prev_lick_{i}' for i in range(1, num_prev_licks + 1)]]
    y = lick_df['lick_time']

    X = sm.add_constant(X)
    model = sm.OLS(y, X.astype(float)).fit()

    # Extracting coefficients, p-values, and confidence intervals
    coefficients = model.params
    p_values = model.pvalues
    confidence_intervals = model.conf_int()

    return coefficients, p_values, confidence_intervals


class Session:
    def __init__(self, animal, file_path, has_block, task_params):
        # within session lists

        self.animal = animal
        self.file_path = file_path
        self.has_block = has_block
        self.task_params = task_params
        self.lickbout = 0.3

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

        self.q25_s = []
        self.q75_s = []
        self.q25_l = []
        self.q75_l = []

        # only for sessions with blocks
        self.blk_start_var = []  # variance of waiting time for the last 10 trials before block switch
        self.blk_end_var = []  # variance of waiting time for the first 10 trials after block switch
        self.blk_window = 5
        self.blk_start_slope = []
        self.blk_end_slope = []

        self.session_reward_rate = []
        self.mean_session_reward_rate = []
        self.session_holding_times = []
        self.reflex_lick_perc_l = []
        self.reflex_lick_perc_s = []

        self.bg_repeats_s = []
        self.bg_repeats_l = []

        # self.animal.holding_diff = []

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
        self.loc_trials_rewarded_s = []
        self.loc_trials_rewarded_l = []
        self.loc_trials_missed_s = []
        self.loc_trials_missed_l = []
        self.loc_licks_rewarded_s = []
        self.loc_licks_rewarded_l = []
        self.prob_at_lick_l_good = []
        self.reflex_length = 0.5
        # number of bg repeats
        self.bg_repeats_s_licks = []
        self.bg_repeats_l_licks = []

        self.miss_perc_s = []
        self.miss_perc_l = []

        self.mean_background_length_s = []
        self.mean_background_length_l = []

        self.mean_background_length_from_consumption_s = []
        self.mean_background_lick_from_consumption_s = []
        self.mean_background_length_from_consumption_l = []
        self.mean_background_lick_from_consumption_l = []

    def process_selected_trials(self, trial_num, df, curr_opt_wait):
        # this function will process selected trials dataframe
        # within block or within a no-block session
        # can be all trials or just the good trials.

        # outputs the miss trials percentage, miss trials,
        # mean prob at lick,
        blk_missed_perc, miss_trials, trials_missed, loc_trials_missed = getMissedTrials(df, trial_num)
        print(self.file_path)
        mean_prob_at_lick = getProbRewarded(df)
        perc_rewarded_perf, trials_rewarded, loc_trials_rewarded = getRewardedPerc(df, trial_num)
        all_licks = df.loc[(df['key'] == 'lick') & (df['value'] == 1)]
        # all_lick_time = all_licks.curr_wait_time

        curr_licks_during_wait = df.loc[
            (df['key'] == 'lick') & (df['state'] == 'in_wait') & (df['value'] == 1)]
        # print(len(curr_licks_during_wait))
        licks = curr_licks_during_wait.curr_wait_time
        lick_trials = curr_licks_during_wait.session_trial_num.tolist()

        lick_mean, mean_lick_diff = self.getLickStats(licks, curr_opt_wait)
        consumption_lengths, background_lengths, lick_counts_consumption, lick_counts_background, \
        mean_consumption_length, mean_consumption_licks, mean_next_background_length, mean_background_licks, \
        perc_bout_into_background = getConsumptionStats(df, self.lickbout, 1)

        print(f'number of trials experienced {len(loc_trials_rewarded)}')
        # print(len(loc_trials_rewarded))
        print(f'number of trials missed{sum(loc_trials_missed)}')
        completed_trials = [trial for trial in range(0, self.session_trial_num) if trial not in trials_missed]

        # print(f'number of completed {len(completed_trials)}')
        # print(lick_trials)
        # print(len(lick_trials))
        lick_not_completed = set(completed_trials) - set(lick_trials)
        filtered_loc_trials_rewarded = \
            [value for index, value in enumerate(loc_trials_rewarded) if index not in lick_not_completed]
        filtered_loc_trials_missed = \
            [value for index, value in enumerate(loc_trials_missed) if index not in lick_not_completed]
        loc_licks_rewarded = [filtered_loc_trials_rewarded[i] for i in range(len(filtered_loc_trials_rewarded))
                              if filtered_loc_trials_missed[i] != 1]
        # if sum(filtered_loc_trials_missed) != len(trials_missed):
        #     print('somehow missed trials get thrown out')
        # # filtered_loc_licks_rewarded = \
        # #     [value for index, value in enumerate(loc_licks_rewarded) if index not in lick_not_completed]
        #
        # if len(loc_licks_rewarded) != len(licks):
        #     print("The lengths of filtered_loc_licks_rewarded and licks are not equal.")
        #     # You can also print the lengths for further investigation
        #     print("Length of filtered_loc_licks_rewarded:", len(loc_licks_rewarded))
        #     print("Length of licks:", len(licks))
        # print(f'this are the trials where its not counted as a trial?! {not_in_completed}')
        # # print(loc_licks_rewarded)
        print(f'all trials add up {len(licks) == len(loc_licks_rewarded)}')
        return blk_missed_perc, miss_trials, filtered_loc_trials_missed, mean_prob_at_lick, licks, lick_mean, \
               mean_lick_diff, perc_rewarded_perf,  filtered_loc_trials_rewarded, loc_licks_rewarded,\
               mean_consumption_length, mean_consumption_licks, \
               mean_next_background_length, mean_background_licks, perc_bout_into_background, \
               #coefficients, p_values, confidence_intervals

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
        self.session_trial_num = session_data["session_trial_num"].max() + 1
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

                blk_bg_repeat_perc, trials_lick_at_bg, trials_good, good_trials_num, average_repeats, mean_background_length = \
                    getBackgroundLicks(blk_cp, blk_trial_num)
                # print(blk_trial_num == good_trials_num + len(trials_lick_at_bg))

                blk_missed_perc, miss_trials, loc_trials_missed, mean_prob_at_lick, licks, lick_mean, \
                mean_lick_diff, perc_rewarded_perf, loc_trials_rewarded, loc_licks_rewarded,\
                mean_consumption_length, mean_consumption_licks, \
                mean_next_background_length, mean_background_licks,perc_bout_into_background, \
                coefficients, p_values, confidence_intervals = \
                                                                    self.process_selected_trials(blk_trial_num, blk_cp,
                                                                                curr_opt_wait, timescape_type)

                blk_missed_perc_good, miss_trials_good, loc_trials_missed_good,\
                mean_prob_at_lick_good, licks_good, lick_mean_good, \
                mean_lick_diff_good, perc_rewarded_perf_good, loc_trials_rewarded_good,loc_licks_rewarded_good,\
                mean_consumption_length_good, mean_consumption_licks_good, \
                mean_next_background_length_good, mean_background_licks_good, perc_bout_into_background_good =\
                                                                    self.process_selected_trials(good_trials_num,
                                                                         trials_good, curr_opt_wait, timescape_type)

                self.block_type.append(timescape_type)
                self.perc_rewarded.append(perc_rewarded_perf)

                self.session_holding_times.append(licks)
                self.animal.mean_consumption_length.append(mean_consumption_length)
                self.animal.mean_consumption_licks.append(mean_consumption_licks)
                self.animal.mean_session_reward_rate.append(self.session_reward_rate.mean())
                if len(licks) >= 10 and blk_trial_num >= 10:
                    self.valid_block_type.append(self.block_type[k-1])
                    self.blkVarianceAnalysis(licks)

                self.opt_diff.append(mean_lick_diff)

                if self.block_type[k - 1] == 's':
                    self.animal.short_properties.all_holding.extend(licks)
                    self.animal.all_holding_s.extend(licks)
                    self.animal.short_properties.all_holding_index.append(len(self.animal.all_holding_s))
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
                        self.miss_perc_s.append(blk_missed_perc)
                    else:
                        self.miss_perc_s.append(np.nan)
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
                        self.miss_perc_l.append(blk_missed_perc)
                    else:
                        self.miss_perc_l.append(np.nan)

        else:
            print(f'processing session {self.file_path} without blocks')

            session_data_cp = session_data.copy()
            session_data_cp['next_state'] = session_data_cp['state'].shift(-1)
            self.animal.session_param.append(session_data.mean_reward_time.iloc[0])
            # session_trial_num = max(session_data_cp.session_trial_num) + 1

            # print('=======================this is the session data feed in for {self.file_path}===============================')
            # print(session_data_cp)
            # print('=======================end of the session data feed in===============================')

            session_repeat_perc, trials_lick_at_bg, trials_good, good_trials_num, average_repeats, mean_background_length = \
                getBackgroundLicks(session_data_cp, self.session_trial_num)

            session_mean_reward_time = session_data.mean_reward_time.iloc[0]
            timescape_type = self.getTimescapeType(session_mean_reward_time)
            curr_opt_wait = utils.get_optimal_time(session_mean_reward_time, 0.9, 2)
            # print(session_mean_reward_time)
            # print(curr_opt_wait)
            if len(self.animal.optimal_wait) >0:
                if curr_opt_wait != self.animal.optimal_wait[-1]:
                    self.animal.default_session_num = len(self.animal.optimal_wait)
            self.animal.optimal_wait.append(curr_opt_wait)

            session_missed_perc, miss_trials, loc_trials_missed, \
            mean_prob_at_lick, licks, lick_mean, \
            mean_lick_diff, perc_rewarded_perf, loc_trials_rewarded, loc_licks_rewarded,\
            mean_consumption_length, mean_consumption_licks, mean_next_background_length, \
            mean_background_licks, perc_bout_into_background\
                = self.process_selected_trials(
                self.session_trial_num, session_data_cp, curr_opt_wait)

            session_missed_perc_good, miss_trials_good, loc_trials_missed_good, \
            mean_prob_at_lick_good, licks_good, lick_mean_good, \
            mean_lick_diff_good, perc_rewarded_perf_good, loc_trials_rewarded_good, loc_licks_rewarded_good,\
            mean_consumption_length_good, mean_consumption_licks_good, \
            mean_next_background_length_good, mean_background_licks_good, perc_bout_into_background_good = \
                                                        self.process_selected_trials(good_trials_num,
                                                                                     trials_good, curr_opt_wait)
            self.animal.mean_consumption_length.append(mean_consumption_length)
            self.animal.mean_consumption_licks.append(mean_consumption_licks)
            self.animal.mean_session_reward_rate.append(self.session_reward_rate.mean())

            if timescape_type == 's':
                self.animal.session_trial_num_s.append(self.animal.session_trial_num_s[-1] + self.session_trial_num)
                print(f'current session trial num index are {self.animal.session_trial_num_s}')
                self.animal.all_holding_s.extend(licks)
                self.animal.all_holding_s_list.append(licks)
                self.animal.all_holding_s_index.append(len(self.animal.all_holding_s))
                self.mean_background_length_s.append(mean_background_length)
                self.mean_background_length_from_consumption_s.append(mean_next_background_length)
                self.mean_background_lick_from_consumption_s.append(mean_background_licks)
                self.animal.perc_bout_into_background_s.append(perc_bout_into_background_good)
                self.reflex_lick_perc_s.append(len(licks[licks <= self.reflex_length])/len(licks))
                self.non_reflexive_s.append((licks[licks > self.reflex_length]))
               # print(f'std of current licking is {np.std(licks)}')
                self.animal.holding_s_std.append(np.std(licks))
                self.animal.holding_s_sem.append(sem(licks))
                self.q25_s.append(licks.quantile(0.25))
                self.q75_s.append(licks.quantile(0.75))

                self.holding_s.append(lick_mean)
                self.opt_diff_s.append(mean_lick_diff)
                self.perc_rewarded_s.append(perc_rewarded_perf)
                self.loc_trials_rewarded_s.extend(loc_trials_rewarded)
                self.loc_trials_missed_s.extend(loc_trials_missed)
                self.loc_licks_rewarded_s.extend(loc_licks_rewarded)
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
                    self.miss_perc_s.append(session_missed_perc)
                else:
                    self.miss_perc_s.append(np.nan)
            else:
                self.animal.session_trial_num_l.append(self.animal.session_trial_num_l[-1] + self.session_trial_num)
                print(f'current session trial num index are {self.animal.session_trial_num_l}')
                self.animal.all_holding_l.extend(licks)
                self.animal.all_holding_l_list.append(licks)
                self.animal.all_holding_l_index.append(len(self.animal.all_holding_l))

                if len(licks) != len(loc_licks_rewarded):
                    print(self.file_path)
                    print("===========================caution not matching!=======================")
                    print(len(licks))
                    print(len(loc_licks_rewarded))
                self.mean_background_length_l.append(mean_background_length)
                self.mean_background_length_from_consumption_l.append(mean_next_background_length)
                self.mean_background_lick_from_consumption_l.append(mean_background_licks)
                self.animal.perc_bout_into_background_l.append(perc_bout_into_background_good)
                self.reflex_lick_perc_l.append(len(licks[licks <= self.reflex_length]) / len(licks))
                self.non_reflexive_l.append(licks[licks > self.reflex_length])
                # print(self.non_reflexive_l)
                self.animal.holding_l_std.append(np.std(licks))
                self.animal.holding_l_sem.append(sem(licks))
                self.q25_l.append(licks.quantile(0.25))
                self.q75_l.append(licks.quantile(0.75))
                self.holding_l.append(lick_mean)
                self.opt_diff_l.append(mean_lick_diff)
                self.perc_rewarded_l.append(perc_rewarded_perf)
                self.loc_trials_rewarded_l.extend(loc_trials_rewarded)
                self.loc_trials_missed_l.extend(loc_trials_missed)
                self.loc_licks_rewarded_l.extend(loc_licks_rewarded)
                self.prob_at_lick_l.append(mean_prob_at_lick)
                self.bg_repeats_l.append(session_repeat_perc)
                self.bg_repeats_l_licks.append(average_repeats)
                self.animal_all_holding_l_good.extend(licks_good)
                self.holding_l_good.append(lick_mean_good)
                self.opt_diff_l_good.append(mean_lick_diff_good)
                self.perc_rewarded_l_good.append(perc_rewarded_perf_good)
                self.prob_at_lick_l_good.append(mean_prob_at_lick_good)

                if not math.isnan(session_missed_perc):
                    self.miss_perc_l.append(session_missed_perc)
                else:
                    self.miss_perc_l.append(np.nan)
    def getTimescapeType(self, mean_reward_time):
        if mean_reward_time >= 3:
            timescape_type = 'l'
        else:
            timescape_type = 's'

        return timescape_type

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
            self.animal.holding_s_median.append(np.nan)
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
            self.animal.holding_l_median.append(np.nan)
        #
        # if not np.isnan(self.animal.holding_l_mean).any() and not np.isnan(self.animal.holding_s_mean).any():
        #     self.animal.holding_diff.append(self.animal.holding_s_mean[-1] - self.animal.holding_l_mean[-1])

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
        self.animal.reflex_lick_perc_s.append(np.mean(self.reflex_lick_perc_s) if len(self.reflex_lick_perc_s)>0
                                              else np.nan)
        self.animal.reflex_lick_perc_l.append(np.mean(self.reflex_lick_perc_l) if len(self.reflex_lick_perc_l) > 0
                                              else np.nan)

        self.animal.holding_s_q25.append(np.mean(self.q25_s) if len(self.q25_s) > 0 else np.nan)
        self.animal.holding_l_q25.append(np.mean(self.q25_l) if len(self.q25_l) > 0 else np.nan)
        self.animal.holding_s_q75.append(np.mean(self.q75_s) if len(self.q75_s) > 0 else np.nan)
        self.animal.holding_l_q75.append(np.mean(self.q75_l) if len(self.q75_l) > 0 else np.nan)

        self.animal.mean_background_length_s.append(np.mean(self.mean_background_length_s)
                                                    if len(self.mean_background_length_s) > 0 else np.nan)
        self.animal.mean_background_length_l.append(np.mean(self.mean_background_length_l)
                                                    if len(self.mean_background_length_l) > 0 else np.nan)

        self.animal.mean_background_length_from_consumption_s.append(
            np.mean(self.mean_background_length_from_consumption_s)
            if len(self.mean_background_length_from_consumption_s) > 0
            else np.nan)

        self.animal.mean_background_length_from_consumption_l.append(
            np.mean(self.mean_background_length_from_consumption_l)
            if len(self.mean_background_length_from_consumption_l) > 0
            else np.nan)

        self.animal.mean_background_lick_from_consumption_l.append(
            np.mean(self.mean_background_lick_from_consumption_l)
            if len(self.mean_background_lick_from_consumption_l) > 0
            else np.nan)

        self.animal.mean_background_lick_from_consumption_s.append(
            np.mean(self.mean_background_lick_from_consumption_s)
            if len(self.mean_background_lick_from_consumption_s) > 0
            else np.nan)

        self.animal.bg_restart_s.append(mean(self.bg_repeats_s) if len(self.bg_repeats_s) > 0 else np.nan)
        self.animal.bg_restart_l.append(mean(self.bg_repeats_l) if len(self.bg_repeats_l) > 0 else np.nan)
        self.animal.bg_restart_licks_s.append(mean(self.bg_repeats_s_licks) if len(self.bg_repeats_s_licks) >0 else np.nan)
        self.animal.bg_restart_licks_l.append(mean(self.bg_repeats_l_licks) if len(self.bg_repeats_l_licks) >0 else np.nan)

        self.animal.miss_perc_s.append(mean(self.miss_perc_s) if len(self.miss_perc_s) > 0 else np.nan)
        self.animal.miss_perc_l.append(mean(self.miss_perc_l) if len(self.miss_perc_l) > 0 else np.nan)
        self.animal.loc_trials_missed_s.extend(self.loc_trials_missed_s)
        self.animal.loc_trials_missed_l.extend(self.loc_trials_missed_l)
        self.animal.loc_licks_rewarded_s.extend(self.loc_licks_rewarded_s)
        self.animal.loc_licks_rewarded_l.extend(self.loc_licks_rewarded_l)

        self.animal.opt_diff_s.append(mean(self.opt_diff_s) if len(self.opt_diff_s) > 0 else np.nan)
        self.animal.opt_diff_s_good.append(mean(self.opt_diff_s_good) if len(self.opt_diff_s) > 0 else np.nan)
        self.animal.opt_diff_l.append(mean(self.opt_diff_l) if len(self.opt_diff_l) > 0 else np.nan)
        self.animal.opt_diff_l_good.append(mean(self.opt_diff_l_good) if len(self.opt_diff_l) > 0 else np.nan)

        self.animal.perc_rewarded_s.append(mean(self.perc_rewarded_s) if len(self.perc_rewarded_s) > 0 else np.nan)
        self.animal.perc_rewarded_s_good.append(mean(self.perc_rewarded_s_good) if len(self.perc_rewarded_s_good)>0 else np.nan)
        self.animal.perc_rewarded_l.append(mean(self.perc_rewarded_l) if len(self.perc_rewarded_l) > 0 else np.nan)
        self.animal.perc_rewarded_l_good.append(mean(self.perc_rewarded_l_good) if len(self.perc_rewarded_l_good)>0 else np.nan)
        # print(f'loc rewarded is {self.loc_trials_rewarded_l}')
        # print(f'loc rewarded is {self.loc_trials_rewarded_s}')
        self.animal.loc_trials_rewarded_s.extend(self.loc_trials_rewarded_s)
        self.animal.loc_trials_rewarded_l.extend(self.loc_trials_rewarded_l)

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




