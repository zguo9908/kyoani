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

def get_rewarded_perc(df, trial_num):
    rewarded_trial = df.loc[(df['key'] == 'reward') & (df['value'] == 1)]
    perc_rewarded_perf = len(rewarded_trial) / trial_num
    return perc_rewarded_perf


def get_consumption_stats(all_df, bout_interval, min_lick_count_consumption):
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

        true_count = sum(background_lengths)
        perc_bout_into_background = true_count/len(consumption_lengths)

    mean_consumption_length = mean(consumption_lengths) if len(consumption_lengths) >0 else np.nan
    mean_consumption_licks = mean(lick_counts_consumption) if len(lick_counts_consumption)>0 else np.nan
    mean_next_background_length = mean(background_lengths) if len(background_lengths)>0 else np.nan
    mean_background_licks = mean(lick_counts_background) if len(lick_counts_background)>0 else np.nan

    return consumption_lengths, background_lengths, lick_counts_consumption, lick_counts_background, \
           mean_consumption_length, mean_consumption_licks, mean_next_background_length, mean_background_licks,\
           perc_bout_into_background


def get_missed_trials(df, trial_num):
    miss_trials = df.loc[(df['state'] == 'in_wait') &
                         (df['next_state'] == 'trial_ends') &
                         (df['key'] == 'wait')]
    miss_trials_num = len(miss_trials)
    missed_perc = miss_trials_num/trial_num
    return missed_perc, miss_trials


def get_prob_rewarded(df):
    df['curr_reward_prob'] = pd.to_numeric(df['curr_reward_prob'], errors='coerce')

    # Create a boolean column indicating which values could be converted to float
    prob_rewarded_licks = ~df['curr_reward_prob'].isna()
    prob_rewarded_licks = prob_rewarded_licks.astype('float')
    mean_prob_at_lick = prob_rewarded_licks.mean()
    return mean_prob_at_lick


def get_background_licks(df, trial_num):
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


def get_lick_bouts_into_bg(df, interval):
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

    print("Lick Bouts:")
    print(lick_bouts)
    print("\nNumber of Instances with 'in_background' state included in 'in_consumption' bout:", count_instances)
    return lick_bouts, count_instances

class OneBlockSession:
    def __init__(self, animal, file_path, has_block, task_params, optimal_wait):
        self.animal = animal
        self.file_path = file_path
        self.has_block = has_block
        self.task_params = task_params
        self.optimal_wait = optimal_wait
        self.lickbout = 0.3

        self.block_type = []
        self.valid_block_type = []
        self.holding = []  # avg time of holding licks for all blocks

        self.opt_diff = []

        self.perc_rewarded = []

        self.prob_at_lick = []
        self.q25 = []
        self.q75 = []

        self.session_reward_rate = []
        self.mean_session_reward_rate = []
        self.session_holding_times = []
        self.reflex_lick_perc = []

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