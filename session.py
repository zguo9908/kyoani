# single session data parsing tool
from animal import Animal


class Session:
    def __init__(self, animal, date, time, task_type):
        self.animal = Animal(animal)


    def parseSessionStats(self, curr_path):
        curr_session_path = curr_path + '\\' + sessions[j]
        os.chdir(curr_session_path)
        file_path = curr_session_path + '\\' + os.listdir()[0]
        #         print(file_path)
        session_data = pd.read_csv(file_path, skiprows=3)
        session_data["session_time"] = session_data["session_time"] - session_data["session_time"][0]

        # find block switches

        session_data['blk_change'] = session_data.loc[:, 'block_num'].diff()
        session_blks = session_data[session_data['blk_change'] != 0]

        # within session lists
        block_type = []
        holding = []  # avg time of holding licks
        holding_s = []  # avg time of holding licks during s blocks
        holding_l = []  # avg time of holding licks during l blocks
        opt_diff = []
        opt_diff_s = []  # avg diff of holding licks and optimal lick time of s block
        opt_diff_l = []  # avg diff of holding licks and optimal lick time of l block
        perc_rewarded = []
        perc_rewarded_s = []
        perc_rewarded_l = []
        prob_at_lick_s = []
        prob_at_lick_l = []

        for k in range(1, session_blks.shape[0] - 1):
            #             print(f'{i}+{j}+{k}')
            curr_blk = session_data[session_data.block_num == k]

            # find number of trials in this block
            blk_trial_num = max(curr_blk.session_trial_num) - min(curr_blk.session_trial_num) + 1
            blk_rewarded_trial = curr_blk.loc[(curr_blk['key'] == 'reward') & (curr_blk['value'] == 1)]
            blk_rewarded_trial_num = len(blk_rewarded_trial)
            perc_rewarded_perf = blk_rewarded_trial_num / blk_trial_num
            perc_rewarded.append(perc_rewarded_perf)

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
            #             print(len(miss_trials))
            print(blk_trial_num)
            blk_missed_perc = len(miss_trials) / blk_trial_num

            prob_rewarded_licks = blk_cp['curr_reward_prob']
            prob_rewarded_licks = prob_rewarded_licks.astype('float')
            mean_prob_at_lick = prob_rewarded_licks.mean()
            #             print(mean_prob_at_lick )

            if curr_blk_mean_reward_time == 3:
                curr_opt_wait = optimal_wait[1]
                block_type.append('l')
            elif curr_blk_mean_reward_time == 1:
                curr_opt_wait = optimal_wait[0]
                block_type.append('s')
            else:
                break

            if 'curr_wait_time' in curr_licks_during_wait:
                licks = curr_licks_during_wait.curr_wait_time
                #                 all_holding_diff = licks - curr_opt_wait
                #                 all_holding_perf = 1-abs(all_holding_diff)/curr_opt_wait

                # block mean analysis
                lick_mean = licks.mean()
                #                 holding.append(lick_mean)
                lick_diff = curr_licks_during_wait.curr_wait_time - curr_opt_wait
                mean_lick_diff = lick_diff.mean()
                opt_diff.append(mean_lick_diff)

                # perforamce for holding lick
                holding_perf = 1 - abs(mean_lick_diff) / curr_opt_wait

                if block_type[k - 1] == 's':
                    curr_animal.all_holding_s.extend(licks)
                    holding_s.append(lick_mean)
                    opt_diff_s.append(mean_lick_diff)
                    perc_rewarded_s.append(perc_rewarded_perf)
                    prob_at_lick_s.append(mean_prob_at_lick)
                    if not math.isnan(holding_perf):
                        curr_animal.holding_perf_s.append(holding_perf)
                    else:
                        curr_animal.holding_perf_s.append(np.nan)
                    if not math.isnan(blk_missed_perc):
                        curr_animal.blk_miss_perc_s.append(blk_missed_perc)
                    else:
                        curr_animal.blk_miss_perc_s.append(np.nan)
                else:
                    curr_animal.all_holding_l.extend(licks)
                    holding_l.append(lick_mean)
                    opt_diff_l.append(mean_lick_diff)
                    perc_rewarded_l.append(perc_rewarded_perf)
                    prob_at_lick_l.append(mean_prob_at_lick)
                    if not math.isnan(holding_perf):
                        curr_animal.holding_perf_l.append(holding_perf)
                    else:
                        curr_animal.holding_perf_l.append(np.nan)
                    if not math.isnan(blk_missed_perc):
                        curr_animal.blk_miss_perc_l.append(blk_missed_perc)
                    else:
                        curr_animal.blk_miss_perc_l.append(np.nan)

            else:
                break

