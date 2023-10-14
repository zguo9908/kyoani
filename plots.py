import os

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import matplotlib.animation as animation
import matplotlib.colors as colors


def set_axis_style(ax, labels):
    ax.set_xticks(np.arange(1, len(labels) + 1), labels=labels)
    ax.set_xlim(0.25, len(labels) + 0.75)


def rawPlots(mice, task_params, has_block, saving):
    #path = os.path.normpath(r'D:\figures\behplots') + "\\" + task_params
    if has_block:
        path = os.path.normpath(r'D:\figures\behplots') + "\\" + "blocks" + "\\" + task_params
    else:
        path = os.path.normpath(r'D:\figures\behplots') + "\\" + "no_blocks" + "\\" + task_params
    os.chdir(path)
    print(f'plotting and saving in {path}')
    for i in range(len(mice)):
        curr_animal_path = path + '\\' + mice[i].name
        print(curr_animal_path)
        os.chdir(curr_animal_path)
        # print(os.listdir())
        # file_path = curr_animal_path + '\\' + os.listdir()[0]

        fig, ax = plt.subplots(figsize=(10, 5))
        plt.plot(mice[i].holding_s_mean, 'bo')
        plt.plot(mice[i].holding_l_mean, 'ro')
        ax.set_title(f'{mice[i].name} holding times')
        ax.set_xlabel("session")
        ax.set_ylabel("holding time(s)")
        ax.legend(['short', 'long'])
        plt.savefig(f'{mice[i].name}_holding_times.svg')
        plt.close()

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.set_title(f'{mice[i].name} holding - optimal time')
        plt.plot(mice[i].opt_diff_s, 'b+')
        plt.plot(mice[i].opt_diff_l, 'r+')
        ax.legend(['short', 'long'])
        ax.set_xlabel("session")
        ax.set_ylabel("avg licking - optimal(s)")
        plt.savefig(f'{mice[i].name} avg licking - optimal.svg')
        plt.close()

        fig, ax = plt.subplots(figsize=(10, 5))
        plt.plot(mice[i].holding_s_mean_good, 'bo')
        plt.plot(mice[i].holding_l_mean_good, 'ro')
        ax.set_title(f'{mice[i].name} good trials holding times')
        ax.set_xlabel("session")
        ax.set_ylabel("holding time(s)")
        ax.legend(['short', 'long'])
        plt.savefig(f'{mice[i].name}_holding_times_good.svg')
        plt.close()

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.set_title(f'{mice[i].name} good trials holding - optimal time')
        plt.plot(mice[i].opt_diff_s_good, 'b+')
        plt.plot(mice[i].opt_diff_l_good, 'r+')
        ax.legend(['short', 'long'])
        ax.set_xlabel("session")
        ax.set_ylabel("avg licking - optimal(s)")
        plt.savefig(f'{mice[i].name} good trials avg licking - optimal.svg')
        plt.close()

        #     locs, labels = plt.xticks()
        #     plt.xticks(np.arange(4), block_type)
        fig, ax = plt.subplots(figsize=(10, 5))
        plt.plot(mice[i].perc_rewarded_s, 'b--')
        plt.plot(mice[i].perc_rewarded_l, 'r--')
        ax.set_title(f'{mice[i].name} percent trial rewarded')
        ax.set_xlabel("session")
        ax.set_ylabel("%")
        ax.legend(['short', 'long'])
        if saving:
            plt.savefig(f'{mice[i].name} perc trial rewarded.svg')
        plt.close()

        fig, ax = plt.subplots(figsize=(10, 5))
        plt.plot(mice[i].lick_prob_s, 'b-')
        plt.plot(mice[i].lick_prob_l, 'r-')
        ax.set_title(f'{mice[i].name} prob at licking')
        ax.set_xlabel("session")
        ax.set_ylabel("%")
        ax.legend(['short', 'long'])
        if saving:
            plt.savefig(f'{mice[i].name}_prob_at_licking.svg')
        plt.close()

        fig, ax = plt.subplots(figsize=(10, 5))
        plt.plot(mice[i].moving_average_s, 'bo')
        plt.plot(mice[i].moving_average_l, 'ro')
        ax.set_title(f'{mice[i].name} moving avg performance')
        ax.set_xlabel("trial")
        ax.set_ylabel("holing time(s)")
        ax.legend(['short', 'long'])
        plt.axis([1400, 2000, 0, 13])
        if saving:
            plt.savefig(f'{mice[i].name}_moving_avg_perf.svg')
        plt.close()

        fig, ax = plt.subplots(figsize=(10, 5))
        plt.plot(mice[i].holding_s_by_block, 'bo')
        plt.plot(mice[i].holding_l_by_block, 'ro')
        ax.set_title(f'{mice[i].name} holding times by block')
        ax.set_xlabel("block")
        ax.set_ylabel("holing time(s)")
        ax.legend(['short', 'long'])
        if saving:
            plt.savefig(f'{mice[i].name}_holding_times_by_block.svg')
        plt.close()

        fig, ax = plt.subplots(figsize=(10, 5))
        plt.plot(mice[i].miss_perc_s, 'b+')
        plt.plot(mice[i].miss_perc_l, 'r+')
        ax.set_title(f'{mice[i].name} missed trials percentage')
        ax.set_xlabel("block")
        ax.set_ylabel("%")
        ax.legend(['short', 'long'])
        if saving:
            plt.savefig(f'{mice[i].name}_missed_trials_perc.svg')
        plt.close()

        # fig, ax = plt.subplots(figsize=(10, 5))
        # plt.plot(mice[i].blk_start_var, 'b+')
        # plt.plot(mice[i].blk_end_var, 'r+')
        # ax.set_title(f'{mice[i].name} block start vs end waiting time variance')
        # ax.set_xlabel("block")
        # ax.set_ylabel("%")
        # ax.legend(['start', 'end'])
        # if saving:
        #     plt.savefig(f'{mice[i].name}_block_wait_variance.svg')
        # plt.cla()

        # create large violin for all animal
   #     locs, labels = plt.xticks()
        #     plt.xticks(np.arange(4), block_type)
        fig, ax = plt.subplots(figsize=(10, 5))
        plt.plot(mice[i].bg_restart_s, 'b--')
        plt.plot(mice[i].bg_restart_l, 'r--')
        ax.set_title(f'{mice[i].name} percent trial restarted')
        ax.set_xlabel("session")
        ax.set_ylabel("%")
        ax.legend(['short', 'long'])
        if saving:
            plt.savefig(f'{mice[i].name} percent trial restarted.svg')
        plt.close()


def violins(mice, task_params, has_block, saving):
    if has_block:
        path = os.path.normpath(r'D:\figures\behplots') + "\\" + "blocks" + "\\" + task_params
    else:
        path = os.path.normpath(r'D:\figures\behplots') + "\\" + "no_blocks" + "\\" + task_params
    os.chdir(path)
    print(f'plotting and saving in {path}')
    violin_vars = []
    labels = []
    for i in range(len(mice)):
        violin_vars.append(mice[i].blk_start_var)
        violin_vars.append(mice[i].blk_end_var)
        labels.append(f'{mice[i].name} start')
        labels.append(f'{mice[i].name} end')

    fig = plt.figure(facecolor=(1, 1, 1))
    # Create an axes instance
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlabel('event type')
    ax.set_ylabel('variance')
    # Create the boxplot
    bp = ax.violinplot(violin_vars, showmeans=True)
    for i, pc in enumerate(bp["bodies"], 1):
        if i % 2 != 0:
            pc.set_facecolor('yellow')
        else:
            pc.set_facecolor('green')
    set_axis_style(ax, labels)
    if saving:
        print("saving violin plot")
        plt.savefig(f'violin_var.svg', bbox_inches='tight')
    plt.close()


    violin_times = []
    labels = []
    for i in range(len(mice)):
        violin_times.append(mice[i].holding_s_mean)
        violin_times.append(mice[i].holding_l_mean)
        labels.append(f'{mice[i].name} s')
        labels.append(f'{mice[i].name} l')

    fig = plt.figure(facecolor=(1, 1, 1))
    # Create an axes instance
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlabel('trial type')
    ax.set_ylabel('holding time (s)')
    # Create the boxplot
    bp = ax.violinplot(violin_times, showmeans=True)
    for i, pc in enumerate(bp["bodies"], 1):
        if i % 2 != 0:
            pc.set_facecolor('blue')
        else:
            pc.set_facecolor('red')

    set_axis_style(ax, labels)
    if saving:
        plt.savefig(f'violin_time.svg', bbox_inches='tight')
    plt.close()

    violin_stable_times = []
    labels = []
    for i in range(len(mice)):
        violin_stable_times.append(mice[i].stable_s)
        print(len(mice[i].stable_s))
        violin_stable_times.append(mice[i].stable_l)
        # violin_stable_times.append(mice[i].moving_average_s)
        # violin_stable_times.append(mice[i].moving_average_l)
        labels.append(f'{mice[i].name} s')
        labels.append(f'{mice[i].name} l')

    fig = plt.figure(facecolor=(1, 1, 1))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlabel('trial type (stable)')
    ax.set_ylabel('holding time (s)')
    # Create the boxplot
    bp = ax.violinplot(violin_stable_times, showmeans=True)
    for i, pc in enumerate(bp["bodies"], 1):
        if i % 2 != 0:
            pc.set_facecolor('blue')
        else:
            pc.set_facecolor('red')

    set_axis_style(ax, labels)
    if saving:
        plt.savefig(f'violin_stable_time.svg', bbox_inches='tight')
    plt.close()

    violin_moving_var = []
    labels = []
    for i in range(len(mice)):
        violin_moving_var.append(mice[i].moving_average_s_var)
        violin_moving_var.append(mice[i].moving_average_l_var)
        labels.append(f'{mice[i].name} s')
        labels.append(f'{mice[i].name} l')

    fig = plt.figure(facecolor=(1, 1, 1))
    # Create an axes instance
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlabel('trial type')
    ax.set_ylabel('variance')
    # Create the boxplot
    bp = ax.violinplot(violin_moving_var, showmeans=True)
    for i, pc in enumerate(bp["bodies"], 1):
        if i % 2 != 0:
            pc.set_facecolor('blue')
        else:
            pc.set_facecolor('red')

    set_axis_style(ax, labels)
    if saving:
        plt.savefig(f'violin_moving_var.svg', bbox_inches='tight')
    plt.close()

    for i in range(len(mice)):
        violin_var_split = []
        labels = []
        violin_var_split.append(mice[i].sl_blk_start_var)
        violin_var_split.append(mice[i].ls_blk_start_var)
        violin_var_split.append(mice[i].sl_blk_end_var)
        violin_var_split.append(mice[i].ls_blk_end_var)

        labels.append(f'{mice[i].name} s-l starts')
        labels.append(f'{mice[i].name} l-s starts')
        labels.append(f'{mice[i].name} s-l ends')
        labels.append(f'{mice[i].name} l-s ends')

        fig = plt.figure(facecolor=(1, 1, 1))
        # Create an axes instance
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlabel('block transition type')
        ax.set_ylabel('waiting time variance')
        # Create the boxplot
        bp = ax.violinplot(violin_var_split, showmeans=True)
        for j, pc in enumerate(bp["bodies"], 1):
            if j % 1 == 0:
                pc.set_facecolor('yellow')
            elif j % 2 == 0:
                pc.set_facecolor('green')
            elif j % 3 == 0:
                pc.set_facecolor('blue')
            else:
                pc.set_facecolor('red')

        set_axis_style(ax, labels)
        if saving:
            plt.savefig(f'{mice[i].name}_violin_var_split.svg', bbox_inches='tight')
        plt.close()

# plots about a particular session index (usually the last to see current stats)
def plotSession(mice, session, task_params, has_block, saving):
    if has_block:
        path = os.path.normpath(r'D:\figures\behplots') + "\\" + "blocks" + "\\" + task_params
    else:
        path = os.path.normpath(r'D:\figures\behplots') + "\\" + "no_blocks" + "\\" + task_params
    os.chdir(path)
    print(f'plotting and saving in {path}')
    violin_vars = []
    labels = []
    # session reward rate
    for i in range(len(mice)):
        curr_animal_path = path + '\\' + mice[i].name
        os.chdir(curr_animal_path)
        file_path = curr_animal_path + '\\' + os.listdir()[0]

        session_to_plot = mice[i].session_list[session]

        fig = plt.figure(facecolor=(1, 1, 1))
        # ax = fig.add_axes([0, 0, 1, 1])

        plt.plot(session_to_plot.session_reward_rate, 'b--')
        plt.title = f'{mice[i].name} the {session} session reward rate'
        plt.xlabel('session time')
        plt.ylabel('reward rate (ul/s)')
        if saving:
            plt.savefig(f'{mice[i].name} {session} session reward rate.svg')
        plt.close()

        # groupings of short vs long blocks wait times
        fig = plt.figure(facecolor=(1, 1, 1))
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlabel('blocks within session')
        ax.set_ylabel('wait time')
        all_licks = session_to_plot.session_holding_times
        print(len(all_licks))
        for j in range(1, len(all_licks)):
            if session_to_plot.block_type[0] == 's':
                if j % 2 == 0:
                    labels.append("s")
                else:
                    labels.append("l")
            else:
                if j % 2 == 0:
                    labels.append("l")
                else:
                    labels.append("s")
        print(session_to_plot.file_path)
        bp = ax.violinplot(all_licks, showmeans=True)
        for i, pc in enumerate(bp["bodies"], 1):
            if i % 2 != 0:
                pc.set_facecolor('blue')
            else:
                pc.set_facecolor('red')

        set_axis_style(ax, labels)
        if saving:
            plt.savefig(f'session_blk_lick_times.svg', bbox_inches='tight')
        plt.close()

