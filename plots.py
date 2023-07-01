import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap
import matplotlib.animation as animation
import matplotlib.colors as colors


def set_axis_style(ax, labels):
    ax.set_xticks(np.arange(1, len(labels) + 1), labels=labels)
    ax.set_xlim(0.25, len(labels) + 0.75)


def rawPlots(mice, task_params, saving):
    path = os.path.normpath(r'D:\figures\behplots') + "\\" + task_params
    os.chdir(path)
    print(f'plotting and saving in {path}')
    for i in range(len(mice)):
        fig, ax = plt.subplots(figsize=(10, 5))
        plt.plot(mice[i].holding_s_blk, 'bo')
        plt.plot(mice[i].holding_l_blk, 'ro')
        ax.set_title(f'{mice[i].name} holding times')
        ax.set_xlabel("session")
        ax.set_ylabel("holding time(s)")
        ax.legend(['short', 'long'])
        plt.savefig(f'{mice[i].name}_holding_times.svg')
        plt.cla()

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.set_title(f'{mice[i].name} holding - optimal time')
        plt.plot(mice[i].opt_diff_s, 'b+')
        plt.plot(mice[i].opt_diff_l, 'r+')
        ax.legend(['short', 'long'])
        ax.set_xlabel("session")
        ax.set_ylabel("avg licking - optimal(s)")
        plt.savefig(f'{mice[i].name} avg licking - optimal.svg')
        plt.cla()

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
        plt.cla()

        fig, ax = plt.subplots(figsize=(10, 5))
        plt.plot(mice[i].lick_prob_s, 'b-')
        plt.plot(mice[i].lick_prob_l, 'r-')
        ax.set_title(f'{mice[i].name} prob at licking')
        ax.set_xlabel("session")
        ax.set_ylabel("%")
        ax.legend(['short', 'long'])
        if saving:
            plt.savefig(f'{mice[i].name}_prob_at_licking.svg')
        plt.cla()

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
        plt.cla()

        fig, ax = plt.subplots(figsize=(10, 5))
        plt.plot(mice[i].holding_s_by_block, 'bo')
        plt.plot(mice[i].holding_l_by_block, 'ro')
        ax.set_title(f'{mice[i].name} holding times by block')
        ax.set_xlabel("block")
        ax.set_ylabel("holing time(s)")
        ax.legend(['short', 'long'])
        if saving:
            plt.savefig(f'{mice[i].name}_holding_times_by_block.svg')
        plt.cla()

        fig, ax = plt.subplots(figsize=(10, 5))
        plt.plot(mice[i].blk_miss_perc_s, 'b+')
        plt.plot(mice[i].blk_miss_perc_l, 'r+')
        ax.set_title(f'{mice[i].name} missed trials percentage')
        ax.set_xlabel("block")
        ax.set_ylabel("%")
        ax.legend(['short', 'long'])
        if saving:
            plt.savefig(f'{mice[i].name}_missed_trials_perc.svg')
        plt.cla()

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


def violins(mice, task_params, saving):
    path = os.path.normpath(r'D:\figures\behplots') + "\\" + task_params
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
            pc.set_facecolor('blue')
        else:
            pc.set_facecolor('red')
    set_axis_style(ax, labels)
    if saving:
        print("saving violin plot")
        plt.savefig(f'violin_var.svg', bbox_inches='tight')
    plt.cla()


    violin_times = []
    labels = []
    for i in range(len(mice)):
        violin_times.append(mice[i].blk_holding_s)
        violin_times.append(mice[i].blk_holding_l)
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
    plt.cla()

    violin_stable_times = []
    labels = []
    for i in range(len(mice)):
        violin_stable_times.append(mice[i].stable_s)
        violin_stable_times.append(mice[i].stable_l)
        labels.append(f'{mice[i].name} s')
        labels.append(f'{mice[i].name} l')

    fig = plt.figure(facecolor=(1, 1, 1))
    # Create an axes instance
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
    plt.cla()