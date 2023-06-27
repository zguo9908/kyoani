import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap
import matplotlib.animation as animation
import matplotlib.colors as colors


path = os.path.normpath(r'D:\figures\behplots')
os.chdir(path)


def set_axis_style(ax, labels):
    ax.set_xticks(np.arange(1, len(labels) + 1), labels=labels)
    ax.set_xlim(0.25, len(labels) + 0.75)

def rawPlots(mice):
    animal_num = len(mice)
    for i in range(animal_num):
        fig, ax = plt.subplots(figsize=(10, 5))
        plt.plot(mice[i].holding_s_blk, 'bo')
        plt.plot(mice[i].holding_l_blk, 'ro')
        ax.set_title(f'{mice[i].name} holding times')
        ax.set_xlabel("session")
        ax.set_ylabel("holing time(s)")
        ax.legend(['short', 'long'])
        plt.savefig('ex1.svg')

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.set_title(f'{mice[i].name} holding - optimal time')
        plt.plot(mice[i].opt_diff_s, 'b+')
        plt.plot(mice[i].opt_diff_l, 'r+')
        ax.legend(['short', 'long'])
        ax.set_xlabel("session")
        ax.set_ylabel("avg licking - optimal(s)")
        if i == 2:
            plt.savefig('ex1.svg')
        #     locs, labels = plt.xticks()
        #     plt.xticks(np.arange(4), block_type)
        fig, ax = plt.subplots(figsize=(10, 5))
        plt.plot(mice[i].perc_rewarded_s, 'b--')
        plt.plot(mice[i].perc_rewarded_l, 'r--')
        ax.set_title(f'{mice[i].name} percent trial rewarded')
        ax.set_xlabel("session")
        ax.set_ylabel("%")
        ax.legend(['short', 'long'])

        fig, ax = plt.subplots(figsize=(10, 5))
        plt.plot(mice[i].lick_prob_s, 'b-')
        plt.plot(mice[i].lick_prob_l, 'r-')
        ax.set_title(f'{mice[i].name} prob at licking')
        ax.set_xlabel("session")
        ax.set_ylabel("%")
        ax.legend(['short', 'long'])

        fig, ax = plt.subplots(figsize=(10, 5))
        plt.plot(mice[i].moving_average_s, 'bo')
        plt.plot(mice[i].moving_average_l, 'ro')
        ax.set_title(f'{mice[i].name} moving avg performance')
        ax.set_xlabel("trial")
        ax.set_ylabel("holing time(s)")
        ax.legend(['short', 'long'])
        plt.axis([1400, 2000, 0, 13])

        fig, ax = plt.subplots(figsize=(10, 5))
        plt.plot(mice[i].holding_s_by_block, 'bo')
        plt.plot(mice[i].holding_l_by_block, 'ro')
        ax.set_title(f'{mice[i].name} holding times by block')
        ax.set_xlabel("block")
        ax.set_ylabel("holing time(s)")
        ax.legend(['short', 'long'])

        fig, ax = plt.subplots(figsize=(10, 5))
        plt.plot(mice[i].blk_miss_perc_s, 'b+')
        plt.plot(mice[i].blk_miss_perc_l, 'r+')
        ax.set_title(f'{mice[i].name} missed trials percentage')
        ax.set_xlabel("block")
        ax.set_ylabel("%")
        ax.legend(['short', 'long'])

        fig = plt.figure(facecolor=(1, 1, 1))
        # Create an axes instance
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlabel('trial type')
        ax.set_ylabel('holding time (s)')
        # Create the boxplot
        violin = [mice[i].blk_holding_s, mice[i].blk_holding_l]
        bp = ax.violinplot(violin, showmeans=True)
        for i, pc in enumerate(bp["bodies"], 1):
            if i % 2 != 0:
                pc.set_facecolor('blue')
            else:
                pc.set_facecolor('red')

        labels = ['short', 'long']
        set_axis_style(ax, labels)

        plt.savefig(f'violin + {mice[i].name}.svg', bbox_inches='tight')