import math
import os
from sklearn.neighbors import KernelDensity
from scipy.signal import find_peaks
from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import matplotlib.animation as animation
import matplotlib.colors as mcolors
import utils
import random
import ruptures as rpt

def run_all_single_animal_plot(mice, optimal_wait, task_params, has_block):
    rawPlots(mice, optimal_wait, task_params=task_params, has_block=has_block, saving=True)
    violins(mice, task_params=task_params, has_block=has_block, saving=False)
    plotSession(mice, -1, task_params=task_params, has_block=has_block, saving=True)
    plot_all_animal_scatter(mice, has_block=has_block, task_params=task_params)
   #plot_change_points_test(mice, has_block=has_block, task_params=task_params)

def plot_change_points_test(mice, has_block, task_params):
    path, user = utils.set_plotting_path(has_block, task_params)
    os.chdir(path)
    print(f'plotting and saving in {path}')
    for i in range(len(mice)):
        if user == 'ziyi':
            curr_animal_path = path + '\\' + mice[i].name
        else:
            curr_animal_path = path + '/' + mice[i].name
        os.chdir(curr_animal_path)
        np.random.seed(42)  # for reproducibility
        merged_perf = utils.merge_lists(mice[i].holding_s_mean, mice[i].holding_l_mean)
        # print([type(item) for item in merged_perf])

        # Convert to NumPy array with a specific dtype, such as float
        signal = np.array(merged_perf, dtype=float)

        # Change point detection
        model = "l2"
        algo = rpt.Binseg(model=model).fit(signal)
        result = algo.predict(pen=8)

        background = ['white', 'gray']
        start = 0
        for idx, end in enumerate(result+[len(signal)]):
            plt.axvspan(start, end, facecolor=background[idx % len(background)], alpha=0.5)
            start = end
        # Display
        # rpt.display(signal, result)
        plt.plot(signal)
        plt.xlabel("session")
        plt.ylabel("wait time (s)")
        plt.axvline(x=mice[i].reverse_index, color='black', linestyle='-', linewidth=1)

        # plt.title('Change Point Detection on Sample Animal Behavior Data')
        plt.savefig(f'{mice[i].name} session change detection test.svg')
        plt.close()


def plot_group_diff(g1, g2, categories, type, plot_patch, find_sig, plot_optimal, **kwargs):
    g1_mean, g1_std, padded_g1 = utils.calculate_padded_averages_and_std(g1)
    g2_mean, g2_std, padded_g2 = utils.calculate_padded_averages_and_std(g2)

    fig, ax = plt.subplots()
    # Plot the line graph for long sessions
    if plot_patch:
        reverse_index = kwargs.get("num_before_transition")
        x = [i - reverse_index for i in range(1, len(g2_mean) + 1)]
        ax.axvspan(min(x), 0.5, color='yellow', alpha=0.1, label='default')
        ax.axvspan(0.5, max(x), color='green', alpha=0.1, label='change')
        # Plot the line graph for short sessions
    else:
        x = list(range(1, len(g2_mean) + 1))
    # plot long
    y = g1_mean
    if len(x) != len(y):
        x_long = list(range(1, len(g1_mean) + 1))
    else:
        x_long = x
    ax.plot(x_long, y, marker='o', label=f'Average_{categories[0]}_group', color='red')

    # Shade the area around the line plot to represent the standard deviation for long sessions
    ax.fill_between(x_long, [mean - std for mean, std in zip(y, g1_std)],
                    [mean + std for mean, std in zip(y, g1_std)], alpha=0.5,
                    label=f'Standard Deviation_{categories[0]}',
                    color='#FFAAAA')
    # plot short
    y = g2_mean
    ax.plot(x, y, marker='o', label=f'Average_{categories[1]}_group', color='blue')

    # Shade the area around the line plot to represent the standard deviation for short sessions
    ax.fill_between(x, [mean - std for mean, std in zip(y, g2_std)],
                    [mean + std for mean, std in zip(y, g2_std)], alpha=0.5,
                    label=f'Standard Deviation_{categories[1]}',
                    color='lightblue')

    ax.set_xlabel('session #')

    if type == 'perc':
        ax.set_ylabel('%')
    elif type == 'time':
        ax.set_ylabel('time (s)')
    elif type == 'count':
        ax.set_ylabel('#')
    elif type == 'rate':
        ax.set_ylabel('ul/s')
    ax.legend()

    if find_sig:
        sig = utils.compare_lists_with_significance(padded_g1, padded_g2)
        for i in sig['high']:
            ax.annotate('***', xy=(x[i], y[i]), xytext=(x[i], 0), textcoords='data',
                        arrowprops=dict(arrowstyle="->"), rotation='vertical')

        for i in sig['low']:
            ax.annotate('*', xy=(x[i], y[i]), xytext=(x[i], 0), textcoords='data',
                        arrowprops=dict(arrowstyle="->"), rotation='vertical')

    if plot_optimal:
        for key, value in kwargs.items():
            print(f"Optional {key}: {value}")
            if key == "opt_long":
                ax.axhline(y=kwargs.get('opt_long'), color='r', linestyle='--', label='Optimal_long')
            if key == "opt_short":
                ax.axhline(y=kwargs.get('opt_short'), color='b', linestyle='--', label='Optimal_short')
            if key == "adjusted_long":
                y = kwargs.get('adjusted_long')
                plt.step(x_long, y, where='post', color='lightcoral', label='adjusted_long_cohort')
                # ax.axhline(y=kwargs.get('adjusted_long'), color='lightcoral', linestyle='--',
                #            label='adjusted_optimal_long')
            if key == "adjusted_short":
                # ax.axhline(y=kwargs.get('adjusted_short'), color='lightblue', linestyle='--',
                #            label=
                y = kwargs.get('adjusted_short')
                plt.step(x, y, where='post', color='lightblue',label='adjusted_short_cohort')
    return g1_mean, g2_mean


def plot_all_animal_waiting(long_mice_list, long_session_mean, short_mice_list, short_session_mean ):
    fig, ax = plt.subplots()
    # Iterate through each sublist and plot it as a line
    for mice, animal_sessions in zip(long_mice_list, long_session_mean):
        # print(animal_sessions)
        x = list(range(1, len(animal_sessions) + 1))  # Generate x values (1, 2, 3, ...)
        y = animal_sessions
        ax.plot(x, y, marker='o', label=mice)

    for mice, animal_sessions in zip(short_mice_list, short_session_mean):
        # print(animal_sessions)
        x = list(range(1, len(animal_sessions) + 1))  # Generate x values (1, 2, 3, ...)
        y = animal_sessions
        ax.plot(x, y, marker='o', label=mice)
    # Customize the plot
    ax.set_xlabel('sessions')
    ax.set_ylabel('mean waiting time')
    ax.legend()
    plt.savefig('all animal waiting.svg')

def adjust_color_intensity(color, factor):
    r, g, b = mcolors.to_rgb(color)
    # Ensure color gets denser for later sessions
    factor = factor ** 1.12 # Adjust this exponent to control the gradient
    return (r + (1 - r) * (1 - factor), g + (1 - g) * (1 - factor), b + (1 - b) * (1 - factor))

def plot_all_animal_scatter(mice, has_block, task_params):
    path,_ = utils.set_plotting_path(has_block, task_params)
    os.chdir(path)
    print(f'plotting and saving in {path}')
    # Manually define specific colors for the red and blue families
    red_family_colors = ['#FF00FF', '#FFC0CB', '#F08080', '#FA8072',
                         '#DC143C']  # Magenta, Pink, Light Coral, Salmon, Crimson
    blue_family_colors = ['#00FFFF', '#87CEEB', '#4169E1', '#0047AB',
                          '#000080']  # Cyan, Sky Blue, Royal Blue, Cobalt, Navy


    fig, ax = plt.subplots(figsize=(8,8))
    plt.xticks(np.arange(0, 6, 1))  # Adjust the x-ticks range as needed
    plt.yticks(np.arange(0, 15, 1))  # Adjust the y-ticks range as needed
    legend_handles = []  # For custom legend
    # Initialize a color counter for each group
    color_counter_short = 0
    color_counter_long = 0

    # Iterate over each mouse and plot their data

    for mouse in mice:

        if mouse.default == 'short':
            # Assign a color from the short (blue) family
            color = blue_family_colors[color_counter_short % len(blue_family_colors)]
            color_counter_short += 1

            # Your existing code to set the data for short group
            session_avg = mouse.holding_s_mean[20:mouse.default_session_num]

        elif mouse.default == 'long':
            # Assign a color from the long (red) family
            color = red_family_colors[color_counter_long % len(red_family_colors)]
            color_counter_long += 1

            # Your existing code to set the data for long group
            session_avg = mouse.holding_l_mean[20:mouse.default_session_num]

        adjusted_optimal_default = mouse.session_adjusted_optimal[20:mouse.default_session_num]
        #     # Adjust the color intensity based on the session number for gradient effect
        # for session_num in range(mouse.default_session_num):
        #     color_intensity = (session_num + 1) / mouse.default_session_num
        #     adjusted_color = adjust_color_intensity(color, color_intensity)
        #
        #     plt.scatter(adjusted_optimal_default[session_num], session_avg[session_num],
        #                 color=adjusted_color, edgecolors='black')
        # legend_handles.append(mpatches.Patch(color=color, label=mouse.name))

        jitter = 0.1
        point_size = 32

        # Adjust the color intensity based on the session number for gradient effect
        for session_num in range(len(adjusted_optimal_default)):
            color_intensity = (session_num + 1) / mouse.default_session_num
            adjusted_color = adjust_color_intensity(color, color_intensity)

            # Jittering the x and y coordinates
            x_jitter = adjusted_optimal_default[session_num] + random.uniform(-jitter, jitter)
            y_jitter = session_avg[session_num] + random.uniform(-jitter, jitter)

            plt.scatter(x_jitter, y_jitter, color=adjusted_color, edgecolors='black', s=point_size)

        legend_handles.append(mpatches.Patch(color=color, label=mouse.name))

    x_d = np.linspace(0, 6, 1000)  # Adjust range as needed
    y_d = x_d  # Diagonal line where y equals x

    # Plotting the diagonal line
    plt.plot(x_d, y_d, label='y = x')
    plt.gca().set_aspect('equal', adjustable='box')
    # Creating legend handles for red colors and blue colors
    # Creating legend handles for red colors and blue colors
    red_handles = [mpatches.Patch(color='red', label='Long Cohort')]
    blue_handles = [mpatches.Patch(color='blue', label='Short Cohort')]

    # Plotting the legend with two columns
    plt.legend(handles=red_handles + blue_handles, bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)

    # Adding legend for individual animals
    animal_handles = [
        mpatches.Patch(color='#FF00FF', label='Mouse1'),  # Magenta
        mpatches.Patch(color='#FFC0CB', label='Mouse2'),  # Pink
        mpatches.Patch(color='#F08080', label='Mouse3'),  # Light Coral
        mpatches.Patch(color='#FA8072', label='Mouse4'),  # Salmon
        mpatches.Patch(color='#DC143C', label='Mouse5'),  # Crimson
        mpatches.Patch(color='#00FFFF', label='Mouse6'),  # Cyan
        mpatches.Patch(color='#87CEEB', label='Mouse7'),  # Sky Blue
        mpatches.Patch(color='#4169E1', label='Mouse8'),  # Royal Blue
        mpatches.Patch(color='#0047AB', label='Mouse9'),  # Cobalt
        mpatches.Patch(color='#000080', label='Mouse10')  # Navy
    ]

    plt.legend(handles=animal_handles, bbox_to_anchor=(1.05, 0.5), loc='center left', title='Individual Animals')
    plt.tight_layout()  # Adjust layout for better appearance
    # Set labels and title
    plt.xlabel('adjusted optimal leaving time (s)')
    plt.ylabel('session avg leaving time (s)')
    plt.savefig(f'all animal color mean vs adjusted.svg')
    plt.close()


def set_axis_style(ax, labels):
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_xlim(0.25, len(labels) + 0.75)


# Define a function to find the modes in a distribution and estimate standard deviations
def find_modes_and_stds(data):
    kde = KernelDensity(bandwidth=0.5)  # Adjust the bandwidth as needed
    kde.fit(data.to_numpy().reshape(-1, 1))  # Convert to a numpy array and reshape
    x_d = np.linspace(min(data), max(data), 1000)
    logprob = kde.score_samples(x_d.reshape(-1, 1))
    density = np.exp(logprob)

    peaks, _ = find_peaks(density)

    modes = []
    stds = []

    for _ in range(min(2, len(peaks))):  # Consider up to the first 2 peaks
        if len(peaks) > 0:
            mode = x_d[peaks[np.argmax(density[peaks])]]
            density[peaks[np.argmax(density[peaks])]] = 0  # Zero out the peak
            modes.append(mode)

            # Estimate the standard deviation based on the IQR
            q1, q3 = np.percentile(data, [25, 75])
            iqr = q3 - q1
            std_estimate = iqr / 1.349  # Approximately 1.349 times the IQR
            stds.append(std_estimate)

    return modes, stds


def background_patches(default, mouse, ax, x):
    if default == 'long':
        ax.axvspan(min(x), mouse.reverse_index + 0.5, color='red', alpha=0.1)  # Red patch before x_split
        ax.axvspan(mouse.reverse_index + 0.5, max(x), color='blue', alpha=0.1)
    else:
        ax.axvspan(min(x), mouse.reverse_index + 0.5, color='blue', alpha=0.1)  # Red patch before x_split
        ax.axvspan(mouse.reverse_index + 0.5, max(x), color='red', alpha=0.1)


def safe_extract(lst, source_label):
    return [x[0] if isinstance(x, tuple) and len(x) > 1 and x[1] == source_label else None for x in lst]


def plotTrialSplit(mouse, default):
    colors = ['blue', 'green', 'red']
    trial_type = ['miss', 'repeat', 'good']
    # Width of each bar
    bar_width = 0.35
    fig, ax = plt.subplots(figsize=(15, 5))
  # Create a new figure and axes for each plot
    if mouse.default_session_num > 0 and mouse.change_session_num > 0:
        # Example list pairs

        list_pairs = [(mouse.miss_perc_s, mouse.miss_perc_l), (mouse.bg_restart_s, mouse.bg_restart_l)]  # and so on...

        merged_lists_with_sources = [utils.merge_lists_with_sources(list1, list2) for list1, list2 in list_pairs]
        if len(merged_lists_with_sources) >= 2:
            # Extracting values for operation
            merged_miss_perc = [x[0] for x in merged_lists_with_sources[0]]
            merged_bg_restart = [x[0] for x in merged_lists_with_sources[1]]

        good_perc = [1 - (p1 + p2) for p1, p2 in zip(merged_miss_perc, merged_bg_restart)]
        session_num = len(merged_miss_perc)
        # print(f'number of sessions plotting is {session_num}')
        x = np.arange(session_num)
        for j in range(session_num):  # Iterate over the sessions
            x_values = [x[j]]  # X-coordinate for the current session
            bottom = [0]  # Initialize the bottom values for the bars
            for trial in range(3):  # Iterate over the three trial types
                data = 0  # Initialize data for the current trial type
                if trial == 0:
                    data = merged_miss_perc[j]
                elif trial == 1:
                    data = merged_bg_restart[j]
                else:
                    data = good_perc[j]

                plt.bar(x_values, data, width=bar_width, label=trial_type[trial], color=colors[trial],
                        bottom=bottom)
                bottom[0] += data
    else:
        if len(mouse.holding_s_std) > 0:
            session_num = sum(1 for x in mouse.holding_s_mean if not math.isnan(x))
            x = np.arange(session_num)
            good_perc = [1 - (p1 + p2) for p1, p2 in zip(mouse.miss_perc_s, mouse.bg_restart_s)]
            for j in range(session_num):  # Iterate over the sessions
                x_values = [x[j]]  # X-coordinate for the current session
                bottom = [0]  # Initialize the bottom values for the bars

                for trial in range(3):  # Iterate over the three trial types
                    data = 0  # Initialize data for the current trial type
                    if trial == 0:
                        data = mouse.miss_perc_s[j]
                    elif trial == 1:
                        data = mouse.bg_restart_s[j]
                    else:
                        data = good_perc[j]

                    plt.bar(x_values, data, width=bar_width, label=trial_type[trial], color=colors[trial],
                            bottom=bottom)
                    bottom[0] += data
        if len(mouse.holding_l_std) > 0:
            session_num = sum(1 for x in mouse.holding_l_mean if not math.isnan(x))
            x = np.arange(session_num)
            good_perc = [1 - (p1 + p2) for p1, p2 in zip(mouse.miss_perc_l, mouse.bg_restart_l)]
            for j in range(session_num):  # Iterate over the sessions
                x_values = [x[j]]  # X-coordinate for the current session
                bottom = [0]  # Initialize the bottom values for the bars
                for trial in range(3):  # Iterate over the three trial types
                    data = 0  # Initialize data for the current trial type
                    if trial == 0:
                        data = mouse.miss_perc_l[j]
                    elif trial == 1:
                        data = mouse.bg_restart_l[j]
                    else:
                        data = good_perc[j]
                    plt.bar(x_values, data, width=bar_width, label=trial_type[trial], color=colors[trial],
                            bottom=bottom)
                    bottom[0] += data
    background_patches(default, mouse, ax, x)

    ax.set_xlabel('Sessions')
    ax.set_ylabel('Percentage')
    ax.set_title(f'{mouse.name} Percentage of Different Trial Types')
    ax.set_xticks(x)
    ax.set_xticklabels(range(1, session_num + 1))
    ax.legend(['default', 'change'] + trial_type)
    plt.savefig(f'{mouse.name}_perc_diff_trials.svg')
    plt.close()

def plotHoldingWithError(mouse, default, optimal_wait):
    fig, ax = plt.subplots(figsize=(10, 5))
    if mouse.default_session_num > 0 and mouse.change_session_num > 0:
        list_pairs = [(mouse.holding_s_median, mouse.holding_l_median),
                      (mouse.holding_s_mean, mouse.holding_l_mean),
                      (mouse.holding_s_q25, mouse.holding_l_q25),
                      (mouse.holding_s_q75, mouse.holding_l_q75)]  # and so on...

        merged_lists_with_sources = [utils.merge_lists_with_sources(list1, list2) for list1, list2 in list_pairs]

        if len(merged_lists_with_sources) >= 2:
            # Extracting values for operation
            merged_median = [x[0] for x in merged_lists_with_sources[0]]
            merged_mean = [x[0] for x in merged_lists_with_sources[1]]
            merged_q25 = [x[0] for x in merged_lists_with_sources[2]]
            merged_q75 = [x[0] for x in merged_lists_with_sources[3]]
        # print(f'merged median {merged_median}')

        error_low = [max(0, float(median) - float(q25)) for median, q25 in
                     zip(merged_median, merged_q25)]
        error_high = [max(0, float(q75) - float(median)) for q75, median in
                      zip(merged_q75, merged_median)]

        values_s_median = safe_extract(merged_median, 'List 1')
        values_l_median = safe_extract(merged_median, 'List 2')
        values_s_mean = safe_extract(merged_mean, 'List 1')
        values_l_mean = safe_extract(merged_mean, 'List 2')

        x = np.arange(1, len(merged_median)+1)
        plt.errorbar(x, merged_median, yerr=[error_low, error_high], fmt='o', color='black', capsize=5, elinewidth=2,
                     barsabove=True, errorevery=1)
        #print(f' where is value {values_s_median}')
        #plt.plot(x, values_s_median, 'bs', label='Short - median', markersize=8, markerfacecolor='none')
        #plt.plot(x, values_l_median, 'rs', label='Long - median', markersize=8, markerfacecolor='none')
        #plt.plot(x, values_s_mean, 'bo', label='Short - mean', markersize=8, markerfacecolor='none')
        #plt.plot(x, values_l_mean, 'ro', label='Long - mean', markersize=8, markerfacecolor='none')
        ax.axhline(y=optimal_wait[1], color='r', linestyle='--', label='Optimal_long')
        ax.axhline(y=optimal_wait[0], color='b', linestyle='--', label='Optimal_Short')

        background_patches(default, mouse, ax, x)
    else:
        if len(mouse.holding_s_std) > 0:
            x = range(1, sum(1 for x in mouse.holding_s_mean if not math.isnan(x)) + 1)
            error_low = [max(0, float(median) - float(q25)) for median, q25 in
                         zip(mouse.holding_s_median, mouse.holding_s_q25)]
            error_high = [max(0, float(q75) - float(median)) for q75, median in
                          zip(mouse.holding_s_q75, mouse.holding_s_median)]

            plt.errorbar(x, mouse.holding_s_median, yerr=[error_low, error_high], fmt='o', color='blue',
                         label='Short - median', capsize=5, elinewidth=2, barsabove=True, errorevery=1)
            plt.plot(x, mouse.holding_s_median, 'bs', markersize=8, markerfacecolor='none')
            plt.plot(x, mouse.holding_s_mean, 'bo', label='Short - mean', markersize=8, markerfacecolor='none')
        if len(mouse.holding_l_std) > 0:
            x = range(1, sum(1 for x in mouse.holding_l_mean if not math.isnan(x)) + 1)
            error_low = [max(0, float(median) - float(q25)) for median, q25 in
                         zip(mouse.holding_l_median, mouse.holding_l_q25)]
            error_high = [max(0, float(q75) - float(median)) for q75, median in
                          zip(mouse.holding_l_q75, mouse.holding_l_median)]
            print(len(x))
            print(mouse.holding_l_median)
            plt.errorbar(x, mouse.holding_l_median, yerr=[error_low, error_high], fmt='o', color='red',
                         label='Long - median', capsize=5, elinewidth=2, barsabove=True, errorevery=1)
            plt.plot(x, mouse.holding_l_median, 'rs', markersize=8, markerfacecolor='none')
            plt.plot(x, mouse.holding_l_mean, 'ro', label='Long - mean', markersize=8, markerfacecolor='none')

    ax.set_title(f'{mouse.name} holding times')
    ax.set_xlabel("session number")
    ax.set_ylabel("holding time(s)")
    ax.legend(loc='upper right')

    plt.savefig(f'{mouse.name}_holding_times.svg')
    plt.close()

def plotJitteredDist(mouse, curr_default):
    fig, axes = plt.subplots(2, 1, figsize=(20, 10))
    # fig, ax = plt.subplots(figsize=(10, 5))
    # Loop through each day and plot jittered data points

    for session, values in enumerate(mouse.all_holding_s_list):
        if curr_default == "long":
            session = session + len(mouse.all_holding_l_list)

        jittered_session = np.random.normal(session, 0.1, len(values))
        axes[0].scatter(jittered_session, values, alpha=0.1, c='blue')

        # Find the modes and estimate the standard deviations
        modes, stds = find_modes_and_stds(values)
        # print(f'modes for the current session{modes} and std {stds}')

        for mode, std in zip(modes, stds):
            if mode is not None:
                # Calculate the bell curves for the identified peaks
                y = np.linspace(min(values), max(values), 100)
                bell_curve = norm.pdf(y, mode, std)
                bell_curve /= max(bell_curve)
                # Overlay the vertical bell curves on top of the vertically jittered lick times
                x = jittered_session[0] + (bell_curve - min(bell_curve)) * 0.5  # Adjust the horizontal position
                axes[1].plot(x, y, linestyle='dotted', c='blue')
                axes[1].fill_betweenx(y, x, jittered_session[0], where=(x > jittered_session[0]),
                                      color='lightblue', alpha=0.5)

                axes[1].hlines(mode, xmin=x[0], xmax=x[-1], color='blue', linestyle='dotted')

    for session, values in enumerate(mouse.all_holding_l_list):
        if curr_default == "short":
            session = session + len(mouse.all_holding_s_list)
        jittered_session = np.random.normal(session, 0.1, len(values))
        axes[0].scatter(jittered_session, values, alpha=0.1, c='red')
        # Find the modes and estimate the standard deviations
        modes, stds = find_modes_and_stds(values)

        for mode, std in zip(modes, stds):
            if mode is not None:
                # Calculate the bell curves for the identified peaks
                y = np.linspace(min(values), max(values), 100)
                bell_curve = norm.pdf(y, mode, std)
                bell_curve /= max(bell_curve)
                # Overlay the vertical bell curves on top of the vertically jittered lick times
                x = jittered_session[0] + (bell_curve - min(bell_curve)) * 0.5  # Adjust the horizontal position
                axes[1].plot(x, y, linestyle='dotted', c='red')
                axes[1].fill_betweenx(y, x, jittered_session[0], where=(x > jittered_session[0]),
                                      color='lightcoral', alpha=0.5)
                axes[1].hlines(mode, xmin=x[0], xmax=x[-1], color='red', linestyle='dotted')

    axes[-1].set_xlabel('session')
    axes[-1].set_ylabel('time (s)')
    plt.savefig(f'{mouse.name} Lick Time Distribution with Bell Curves.svg')
    plt.close()


def plotPairingLists(slist, llist, default, figuretype):
    fig, ax = plt.subplots(figsize=(10, 5))
    merged_list = utils.merge_lists_with_sources(slist, llist)

    values_s = [x[0] if x[1] == 'List 1' else None for x in merged_list]
    values_l = [x[0] if x[1] == 'List 2' else None for x in merged_list]

    x = np.arange(1, len(merged_list)+1)
   # print(values_s)

    plt.plot(x, values_s, 'bo', label=f'{"Default" if default == "short" else "Change"} ')
    plt.plot(x, values_l, 'ro', label=f'{"Default" if default == "long" else "Change"}')
    if figuretype == 'perc':
        ax.set_ylabel('Percentage')
    elif figuretype == "time":
        ax.set_ylabel('time (s)')
    elif figuretype == 'count':
        ax.set_ylabel('#')
    ax.set_xlabel('Sessions')
    ax.set_xticks(x)

def rawPlots(mice, optimal_wait, task_params, has_block, saving):
    #path = os.path.normpath(r'D:\figures\behplots') + "\\" + task_params
    path, user = utils.set_plotting_path(has_block, task_params)
    print(f'plotting and saving in {path}')
    for i in range(len(mice)):
        if user == "ziyi":
            curr_animal_path = path + '\\' + mice[i].name
        else:
            curr_animal_path = path + '/' + mice[i].name
        if not os.path.exists(curr_animal_path):
            os.makedirs(curr_animal_path)
        if mice[i].default == 'long':
            curr_default = 'long'
            curr_change = 'short'
        else:
            curr_default= 'short'
            curr_change = 'long'
        # overall coloring
        color_list1, color_list2 = ('bo-', 'ro-') if curr_default == 'long' else ('ro-', 'bo-')

        os.chdir(curr_animal_path)

        # trial percentages
        plotTrialSplit(mice[i], curr_default)
        # holding time with error bars
        plotHoldingWithError(mice[i], curr_default, optimal_wait)
        # jittered daily distribution
        plotJitteredDist(mice[i], curr_default)

        #non impulsive licks
        fig, ax = plt.subplots(figsize=(10, 5))
        if len(mice[i].holding_s_std) > 0:
            plt.errorbar(range(len(mice[i].non_reflexive_s_mean)), mice[i].non_reflexive_s_mean, yerr=mice[i].non_reflexive_s_std,
                         fmt='o', color='blue')
            ax.legend(['short'])
        if len(mice[i].holding_l_std) > 0:
            plt.errorbar(range(len(mice[i].non_reflexive_l_mean)), mice[i].non_reflexive_l_mean, yerr=mice[i].non_reflexive_l_std,
                         fmt='o', color='red')
            ax.legend(['long'])
        ax.set_title(f'{mice[i].name} holding times')
        ax.set_xlabel("session number")
        ax.set_ylabel("holding time(s)")
        plt.savefig(f'{mice[i].name}_non_impulsive_licks.svg')
        plt.close()

        # reward binary, with session indices
        fig, ax = plt.subplots(figsize=(30, 5))
        # print(f's sessions lick index {(mice[i].all_holding_s_index)}')
        if curr_default == "long":
            mice[i].session_trial_index_s = [x + mice[i].session_trial_num_l[-1] for x in mice[i].session_trial_num_s]
            loc_trials_rewarded = mice[i].loc_trials_rewarded_l + mice[i].loc_trials_rewarded_s
        if curr_default == "short":
            mice[i].session_trial_index_l = [x + mice[i].session_trial_num_s[-1] for x in mice[i].session_trial_num_l]
            loc_trials_rewarded = mice[i].moving_average_s + mice[i].moving_average_l
        plt.plot(loc_trials_rewarded)
        for j in range(len(mice[i].session_trial_index_s)):
            plt.axvline(x=mice[i].session_trial_index_s[j], color='b')
        for j in range(len(mice[i].session_trial_index_l)):
            plt.axvline(x=mice[i].session_trial_index_l[j], color='r')
        ax.set_title(f'{mice[i].name} loc_trials_rewarded')
        ax.set_xlabel("trial")
        ax.set_ylabel("rewarded")
        ax.legend(['short', 'long'])
        plt.axis([0, 15000, 0, 1])
        if saving:
            plt.savefig(f'{mice[i].name}_reward_history.svg')
        plt.close()

        #---------------------holding mean - optimal-----------------------#
        plotPairingLists(mice[i].opt_diff_s, mice[i].opt_diff_l, curr_default, 'time')
        ax.set_title(f'{mice[i].name} holding - optimal time')
        plt.savefig(f'{mice[i].name} avg licking - optimal.svg')
        plt.close()

        # ---------------------background restarting licks-----------------------#
        plotPairingLists(mice[i].bg_restart_licks_s, mice[i].bg_restart_licks_l, curr_default, 'count')
        ax.set_title(f'{mice[i].name} background repeat lick average')
        plt.savefig(f'{mice[i].name} repeat triggered times.svg')
        plt.close()

        fig, ax = plt.subplots(figsize=(10, 5))
        plt.plot(mice[i].mean_consumption_length)
        ax.set_title(f'{mice[i].name} mean consumption length')
        ax.set_xlabel("session")
        ax.set_ylabel("consumption time(s)")
        x = np.arange(1, len(mice[i].mean_consumption_length) + 1)
        background_patches(curr_default, mice[i], ax, x)
        plt.savefig(f'{mice[i].name} mean consumption lengths.svg')
        plt.close()

        fig, ax = plt.subplots(figsize=(10, 5))
        plt.plot(mice[i].mean_consumption_licks)
        ax.set_title(f'{mice[i].name} mean consumption licks')
        ax.set_xlabel("session")
        ax.set_ylabel("consumption licks")
        x = np.arange(1, len(mice[i].mean_consumption_licks) + 1)
        background_patches(curr_default, mice[i], ax, x)
        plt.savefig(f'{mice[i].name} mean consumption licks.svg')
        plt.close()

        fig, ax = plt.subplots(figsize=(10, 5))
        plt.plot(mice[i].mean_session_reward_rate)
        ax.set_title(f'{mice[i].name} mean reward rate')
        ax.set_xlabel("session")
        ax.set_ylabel("reward rate (ul/s)")
        x = np.arange(1, len(mice[i].mean_consumption_licks) + 1)
        background_patches(curr_default, mice[i], ax, x)
        plt.savefig(f'{mice[i].name} mean session reward rate.svg')
        plt.close()

        fig, ax = plt.subplots(figsize=(10, 5))
        plt.plot(mice[i].session_adjusted_optimal)
        ax.set_title(f'{mice[i].name} session adjusted optimal times')
        ax.set_xlabel("session")
        ax.set_ylabel("time(s)")
        x = np.arange(1, len(mice[i].session_adjusted_optimal) + 1)
        background_patches(curr_default, mice[i], ax, x)
        plt.savefig(f'{mice[i].name} session adjusted optimal.svg')
        plt.close()


        #---------------------background length from consumption-----------------------#
        plotPairingLists(mice[i].mean_background_length_from_consumption_s,
                         mice[i].mean_background_length_from_consumption_l, curr_default, 'time')
        ax.set_title(f'{mice[i].name} background repeat from consumption bout')
        if saving:
            plt.savefig(f'{mice[i].name} mean_background_length_from_consumption.svg')
        plt.close()
        #---------------------background licks from consumption-----------------------#
        plotPairingLists(mice[i].mean_background_lick_from_consumption_s,
                         mice[i].mean_background_lick_from_consumption_l, curr_default, 'count')
        ax.set_title(f'{mice[i].name} mean number of licks from previous consumption bout')
        if saving:
            plt.savefig(f'{mice[i].name} mean_background_lick_from_consumption.svg')
        plt.close()


        #---------------------percentage lick bout from consumption extending into bg-----------------------#
        plotPairingLists(mice[i].perc_bout_into_background_s,
                         mice[i].perc_bout_into_background_l, curr_default, 'perc')
        ax.set_title(f'{mice[i].name} percentage of consumption bout that cross into next background')
        if saving:
            plt.savefig(f'{mice[i].name} perc_bout_into_background.svg')
        plt.close()

        # ---------------------percentage rewarded-----------------------#
        plotPairingLists(mice[i].perc_rewarded_s,
                         mice[i].perc_rewarded_l, curr_default, 'perc')
        ax.set_title(f'{mice[i].name} percent trial rewarded')
        if saving:
            plt.savefig(f'{mice[i].name} perc trial rewarded.svg')
        plt.close()
        # ---------------------implusive lick perc-----------------------#
        plotPairingLists(mice[i].reflex_lick_perc_s,
                         mice[i].reflex_lick_perc_l, curr_default, 'perc')
        ax.set_title(f'{mice[i].name} impulsive licks perc')
        if saving:
            plt.savefig(f'{mice[i].name}_impulsive_lick_perc.svg')
        plt.close()
        # ---------------------background length-----------------------#
        plotPairingLists(mice[i].mean_background_length_s,
                         mice[i].mean_background_length_l, curr_default, 'time')
        ax.set_title(f'{mice[i].name} mean background length')
        if saving:
            plt.savefig(f'{mice[i].name} mean background length.svg')
        plt.close()
        # ---------------------prob (reward) at licking-----------------------#
        plotPairingLists(mice[i].lick_prob_s,
                         mice[i].lick_prob_l, curr_default, 'perc')
        ax.set_title(f'{mice[i].name} prob at licking')
        if saving:
            plt.savefig(f'{mice[i].name}_prob_at_licking.svg')
        plt.close()
        # ---------------------moving averages-----------------------#
        fig, ax = plt.subplots(figsize=(30, 5))
        # print(f's sessions lick index {(mice[i].all_holding_s_index)}')
        if curr_default == "long":
            mice[i].all_holding_s_index = [x + mice[i].all_holding_l_index[-1] for x in mice[i].all_holding_s_index]
            moving_averages = mice[i].moving_average_s + mice[i].moving_average_l
        if curr_default == "short":
            mice[i].all_holding_l_index = [x + mice[i].all_holding_s_index[-1] for x in mice[i].all_holding_l_index]
            moving_averages = mice[i].moving_average_l + mice[i].moving_average_s
        plt.plot(moving_averages)
        for j in range(len(mice[i].all_holding_s_index)):
            plt.axvline(x=mice[i].all_holding_s_index[j], color='b')
        for j in range(len(mice[i].all_holding_l_index)):
            plt.axvline(x=mice[i].all_holding_l_index[j], color='r')
        ax.set_title(f'{mice[i].name} moving avg performance')
        ax.set_xlabel("trial")
        ax.set_ylabel("holding time(s)")
        ax.legend(['short', 'long'])
        plt.axis([0, 18000, 0, 30])
        if saving:
            plt.savefig(f'{mice[i].name}_moving_avg_perf.svg')
        plt.close()
       #----------------------scatter for session optimal vs adjusted actual optimal --------------------------#
        if mice[i].default == 'short':
            cmap = plt.get_cmap('Blues')
            adjusted_optimal_default = mice[i].session_adjusted_optimal[:mice[i].default_session_num]
            optimal_default = [optimal_wait[0]] * len(adjusted_optimal_default)
        elif mice[i].default == 'long':
            cmap = plt.get_cmap('Reds')
            adjusted_optimal_default = mice[i].session_adjusted_optimal[:mice[i].default_session_num]
            optimal_default = [optimal_wait[1]] * len(adjusted_optimal_default)
        # Normalize the colors based on the range of values
      #  print(adjusted_optimal_default)
        colors = np.arange(len(adjusted_optimal_default))
        #print(len(colors))

        norm = plt.Normalize(min(colors), max(colors))
        # Create the scatter plot with a gradient of colors
        plt.scatter(adjusted_optimal_default, optimal_default, c=colors, cmap=cmap, norm=norm, edgecolors='black')
        # Set labels and title
        plt.xlabel('adjusted optimal')
        plt.ylabel('optimal')
        if saving:
            plt.savefig(f'{mice[i].name} color optimal vs adjusted.svg')
        plt.close()

        # ----------------------scatter for session average vs adjusted actual optimal --------------------------#
        if mice[i].default == 'short':
            cmap = plt.get_cmap('Blues')
            adjusted_optimal_default = mice[i].session_adjusted_optimal[:mice[i].default_session_num]
            session_avg = mice[i].holding_s_mean[:mice[i].default_session_num]
        elif mice[i].default == 'long':
            cmap = plt.get_cmap('Reds')
            adjusted_optimal_default = mice[i].session_adjusted_optimal[:mice[i].default_session_num]
            session_avg = mice[i].holding_l_mean[:mice[i].default_session_num]
        # Normalize the colors based on the range of values
        #  print(adjusted_optimal_default)
        colors = np.arange(len(adjusted_optimal_default))

        norm = plt.Normalize(min(colors), max(colors))
        # Create the scatter plot with a gradient of colors
        plt.scatter(adjusted_optimal_default, session_avg, c=colors, cmap=cmap, norm=norm, edgecolors='black')

        x_d = np.linspace(0, 10, 1000)  # Adjust range as needed
        y_d = x_d  # Diagonal line where y equals x

        # Plotting the diagonal line
        plt.plot(x_d, y_d, label='y = x')
        plt.gca().set_aspect('equal', adjustable='box')

        plt.xticks(np.arange(0, 8, 1))  # Adjust range and interval as needed
        plt.yticks(np.arange(0, 8, 1))  # Adjust range and interval as needed
        # Set labels and title
        plt.xlabel('adjusted optimal')
        plt.ylabel('session avg')
        if saving:
            plt.savefig(f'{mice[i].name} color mean vs adjusted.svg')
        plt.close()

        # ---------------------moving averages for block perf-----------------------#
        if has_block:
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

def violins(mice, task_params, has_block, saving):
    if has_block:
        path, _ = utils.set_plotting_path(has_block, task_params)
        os.chdir(path)
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
    else:
        path, _ = utils.set_plotting_path(has_block, task_params)
        os.chdir(path)

    print(f'plotting and saving in {path}')

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

    # violin_stable_times = []
    # labels = []
    # for i in range(len(mice)):
    #     violin_stable_times.append(mice[i].stable_s)
    #     print(len(mice[i].stable_s))
    #     violin_stable_times.append(mice[i].stable_l)
    #     # violin_stable_times.append(mice[i].moving_average_s)
    #     # violin_stable_times.append(mice[i].moving_average_l)
    #     labels.append(f'{mice[i].name} s')
    #     labels.append(f'{mice[i].name} l')

    # fig = plt.figure(facecolor=(1, 1, 1))
    # ax = fig.add_axes([0, 0, 1, 1])
    # ax.set_xlabel('trial type (stable)')
    # ax.set_ylabel('holding time (s)')
    # # Create the boxplot
    # bp = ax.violinplot(violin_stable_times, showmeans=True)
    # for i, pc in enumerate(bp["bodies"], 1):
    #     if i % 2 != 0:
    #         pc.set_facecolor('blue')
    #     else:
    #         pc.set_facecolor('red')
    #
    # set_axis_style(ax, labels)
    # if saving:
    #     plt.savefig(f'violin_stable_time.svg', bbox_inches='tight')
    # plt.close()

    # violin_moving_var = []
    # labels = []
    # for i in range(len(mice)):
    #     violin_moving_var.append(mice[i].moving_average_s_var)
    #     violin_moving_var.append(mice[i].moving_average_l_var)
    #     labels.append(f'{mice[i].name} s')
    #     labels.append(f'{mice[i].name} l')
    #
    # fig = plt.figure(facecolor=(1, 1, 1))
    # # Create an axes instance
    # ax = fig.add_axes([0, 0, 1, 1])
    # ax.set_xlabel('trial type')
    # ax.set_ylabel('variance')
    # # Create the boxplot
    # bp = ax.violinplot(violin_moving_var, showmeans=True)
    # for i, pc in enumerate(bp["bodies"], 1):
    #     if i % 2 != 0:
    #         pc.set_facecolor('blue')
    #     else:
    #         pc.set_facecolor('red')
    #
    # set_axis_style(ax, labels)
    # if saving:
    #     plt.savefig(f'violin_moving_var.svg', bbox_inches='tight')
    # plt.close()

# plots about a particular session index (usually the last to see current stats)
def plotSession(mice, session, task_params, has_block, saving):
    path, user = utils.set_plotting_path(has_block, task_params)
    print(f'plotting and saving in {path}')
    violin_vars = []
    labels = []
    # session reward rate
    for i in range(len(mice)):
        curr_animal_path = path + '\\' + mice[i].name
        if user == "ziyi":
            curr_animal_path = path + '\\' + mice[i].name
        else:
            curr_animal_path = path + '/' + mice[i].name
        os.chdir(curr_animal_path)
        #file_path = curr_animal_path + '\\' + os.listdir()[0]

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
        if has_block:
            fig = plt.figure(facecolor=(1, 1, 1))
            ax = fig.add_axes([0, 0, 1, 1])
            ax.set_xlabel('blocks within session')
            ax.set_ylabel('wait time')
            all_licks = session_to_plot.session_holding_times
            # print(len(all_licks))
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
            # print(session_to_plot.file_path)
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

