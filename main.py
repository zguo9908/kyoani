import os

import utils
from behavior import BehaviorAnalysis
import plots
import pickle

def main():
    pass


global task_type, has_block, task_params
task_type = "regular"
has_block = False
task_params = "old_params"

global m1, p1, m2, p2, bg1, bg2
m1 = 1
m2 = 3
p1 = p2 = 0.9
bg1 = bg2 = 2

optimal_wait_short = utils.getOptimalTime(m1, p1, bg1)
print(f'optimal wait time for short timescape is {optimal_wait_short}')
optimal_wait_long = utils.getOptimalTime(m2, p2, bg2)
print(f'optimal wait time for long timescape is {optimal_wait_long}')
optimal_wait = [optimal_wait_short, optimal_wait_long]

params_dict = {"m1": m1, "p1": p1, "bg1": bg1, "m2": m2, "p2": p2, "bg2": bg2}
need_checkpoint = False

if __name__ == '__main__':
    beh = BehaviorAnalysis("exp1", optimal_wait, params_dict, task_type=task_type, has_block=has_block, task_params=task_params)
    if need_checkpoint:
        mice = beh.allAnimal(["ZG023", "ZG026", "ZG027", "ZG022", "ZG021", "ZG020", "ZG024", "ZG025", 'ZG028', 'ZG029'])
        utils.set_analysis_path(has_block, task_params)
        with open('mice_data.pkl', 'wb') as file:
            # Serialize and save the Python object to the file
            pickle.dump(mice, file)
    else:
        utils.set_analysis_path(has_block, task_params)
        print("loading previously saved checkpoint")
        with open('mice_data.pkl', 'rb') as file:
            mice = pickle.load(file)
            beh.mice = mice

    plots.run_all_single_animal_plot(mice, optimal_wait, task_params=task_params, has_block=has_block)
    if has_block:
        beh.testBlockDiff()
    else:
           # beh.PlotCohortDiff(default_only=True)
        beh.PlotCohortDiff(False, 15)
        beh.PlotCohortSessionPDEDiff()

