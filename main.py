# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from behavior import BehaviorAnalysis
import plots


def main():
    pass


global task_type, task_params
task_type = "regular"
task_params = "curr_params"

if task_params == "curr_params":
    optimal_wait = [1.74, 3.45]
elif task_params == "old_params":
    optimal_wait = [1.52, 2.93]

if __name__ == '__main__':
    beh = BehaviorAnalysis(optimal_wait, task_type=task_type, task_params=task_params)

    mice = beh.allAnimal()

    beh.testBlockDiff()
    plots.rawPlots(mice, task_params=task_params, saving=True)
    plots.violins(mice, task_params=task_params, saving=True)
