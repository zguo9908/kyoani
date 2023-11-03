from behavior import BehaviorAnalysis
import plots


def main():
    pass


global task_type, has_block, task_params
task_type = "regular"
has_block = False
task_params = "old_params"

if task_params == "curr_params":
    optimal_wait = [1.74, 3.45]
elif task_params == "old_params":
    optimal_wait = [1.52, 2.92]

if __name__ == '__main__':
    beh = BehaviorAnalysis("exp1", optimal_wait, task_type=task_type, has_block=has_block, task_params=task_params)

    # mice = beh.allAnimal(["ZG023", "ZG024", "ZG025", "ZG026", "ZG027", "ZG028", "ZG029", "ZG020"])
    mice = beh.allAnimal(["ZG022", "ZG021", "ZG023", "ZG024", "ZG025", "ZG026", "ZG027", "ZG028", "ZG029", "ZG020"])

    plots.rawPlots(mice, task_params=task_params, has_block=has_block, saving=True)
    plots.violins(mice, task_params=task_params, has_block=has_block, saving=False)
    plots.plotSession(mice, -1, task_params=task_params, has_block=has_block, saving=True)

    if has_block:
        beh.testBlockDiff()
    else:
        beh.PlotCohortDiff()
        beh.PlotCohortSessionPDEDiff()

