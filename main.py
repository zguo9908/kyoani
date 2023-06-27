# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from beh_draft import BehaviorAnalysis
import plots

def main():
    pass


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    beh = BehaviorAnalysis(task_type="regular")
    mice = beh.allAnimal()
    print(mice[0].name)
    plots.rawPlots(mice)
