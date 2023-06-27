import os

from animal import Animal
from session import Session

class BehaviorAnalysis:

    def __init__(self, task_type):
        self.task_type = task_type
        self.path = os.path.normpath(r'D:\behavior_data\curr_params')
        os.chdir(self.path)
        self.animal_list = os.listdir()
        self.animal_num = len(self.animal_list)
        self.mice = [] # this stores the animal object


    def allAnimal(self):
        for i in range(self.animal_num):
            animal = self.animal_list[i]
            curr_animal = Animal(animal)
            self.mice.append(curr_animal)
            curr_path = self.path + "\\" + animal
            os.chdir(curr_path)
            session_list = os.listdir()
            # filter all the items that are regular
            sessions = [session for session in session_list if self.task_type in session]
            curr_animal.sessions = sessions

            curr_animal.allSession(curr_path)
            print(f'processing all sessions for mice {animal}')

        return self.mice
