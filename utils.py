import math

import numpy as np
from sympy.abc import x,y
from sympy import *

def getOptimalTime(m,p,bg):
    f = (1 - exp(-(1/m)*(x-bg)))*p
    k = np.linspace(3, 30, 1600) # range of x
    ml1 = dict()
    for i in k:
        ml1[i]= f.subs(x,i)/i
    optimal_time = max(ml1, key=ml1.get) -bg
    return optimal_time

def merge_lists_with_sources(list1, list2):
    # Merging lists and marking the source
    return [(x, 'List 1') if not math.isnan(x) else (y, 'List 2') for x, y in zip(list1, list2)]