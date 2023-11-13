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