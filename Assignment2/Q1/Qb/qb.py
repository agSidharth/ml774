# Importing libraries...
import math
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import re

training_dir = str(sys.argv[1])
testing_dir  = str(sys.argv[2])

totalPos = len(os.listdir(testing_dir + "/pos"))
totalNeg = len(os.listdir(testing_dir + "/neg"))
totalDoc = totalPos + totalNeg

print("For b.i the accuracy will be : 0.5")
print("For b.ii the accuracy will be : "+str(totalPos/totalDoc))