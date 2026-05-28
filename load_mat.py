import scipy.io as sci
import numpy as np

data = sci.loadmat('data/patient01/trial01.mat')

print(data.keys())