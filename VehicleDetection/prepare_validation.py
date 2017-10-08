import numpy as np
from sklearn.model_selection import train_test_split
import pickle

n_vehicle = 8792
n_nonvehicle = 8968

res = []

for n in [n_vehicle, n_nonvehicle]:
    train, test = train_test_split(list(range(n_vehicle)), test_size=0.3)
    res.append((train,test))

with open('validation_split.pickle', 'wb') as f:
    pickle.dump(res, f)

