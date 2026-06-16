import numpy as np
from sklearn.preprocessing import StandardScaler

arr = np.array([[1,2], [3,4], [5,6]])
print(arr)
scaler = StandardScaler()
scaled_arr = scaler.fit_transform(arr)
print(scaled_arr)