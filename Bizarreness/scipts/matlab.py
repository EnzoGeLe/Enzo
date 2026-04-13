import numpy as np
import pandas as pd
from scipy.io import loadmat

def mat_to_py(x):
    if isinstance(x, np.ndarray):
        if x.dtype == 'O':
            if x.size == 1:
                return mat_to_py(x.flatten()[0])
            return [mat_to_py(i) for i in x.flatten()]
        return x.tolist()
    if isinstance(x, np.void):
        return {n: mat_to_py(x[n]) for n in x.dtype.names}
    if isinstance(x, bytes):
        return x.decode('utf-8', errors='ignore')
    return x

mat = loadmat(r'C:\Users\Enzob\Downloads\somnieve_dataset_geneva20260320.mat')
mat_key = next(k for k in mat if not k.startswith('__'))
shape = mat_to_py(mat[mat_key][0])

print(shape)
# access
arr_data = shape['arr']
df = pd.DataFrame(arr_data, columns=['wake_table_value'])
print(df)