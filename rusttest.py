import numpy as np
import time

arr = np.arange(0,640000000,1, dtype='float64').reshape(4000,-1)
t1=time.time()

res = np.sum(arr,axis=1)
t2=time.time()
res2 = np.einsum("ij->i", arr)
t3=time.time()
print(t2-t1)
print(t3-t2)