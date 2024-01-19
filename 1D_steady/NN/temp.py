import numpy as np

u = np.round(np.arange(0.01,0.21,0.01),2)
print(len(u))

for i in range(10):
    r = np.random.randint(0,len(u))
    print(u[r])