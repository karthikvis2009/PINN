#### Python code for steady-state diffusion

import numpy as np

class diff:
    def __init__(self,n):

        #Geometry and mesh
        x = np.linspace(0,0.1,n)
        y = np.linspace(0,0.1,n)
        self.Y,self.X = np.meshgrid(y,x)

    def solve(self):
        C = np.zeros_like(self.X)
        

