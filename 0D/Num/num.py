import os
import numpy as np
import matplotlib.pyplot as plt

class ZeroD:
    def __init__(self,T,nt):

        self.t = np.linspace(0,T,nt)
        self.dt = self.t[1]-self.t[0]

    def solver(self,max_iter = 10000,tol=1e-6):
        k1 = 1
        k2 = 0.9
        k3 = 0.3

        C = np.zeros(shape=(len(self.t),3))
        C[0,0] = 1.0
        C[0,1] = 0.5
        C[0,2] = 0.2
        i = 0
        err = 1
        while i < max_iter and err>tol:
            C_new = C.copy()
            C_new[1:,0] = C[:-1,0] + self.dt*(-k1*C[:-1,0]*C[:-1,1] + k2*C[:-1,1]*C[:-1,2] - k3*C[:-1,0]*C[:-1,2])
            C_new[1:,1] = C[:-1,1] + self.dt*(-k1*C[:-1,0]*C[:-1,1] - k2*C[:-1,1]*C[:-1,2] + k3*C[:-1,0]*C[:-1,2])
            C_new[1:,2] = C[:-1,2] + self.dt*(k1*C[:-1,0]*C[:-1,1] - k2*C[:-1,1]*C[:-1,2] - k3*C[:-1,0]*C[:-1,2])

            err = np.max(np.abs(C-C_new))
            C = C_new
        return(C)
    
    def plot(self,t,C):
        fig,ax = plt.subplots(figsize=(12,12))
        ax.plot(t,C[:,0],label='A')
        ax.plot(t,C[:,1],label='B')
        ax.plot(t,C[:,2],label='C')
        ax.set_xlabel('Time')
        ax.set_ylabel('Concentrations')
        ax.legend(loc='upper right')
        ax.set_xlim(t.min(),t.max())
        ax.set_ylim(0,1.0)
        plt.tight_layout()
        plt.show()

    def save(self,t,C):
        os.chdir(r"./0D/Num")
        np.save('t.npy',t)
        np.save('C.npy',C)

    def load(self):
        t1 = np.load('t.npy')
        C1 = np.load('C.npy')
        return(t1,C1)

if __name__=="__main__":
    z = ZeroD(10,1001)
    C = z.solver()
    z.save(z.t,C)
    t1,C1 = z.load()
    z.plot(t1,C1)