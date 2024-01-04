import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

class num():
    def __init__(self,nx=11,nt=10001,L=0.5,T=10):

        self.L = L
        self.T = T

        self.x = np.linspace(0,L,nx)
        self.t = np.linspace(0,T,nt)

        self.dx = self.x[1]-self.x[0]
        self.dt = self.t[1]-self.t[0]

        #Constants
        self.u = 0.1
        self.k = 0.5
        self.n = 1

        self.path_save = os.getcwd()+r"/1D_unsteady/Num"

    def pde(self):

        C = np.zeros(shape=(len(self.t),len(self.x)))

        #Initial Condition
        C[0,:] = 0.0

        #Boundary Condition
        C[:,0] = 1.0

        for i in range(len(self.t)-1):
            C_new = C.copy()
            C_new[1:,1:] = C[:-1,1:] + self.dt*(-self.u*(C[:-1,1:]-C[:-1,:-1])/self.dx - self.k*C[:-1,1:]**self.n)
            C = C_new
        print(C.shape)
        return C


    def plot(self,t,x,C):

        fig,ax = plt.subplots()
        ax.set_xlabel("Location (x)")
        ax.set_ylabel("Time (s)")
        
        def animate(i):
            ax.clear()
            ax.plot(x,C[i])
            ax.set_title('%03d' % (i))
            ax.set_xlabel("Location (x)")
            ax.set_ylabel("Concentration")
            ax.set_ylim([0.0,1.2])

        ani = animation.FuncAnimation(fig, animate,frames=len(t), interval=10,blit=False,repeat = True)
        #ani.save(r'C:\Users\zkat4\Videos\pfr.gif', writer='pillow')
        plt.show()

    def save(self,t,x,C):
        os.chdir(self.path_save)
        np.save("t.npy", t)
        np.save("x.npy", x)
        np.save("C.npy", C)

    def load(self):
        os.chdir(self.path_save)
        t1 = np.load("t.npy")
        x1 = np.load("x.npy")
        C1 = np.load("C.npy")
        return (t1, x1, C1)

if __name__=="__main__":

    m = num(nx=101,nt=1001)
    # C=m.pde()
    # m.save(m.t,m.x,C)
    t,x,C = m.load()
    m.plot(t[::10],x,C[::10,:])
    





