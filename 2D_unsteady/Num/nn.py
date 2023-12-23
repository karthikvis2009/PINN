import os
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import progressbar

import warnings
warnings.filterwarnings("error",category=RuntimeWarning)

#Radial heat transfer

class ht_rad:
    def __init__(self,vals,material = 'porcelain'):

        #Properties
        self.k = vals['k']
        self.rho = vals['rho']
        self.Cp = vals['Cp']
        self.h = vals['h']

        #Mesh
        self.r = np.linspace(vals['ri'],vals['ro'],vals['nr'])
        self.th = np.linspace(0,np.pi/2,vals['nth'])
        self.Th,self.R = np.meshgrid(self.th,self.r)
        self.dr = self.r[1]-self.r[0]
        self.dth = self.th[1]-self.th[0]

        #BC
        self.T_i = vals['T_i']
        self.T_a = vals['T_a']

        #Time conditions
        self.t = np.linspace(0,vals['End time'],vals['timesteps'])
        self.dt = self.t[1]-self.t[0]
        self.save_int = vals['save_interval']

        #Ext for multiple files
        self.save_ext = vals['mat']+'_'

    def plot_mesh(self):
        # plt.scatter(self.R*np.cos(self.Th),self.R*np.sin(self.Th))
        fig,ax = plt.subplots(subplot_kw={'projection': 'polar'})
        ax.scatter(self.Th,self.R)
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_frame_on(False)
        plt.show()
    
    def diff2(self,f):
        dr2,dth2 = np.zeros_like(f),np.zeros_like(f)
        dr2[1:-1,1:-1] = (f[2:,1:-1]-2*f[1:-1,1:-1]+f[:-2,1:-1])/self.dr**2
        dth2[1:-1,1:-1] = (f[1:-1,2:]-2*f[1:-1,1:-1]+f[1:-1,:-2])/self.dth**2
        return(dr2,dth2)
    
    def diff1(self,f):
        d1 = np.zeros_like(f)
        d1[1:-1,1:-1] = (f[2:,1:-1]-f[:-2,1:-1])/self.dr
        return d1

    def solver(self):

        #dT/dt = k/rhoCp * ((1/r)*d/dr(r*dT/dr) + (1/r2)*d2T/dth2)
        #BC r @ outer boundary : -kdT/dr = h(T-T_a)

        #IC
        T = 25*np.ones_like(self.R)

        #BC
        T[0,:] = self.T_i
        T[-1,:] = (1/(self.h + self.k/self.dr))*(self.k*T[-2,:]/self.dr + self.h*self.T_a)
        T[:,0] = T[:,1]
        T[:,-1] = T[:,-2]

        yield 0,T

        widgets = ['| ', progressbar.Timer(), ' | ', progressbar.Percentage(), ' ', progressbar.GranularBar(), ' ', progressbar.Counter(format='%(value)d/%(max_value)d'), ' ', ' | ', progressbar.ETA(), ' | ',progressbar.FormatLabel(""), ' | ']
        bar = progressbar.ProgressBar(max_value=len(self.t), widgets=widgets, term_width=150).start()
        sol_T=[]

		#PDE
        for t_iter in range(len(self.t)):
            T_new = T.copy()

            dTdr = self.diff1(T_new)
            d2Tdr2,d2Tdth2 = self.diff2(T_new)

            T[1:-1,1:-1] = T_new[1:-1,1:-1]+self.k/(self.rho*self.Cp)*self.dt*\
                ((1/self.R[1:-1,1:-1])*(self.R[1:-1,1:-1]*d2Tdr2[1:-1,1:-1]+dTdr[1:-1,1:-1]) +\
                 (1/self.R[1:-1,1:-1]**2)*d2Tdth2[1:-1,1:-1])
            T[0,:] = self.T_i
            T[-1,:] = T[-2,:]
            T[:,0] = T[:,1]
            T[:,-1] = T[:,-2]
            widgets[-2] = progressbar.FormatLabel("Time : {0:2f} s".format(self.t[t_iter]))
            bar.update(t_iter+1)
            if t_iter%self.save_int == 0:
                sol_T.append(((t_iter+1)*self.dt,T.copy()))
            else:
                continue
        t_sol = np.array([sol[0] for sol in sol_T])
        T_sol = np.array([sol[1] for sol in sol_T])

        bar.finish()
        self.save(t_sol,self.r,self.th,T_sol)
    
    def save(self,t,r,th,T):
        np.save("t.npy", t)
        np.save("r.npy", r)
        np.save("th.npy", th)
        np.save(f"{self.save_ext}T.npy", T)

    def load(self):
        t1 = np.load("t.npy")
        r1 = np.load("r.npy")
        th1 = np.load("th.npy")
        T1 = np.load(f"{self.save_ext}T.npy")
        return (t1, r1, th1, T1)
    
    def plot(self):
        t,r,th,T = self.load()

        fig, ax = plt.subplots(figsize=(12, 12),subplot_kw={'projection':'polar'})
        ax.grid(False)
        ax.set_xlim((0,np.pi/2))
        # ax.set_xticks([])
        # ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_frame_on(False)

        th,r = np.meshgrid(th,r)
        #im = ax.imshow(T[0], extent=[th.min(), th.max(), r.min(), r.max()], aspect='auto', cmap='jet', origin='lower')
        c = ax.pcolormesh(th,r,T[0],cmap='jet')

        ax.set_title('Time : 0')

        def animate(i):
            #im.set_array(T[i])
            c.set_array(T[i])
            ax.set_title('Time : {0:1f}'.format(t[i]))

        cbar = fig.colorbar(c, ax=[ax], orientation='vertical')
        cbar.set_label('Temperature')
        ani = animation.FuncAnimation(fig, animate, len(t), interval=100, blit=False, repeat=True)
        plt.show()
    
    def plot_surf_avg_temp(self):
        t,r,th,T = self.load()

        fig, ax = plt.subplots(figsize=(12, 12))

        T_avg = [np.mean(T[i,-1,:]) for i in range(len(t))]
        ax.plot(t,T_avg)
        ax.set_title('Plot of temperature at outer boundary')
        ax.set_xlabel('Time')
        ax.set_ylabel('Temperature')
        plt.show()

if __name__=="__main__":

    def para(i):
        mat = ['al','cu','ss','br','cs']
        k = [205.0,398.0,16.0,109.0,50.0]
        rho = [2700.0,8960.0,8000.0,8530.0,7850.0]
        Cp = [900.0,390.0,500.0,380.0,490.0]
        vals={
        #Material
        'mat':mat[i],
        #Properties
        #'k':1.5,'rho':2400.0,'Cp':1050.0, Porcelain
        'k' : k[i], 'rho' : rho[i], 'Cp' : Cp[i], 'h' : 70.0,
        #Dim
        'ri':0.04,'ro':0.05,'nr':101,'nth':51,
        #BC
        'T_i':90.0, 'T_a':25.0,
        #Time conditions
        'End time':10.0,'timesteps':1000001,'save_interval':1000
        }

        return vals
    
    for i in range(5):
        print(f"{i+1} Material Started")
        v = para(i)
        h = ht_rad(v)
        for t,T in h.solver():
            pass
        h.plot()

        print(f"{i+1} Material Ended")

    # h = ht_rad(vals)
    # for t,T in h.solver():
    #     pass
    h.plot()
    h.plot_surf_avg_temp()

