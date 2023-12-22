#### Python code for steady-state Temp and reaction


import numpy as np
import matplotlib.pyplot as plt
import progressbar

class diff:
    def __init__(self,n):

        #Geometry and mesh
        self.x = np.linspace(0,0.1,n)
        self.y = np.linspace(0,0.1,n)
        self.dxy = self.x[1]-self.x[0]
        self.Y,self.X = np.meshgrid(self.y,self.x)
        u_max = 0.01
        self.mu = 0.001
        self.rho = 1000.0
        self.k = 0.1
        self.D = 6e-6

        self.u = np.broadcast_to(u_max*(self.y/self.y[-1])*(2.0-(self.y/self.y[-1])),(self.X.shape))

        fig,ax = plt.subplots(figsize=(12,12))
        ax.plot(self.y,self.u[int(len(self.x)/2),:])
        ax.set_xlabel('y')
        ax.set_ylabel(f'Velocity at x = {self.x[int(len(self.x)/2)]}')
        plt.show()

    def apply_C_BC(self,f):
        f[0,:] = 1.0
        f[-1,:] = f[-2,:]
        f[:,0] = f[:,1]
        f[:,-1] = f[:,-2]



    def solve(self,max_iterations = 10000, atol = 1e-10):
        C = np.zeros_like(self.X)
        T = np.zeros_like(self.X)
        self.apply_C_BC(C)

        

        ## udC/dx = -kC
        ## dC/dx = -kC/u 
        ## C = -kC/u * x + c1
        # dC = (dx/u)*( D*(d2C/dx2 + d2C/dy2)-kC )
        # C[i+1] = C[i] + (dx/u)*( D*(d2C/dx2 + d2C/dy2)-kC )

        widgets = ['| ', progressbar.Timer(), ' | ', progressbar.Percentage(), ' ', progressbar.GranularBar(), ' ', progressbar.Counter(format='%(value)d/%(max_value)d'), ' ', ' | ', progressbar.ETA(), ' | ',progressbar.FormatLabel(""), ' | ']
        bar = progressbar.ProgressBar(max_value=max_iterations, widgets=widgets, term_width=150).start()

        i = 0
        err = 1.0

        while i <max_iterations and err >= atol:
            C_new = C.copy()
            C[1:,1:-1] = C_new[:-1,1:-1] + (self.dxy/self.u[:-1,1:-1])*(-self.k*C_new[:-1,1:-1])
            self.apply_C_BC(C)

            err = np.max(np.abs(C-C_new))
            widgets[-2] = progressbar.FormatLabel("Err : {0:4f}".format(err))
            bar.update(i+1)
            i+=1
        bar.finish()
        #self.save(self.x,self.y,C)
        return(C)
    

        




    def plot(self,C):
        fig,[ax1,ax2] = plt.subplots(2)
        im1 = ax1.imshow(C.T,extent=[self.x.min(),self.x.max(),self.y.min(),self.y.max()],origin='lower',cmap='jet',aspect = "auto",interpolation='bicubic')
        im2 = ax2.imshow(self.u.T,extent=[self.x.min(),self.x.max(),self.y.min(),self.y.max()],origin='lower',cmap='jet',aspect = "auto",interpolation='bicubic')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        cbar1 = fig.colorbar(im1, ax=[ax1], orientation='vertical')
        cbar2 = fig.colorbar(im2, ax=[ax2], orientation='vertical')
        cbar1.set_label('Concentration')
        cbar2.set_label('Velocity')
        plt.title('Concentration Profile at Steady state')
        plt.show()

if __name__=="__main__":
    m = diff(101)
    C = m.solve()
    m.plot(C)
