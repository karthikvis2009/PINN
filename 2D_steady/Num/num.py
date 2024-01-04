#### Python code for steady-state 2D mixing of hot and cold water in a T-bend


import numpy as np
import matplotlib.pyplot as plt
import progressbar
from mesh import msh
from ns_new import NS2D as ns

class T_junct:
    def __init__(self):
        
        m = msh()
        self.X,self.Y = m.X,m.Y
        self.inds = m.gen_mesh()    #ind_in_main,ind_in_side,ind_out,ind_wall
        
        mesh_dict = {'X':m.X,'Y':m.Y,'dx' : m.dx,'dy' : m.dy,'inds':(self.inds,m.ind)}

        # m.plot_mesh(self.inds)

        self.mu = 0.001
        self.rho = 1000.0
        self.k = 0.1
        self.D = 6e-6

        consts_dict = {'rho':self.rho,'mu':self.mu}
        bc_dict = {'u_in':0.001,'v_in':-0.001}

        n = ns(mesh_dict,consts_dict,bc_dict)
        u,v = n.solver()
        n.plot(u,v)
        print('Solved')
        #n.plot(m.x,m.y,u,v)


    def apply_C_BC(self,f):
        f[0,:] = 1.0
        f[-1,:] = f[-2,:]
        f[:,0] = f[:,1]
        f[:,-1] = f[:,-2]



    def solve(self,max_iterations = 10000, atol = 1e-10):
        C = np.zeros_like(self.X)
        T = np.zeros_like(self.X)
        self.apply_C_BC(C)

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

    te = T_junct()

#     C = m.solve()
#     m.plot(C)
