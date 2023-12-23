import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider,Button

class plot:
    def __init__(self):
        self.r = np.load('r.npy')
        self.th = np.load('th.npy')
        self.t = np.load('t.npy')
        T_al = np.load('al_T.npy')
        T_br = np.load('br_T.npy')
        T_cs = np.load('cs_T.npy')
        T_cu = np.load('cu_T.npy')
        T_ss = np.load('ss_T.npy')

        self.T = [T_al, T_br, T_cs, T_cu, T_ss]
        
        self.T_avg = [self.T_average(T) for T in self.T]
        self.labels = ['Al', 'Br', 'CS', 'Cu', 'SS']

        #self.plot_surf_avg_temp()
        

    def T_average(self,T):
        T_avg = [np.mean(T[i,-1,:]) for i in range(len(self.t))]
        return(T_avg)
    
    def plot_surf_avg_temp(self):
        fig,ax = plt.subplots(figsize=(12,12))
        for i in range(len(self.T_avg)):
            ax.plot(self.t,self.T_avg[i], label = self.labels[i])
        ax.legend(loc='center right')
        plt.show()

    def plot_contour(self):
        fig, ax = plt.subplots(2,3,figsize=(12, 12),subplot_kw={'projection':'polar'})
        c=[]
        th,r = np.meshgrid(self.th,self.r)


        for a in ax.flatten():
            a.grid(False)
            a.set_xlim((0,np.pi/2))
            a.set_xticklabels([])
            a.set_yticklabels([])
            a.set_frame_on(False)



        # for i in range(5):
        #     c.append(ax.flatten()[i].pcolormesh(th,r,self.T[i][0],cmap='jet'))
        #     ax.flatten()[i].set_title(self.labels[i])
        c.append(ax[0,0].pcolormesh(th,r,self.T[0][0],cmap='jet'))
        c.append(ax[0,1].pcolormesh(th,r,self.T[1][0],cmap='jet'))
        c.append(ax[0,2].pcolormesh(th,r,self.T[2][0],cmap='jet'))
        c.append(ax[1,0].pcolormesh(th,r,self.T[3][0],cmap='jet'))
        c.append(ax[1,1].pcolormesh(th,r,self.T[4][0],cmap='jet'))


        ax[0,0].set_title('Al')
        ax[0,1].set_title('Br')
        ax[0,2].set_title('CS')
        ax[1,0].set_title('Cu')
        ax[1,1].set_title('SS')

        title = plt.suptitle("Time : ")

        def animate(i):
            #im.set_array(T[i])
            for a in range(len(self.labels)):
                c[a].set_array(self.T[a][i])
            title.set_text('Time : {0:1f} s'.format(self.t[i]))

        cbar = fig.colorbar(c[-1], ax=[ax[0,2],ax[1,2]], orientation='vertical')

        cbar.set_label('Temperature')
        ani = animation.FuncAnimation(fig, animate, len(self.t), interval=100, blit=False, repeat=True)
        plt.show()

    def plot_slider(self):

        fig, ax = plt.subplots(2,3,figsize=(12, 12),subplot_kw={'projection':'polar'})
        plt.subplots_adjust(bottom=0.25)
        ax_time = plt.axes([0.2,0.1,0.65,0.1])
        time = Slider(ax_time,'Time',0,len(self.t)-1,self.t[0],valstep=1,handle_style={'size':'50'})
        c=[]
        th,r = np.meshgrid(self.th,self.r)


        for a in ax.flatten():
            a.grid(False)
            a.set_xlim((0,np.pi/2))
            a.set_xticklabels([])
            a.set_yticklabels([])
            a.set_frame_on(False)



        # for i in range(5):
        #     c.append(ax.flatten()[i].pcolormesh(th,r,self.T[i][0],cmap='jet'))
        #     ax.flatten()[i].set_title(self.labels[i])
        c.append(ax[0,0].pcolormesh(th,r,self.T[0][0],cmap='jet'))
        c.append(ax[0,1].pcolormesh(th,r,self.T[1][0],cmap='jet'))
        c.append(ax[0,2].pcolormesh(th,r,self.T[2][0],cmap='jet'))
        c.append(ax[1,0].pcolormesh(th,r,self.T[3][0],cmap='jet'))
        c.append(ax[1,1].pcolormesh(th,r,self.T[4][0],cmap='jet'))


        ax[0,0].set_title('Al')
        ax[0,1].set_title('Br')
        ax[0,2].set_title('CS')
        ax[1,0].set_title('Cu')
        ax[1,1].set_title('SS')

        title = plt.suptitle("Time : ")

        def update(val):
            ind = int(time.val)
            for a in range(len(self.labels)):
                c[a].set_array(self.T[a][ind])
            title.set_text('Time : {0:1f} s'.format(self.t[ind]))
            fig.canvas.draw_idle()


        time.on_changed(update)
        cbar = fig.colorbar(c[-1], ax=[ax[0,2],ax[1,2]], orientation='vertical')

        cbar.set_label('Temperature')
        plt.title('Temperature profiles for different materials')
        plt.show()


if __name__ == "__main__":
    p = plot()
    # p.plot_surf_avg_temp()
    # p.plot_contour()
    p.plot_slider()


