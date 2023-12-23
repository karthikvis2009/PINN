import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from timeit import default_timer as timer
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import progressbar


print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


############################## PINN class ##############################

class PINN(Model):

    ########## Init class to subclass keras Model and initialize the architecture and data ##########
    
    def __init__(self, nhl, npl, act):

        super(PINN, self).__init__()

        ###################### Build the network architecture ####################
        self.nhl = nhl
        self.Mod = Sequential()
        for i in range(nhl):
            self.Mod.add(Dense(npl,activation=act,kernel_initializer=tf.keras.initializers.GlorotUniform(seed=i)))
        
        self.Mod.add(Dense(1,kernel_initializer=tf.keras.initializers.GlorotUniform(seed=nhl+1)))   # Temperature as output
        self.Mod.build([None,6])    # r, theta, t, k, rho, Cp as inputs
        print(self.Mod.summary())

        ############################################################

        actDict = {tf.nn.tanh: "tanh", tf.nn.relu: "relu", tf.nn.sigmoid: "sigmoid", tf.nn.elu: "elu"}
        self.save_ext = f"{nhl}_{npl}_{actDict[act]}_heat"

        ##############################Training scheme ##############################

        self.batch_size = 32

        lr=tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-3,decay_rate=0.09,decay_steps=100)
        self.train_op1 = tf.keras.optimizers.Adam(learning_rate=lr)

        #################### Paths to load/save ####################

        self.path_wts = os.getcwd()+r"/2D_unsteady/NN/wts"
        self.path_data = os.getcwd()+ r"/2D_unsteady/NN/Training_data"
        self.path_save = os.getcwd()+ r"/2D_unsteady/NN/graphs"

        ##########  Load all the training data (IC,BC_r1-2,BC_th1-2,Collocation points,Validation points)   ##########

        os.chdir(self.path_data)
        self.ip_ic,self.ip_bc_r1,self.ip_bc_r2,self.ip_bc_th1,self.ip_bc_th2,self.ip_coll,self.ip_val = \
            np.load('ip_ic.npy').astype(np.float32),np.load('ip_bc_r1.npy').astype(np.float32),\
                np.load('ip_bc_r2.npy').astype(np.float32),np.load('ip_bc_th1.npy').astype(np.float32),\
                    np.load('ip_bc_th2.npy').astype(np.float32),np.load('ip_coll.npy').astype(np.float32),np.load('ip_val.npy').astype(np.float32)

        self.T_ic,self.T_bc_r1,self.T_coll,self.T_val = np.load('T_ic.npy'),np.load('T_bc_r1.npy'),np.load('T_coll.npy'),np.load('T_val.npy')

    #################### Forward pass ####################

    def call(self, inputs):
        T = self.Mod(inputs)
        return T
    
    #################### Loss function for training data ####################

    def loss_d(self,mb_coll):
        t,r,th,k,rho,Cp,T_d = tf.split(tf.convert_to_tensor(mb_coll,dtype=tf.float32),num_or_size_splits=6,axis=1)
        with tf.GradientTape() as tape:
            tape.watch((t,r,th,k,rho,Cp))
            T = self.call(tf.concat([t,r,th,k,rho,Cp],axis=1))
        L = tf.reduce_mean((T-T_d)**2)
        return L
    
    #################### Loss function for initial condition ####################

    def loss_ic(self):
        t,r,th,k,rho,Cp = tf.split(tf.convert_to_tensor(self.ip_ic,dtype=tf.float32),num_or_size_splits=6,axis=1)
        with tf.GradientTape() as tape:
            tape.watch((t,r,th,k,rho,Cp))
            T = self.call(tf.concat([t,r,th,k,rho,Cp],axis=1))
        L = tf.reduce_mean((T-self.T_ic)**2)
        return L
    
    #################### Loss function for boundary conditions ####################

    def loss_bc_th1(self):
        t,r,th,k,rho,Cp = tf.split(tf.convert_to_tensor(self.ip_bc_th1,dtype=tf.float32),num_or_size_splits=6,axis=1)
        with tf.GradientTape() as tape:
            tape.watch((t,r,th,k,rho,Cp))
            T = self.call(tf.concat([t,r,th,k,rho,Cp],axis=1))
        dTdth = tape.gradient(T,th)
        L = tf.reduce_mean((dTdth)**2)
        return L
    
    def loss_bc_th2(self):
        t,r,th,k,rho,Cp = tf.split(tf.convert_to_tensor(self.ip_bc_th2,dtype=tf.float32),num_or_size_splits=6,axis=1)
        with tf.GradientTape() as tape:
            tape.watch((t,r,th,k,rho,Cp))
            T = self.call(tf.concat([t,r,th,k,rho,Cp],axis=1))
        dTdth = tape.gradient(T,th)
        L = tf.reduce_mean((dTdth)**2)
        return L
    
    def loss_bc_r1(self):
        t,r,th,k,rho,Cp = tf.split(tf.convert_to_tensor(self.ip_bc_r1,dtype=tf.float32),num_or_size_splits=6,axis=1)
        with tf.GradientTape() as tape:
            tape.watch((t,r,th,k,rho,Cp))
            T = self.call(tf.concat([t,r,th,k,rho,Cp],axis=1))
        L = tf.reduce_mean((T-self.T_bc_r1)**2)
        return L
    
    def loss_bc_r2(self):
        t,r,th,k,rho,Cp = tf.split(tf.convert_to_tensor(self.ip_bc_r2,dtype=tf.float32),num_or_size_splits=6,axis=1)
        with tf.GradientTape() as tape:
            tape.watch((t,r,th,k,rho,Cp))
            T = self.call(tf.concat([t,r,th,k,rho,Cp],axis=1))
        dTdr = tape.gradient(T,r)
        #BC r @ outer boundary : -kdT/dr = h(T-T_a)
        h = 70.0
        Ta = 25.0
        L = tf.reduce_mean((k*dTdr+h*(T-Ta))**2)
        return L

    #################### Loss function for validation ####################

    def loss_v(self,mb_val):
        t,r,th,k,rho,Cp,T_val = tf.split(tf.convert_to_tensor(mb_val,dtype=tf.float32),num_or_size_splits=7,axis=1)
        T = self.call(tf.concat([t,r,th,k,rho,Cp],axis=1))
        L = tf.reduce_mean((T-T_val)**2)
        return L

    #################### Loss function for PDE ####################

    def loss_PDE(self, mb_coll):
        t,r,th,k,rho,Cp,_ = tf.split(tf.convert_to_tensor(mb_coll,dtype=tf.float32),num_or_size_splits=7,axis=1)
        with tf.GradientTape(persistent=True) as tape:
            tape.watch((t,r,th,k,rho,Cp))
            T = self.call(tf.concat([t,r,th,k,rho,Cp],axis=1))
            Tr = tape.gradient(T,r)     # Gradient computed within the loop for computing second order derivative
            Tth = tape.gradient(T,th)
        Tt = tape.gradient(T,t)
        T2r = tape.gradient(Tr,r)
        T2Th = tape.gradient(Tth,th)
        L = tf.reduce_mean((Tt-(k/(rho*Cp))*((1/r)*(r*T2r+Tr)+(1/r**2)*T2Th))**2)   # Residual of the transient conduction equation
        return L
    
    #################### grad function to compute gradients for optimizer ####################

    def grad(self, mb_coll,mb_val):
        with tf.GradientTape() as grad:
            grad.watch((self.trainable_variables))
            L_ic,L_bc = self.loss_ic(),self.loss_bc_r1() + self.loss_bc_r2() + self.loss_bc_th1() + self.loss_bc_th2()
            L_pde = self.loss_PDE(mb_coll)
            L_total = L_ic+L_bc+L_pde
        L_val = self.loss_v(mb_val)
        g = grad.gradient(L_total, self.trainable_variables)

        return g, L_total, L_val

    def get_data(self, N):

        # Collocation points
        # r_c_t = np.random.choice(np.arange(0, len(self.tc)), N, replace=False)
        # r_c_r = np.random.choice(np.arange(0, len(self.xc)), N, replace=False)
        # r_c_th = np.random.choice(np.arange(0, len(self.yc)), N, replace=False)

        r_coll = np.random.choice(np.arange(0,len(self.ip_coll)),N,replace=False)
        r_val = np.random.choice(np.arange(0,len(self.ip_val)),N,replace=False)

        mb_coll = tf.concat([self.ip_coll[r_coll,:],self.T_coll[r_coll,:]],axis=1)
        mb_val = tf.concat([self.ip_coll[r_val,:],self.T_val[r_val,:]],axis=1)

        return mb_coll,mb_val

    def train(self, max_epochs, pretrain=True):
        epch, loss_t, loss_val = [], [], []
        i = 0
        os.chdir(self.path_wts)

        if pretrain:
            self.load_weights(f"weight_f_{self.save_ext}")
            print("Weights have been loaded!")

        else:
            print("Training with Xavier initialization!")

        _,mb_val = self.get_data(self.batch_size)

        L = self.loss_v(mb_val)

        widgets = ['| ', progressbar.Timer(),' | ',progressbar.Percentage(), ' ', progressbar.GranularBar(), ' ', progressbar.Counter(format='%(value)d/%(max_value)d'),' ',' | ',progressbar.ETA(),\
                   ' | ',progressbar.FormatLabel(""), progressbar.FormatLabel("")]

                   # ' ', progressbar.FormatLabel('Lp: %0.4f, Lv : %0.4f '%(L1,L2))]
        self.bar = progressbar.ProgressBar(max_value=max_epochs, widgets=widgets, term_width=150).start()
        while i < max_epochs and L > 0.00001:
            mb_coll,mb_val = self.get_data(self.batch_size)

            g, Lt, L = self.grad(mb_coll,mb_val)
            self.train_op1.apply_gradients(zip(g, self.trainable_variables))
            epch.append(i)
            loss_t.append(Lt)
            loss_val.append(L)

            widgets[-2],widgets[-1] = progressbar.widgets.FormatLabel(" Lt : {0:.6f}".format(Lt)),\
                                                   progressbar.widgets.FormatLabel(", Lv : {0:.4f} |".format(L))
            self.bar.update(i + 1)
            i += 1

        self.bar.finish()

        os.chdir(self.path_save)
        np.save(f"L_{self.save_ext}.npy", np.vstack((loss_t, loss_val)))
        os.chdir(self.path_wts)
        self.save_weights(f"weight_f_{self.save_ext}")
        #self.loss_plot()

        return epch, (loss_t, loss_val)

    def predict(self, inputs):
        os.chdir(path_wts)
        self.load_weights(f"weight_f_{self.save_ext}").expect_partial()
        op = self.call(inputs)
        return (op)

    def err_pred(self, c, cp):

        ec = np.linalg.norm(c - cp, 2) / np.linalg.norm(cp, 2)
        os.chdir(path_save)
        np.save("error_" + self.save_ext + ".npy", (ec))


    def loss_plot(self):
        os.chdir(path_save)
        label = ["loss_p", "loss_val"]
        L = np.load(f"L_{self.save_ext}.npy")
        Epch = np.linspace(0, len(L[0]), len(L[0]))
        # plt.plot(Epch, L[0] + L[1], label="loss_t")
        for l in range(0, len(label)):
            plt.plot(Epch, L[l], label=label[l])
        plt.xlabel("Iterations")
        plt.ylabel(r"Loss (Mean squared)")
        plt.legend(loc=r"upper right")
        plt.title("Plot of Losses during training")
        plt.savefig(r"loss_plot" + self.save_ext + ".png")
        plt.show()

    def plot_contour(self, x, y, C1,C2):
        fig, [ax1,ax2] = plt.subplots(2,1,sharex=True)
        im1 = ax1.imshow(C1.T, extent=[x.min(), x.max(), y.min(), y.max()], aspect='auto', cmap='viridis',origin='lower')
        im2 = ax2.imshow(C2.T, extent=[x.min(), x.max(), y.min(), y.max()], aspect='auto', cmap='viridis',origin='lower')
        ax1.set_ylabel('Y')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax1.set_title("Data")
        ax2.set_title("Predicted")
        cbar = fig.colorbar(im1, ax=[ax1,ax2], orientation='vertical')
        cbar.set_label('Concentration')
        plt.suptitle("Predicted concentration plot for the last timestep")
        plt.show()

    def plot_anim(self, t, x, y, c_data, c_pred):
            ## Plotting both videos simultaneously
        fig, [ax1,ax2] = plt.subplots(2,1,sharex = True)

        im1 = ax1.imshow(c_data[0].T, extent=[x.min(), x.max(), y.min(), y.max()], aspect='auto', cmap='viridis',origin='lower')
        im2 = ax2.imshow(c_pred[0].T, extent=[x.min(), x.max(), y.min(), y.max()], aspect='auto', cmap='viridis',origin='lower')
        ax1.set_ylabel("Y")
        ax2.set_ylabel("Y")
        ax2.set_xlabel("X")
        ax1.set_title("Data")
        ax2.set_title("Predicted")
        title = plt.suptitle("Time : ")
        def animate(i):
            im1.set_array(c_data[i,:,:].T)
            im2.set_array(c_pred[i,:,:].T)
            title.set_text('Time : %02d s' % t[i])
        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes([0.88, 0.15, 0.04, 0.7])
        cbar= fig.colorbar(im1, cax=cbar_ax, orientation = 'vertical')
        cbar.set_label('Concentration')
        
        ani = animation.FuncAnimation(fig, animate, len(t), interval=10, blit=False)
        plt.show()




# Training the model
if __name__ == "__main__":

    model = PINN(8, 20, tf.nn.relu)
    start = timer()
    epch, L = model.train(max_epochs=10, pretrain=False)
    stop = timer()
    print(f'\nTime taken for training is {stop - start} s')
 
    print("Success!")


