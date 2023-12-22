import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import InputLayer,Dense, Concatenate, RepeatVector
from tensorflow.keras.models import Model
from timeit import default_timer as timer
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import progressbar



print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

path_wts = r"/home/karvis/PINN/C_rxn_diff/NN/wts"
path_data = r"/home/karvis/PINN/C_rxn_diff/Numerical"
path_save = r"/home/karvis/PINN/C_rxn_diff/NN/results/graphs"

os.chdir(path_data)

#Collocation points data
c_data = np.load("C_PFR_num.npy").astype(dtype="float32")
t_data = np.load("t_PFR_num.npy").astype(dtype="float32")
x_data = np.load("x_PFR_num.npy").astype(dtype="float32")

xd1 = x_data[:,np.newaxis]
xd2 = np.broadcast_to(xd1.T, (len(t_data),len(x_data)))
x_train,x_test,t_train,t_test,c_train,c_test = train_test_split(xd2,t_data,c_data)
##Dense Network

class PINN(Model):
    def __init__(self, nhl, npl, act):

        #nhl : [nhl for x-model, nhl for t-model, nhl for shared model]
        super(PINN, self).__init__()

        self.nhl = nhl
        self.Mod = Sequential()
        for i in range(nhl):
            self.Mod.add(Dense(npl,activation=act,kernel_initializer=tf.keras.initializers.GlorotUniform(seed=i)))
        
        self.Mod.add(Dense(1,kernel_initializer=tf.keras.initializers.GlorotUniform(seed=nhl+1)))
        self.Mod.build([None,2])
        print(self.Mod.summary())
        actDict = {tf.nn.tanh: "tanh", tf.nn.relu: "relu", tf.nn.sigmoid: "sigmoid", tf.nn.elu: "elu"}

        self.save_ext = f"{nhl}_{npl}_{actDict[act]}_PFR_1D"

        self.batch_size = 32

        # lr=tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-2,decay_rate=0.09,decay_steps=100)

        self.L = 0.5
        self.T = 10.0

        self.n_t = 10
        self.n_x = 11
        self.train_op1 = tf.keras.optimizers.Adam(learning_rate=0.0001)

        # Initial condition data
        n_ic = 100
        self.t0 = tf.constant(tf.zeros(shape=(n_ic, 1)))
        self.x0 = tf.linspace(tf.constant([self.L/n_ic]),tf.constant([self.L]),n_ic)
        self.C0 = tf.zeros(shape=(n_ic, 1))

        # Boundary condition data
        n_bc = 100
        self.t_bc = tf.linspace(tf.constant([0.0]),tf.constant([self.T]),n_bc)
        self.x_bc = tf.zeros(shape=(n_bc, 1))
        self.C_bc = tf.ones(shape=(n_bc, 1))

        #Collocation points
        ncx,nct = 100,1000
        xc = np.linspace(0.0,self.L,ncx)
        tc = np.linspace(0.0,self.T,nct)
        tc,xc = np.meshgrid(tc,xc)
        self.t_c = tc.flatten()
        self.x_c = xc.flatten()

        #Training Data

        # Td,Xd = np.meshgrid(t_data[::1000],x_data)
        # self.t_d = tf.constant(Td.flatten().reshape(-1,1))
        # self.x_d = tf.constant(Xd.flatten().reshape(-1,1))
        # self.c_d = tf.constant(c_data[::1000,:].T.flatten().reshape(-1,1))

        #Validation data
        Tv,Xv = np.meshgrid(t_data[::10],x_data)
        self.t_val = tf.constant(Tv.flatten().reshape(-1,1))
        self.x_val = tf.constant(Xv.flatten().reshape(-1,1))
        self.C_val = tf.constant(c_data[::10,:].T.flatten().reshape(-1,1))

        tv = t_test
        xv = x_test

        tv = np.repeat(tv.reshape(-1,1),repeats=xv.shape[1])

        self.t_val = tv.flatten().reshape(-1,1)
        self.x_val = xv.flatten().reshape(-1,1)

        self.C_val = c_test.flatten().reshape(-1,1)

        # Constants for PFR with Monod kinetics with convection and diffusion

        self.u = tf.constant([[0.1]])
        self.k = tf.constant([[0.5]])
        self.n = tf.constant([[1.0]])



    def call(self, inputs):
        c = self.Mod(inputs)
        return c


    def loss_function(self, Coll_data):

        t,x = Coll_data

        with tf.GradientTape(persistent=True) as tape:
            tape.watch((self.t0,self.x0,self.t_bc,self.x_bc,t,x))
            c0 = self.call(tf.concat([self.x0,self.t0],axis=1))
            c_bc = self.call(tf.concat([self.x_bc,self.t_bc],axis=1))
            c = self.call(tf.concat([x,t],axis=1))
        cx = tape.gradient(c, x)
        ct = tape.gradient(c,t)

        #Initial condition
        L_IC = (1/self.C0.shape[0])*tf.reduce_sum((c0-self.C0)**2)


        #Boundary conditions
        #X
        L_BC = (1/self.C_bc.shape[0])*tf.reduce_sum((c_bc-self.C_bc)**2)

        L_PDE = (1/c.shape[0])*tf.reduce_sum((ct+self.u * cx + self.k*c**(self.n))**2)

        L_PINN = L_IC + L_BC + L_PDE

        # L_data = (1/self.c_d.shape[0]) * tf.reduce_sum((c_d-self.c_d)**2)

        c_val = self.call(tf.concat([self.x_val,self.t_val],axis=1))

        L_val = (1/self.C_val.shape[0])*tf.reduce_sum((c_val-self.C_val)**2)

        return L_PINN, L_val

    def grad(self, Coll_data):
        t,x = Coll_data
        with tf.GradientTape() as grad:
            grad.watch((self.trainable_variables,self.t0,self.x0,self.t_bc,self.x_bc,t,x))
            Lp, Lv = self.loss_function(Coll_data)

        g = grad.gradient(Lp, self.trainable_variables)

        return g, Lp,Lv

    def get_data(self, N):

        # Collocation points
        r_c_t = np.random.choice(np.arange(0, len(self.t_c)), N, replace=False)
        r_c_x = np.random.choice(np.arange(0, len(self.x_c)), N, replace=False)
        #if r_c_t != 0:
            #mb_t_c = tf.constant(np.concatenate([np.repeat(self.tc[r_c_t],repeats = N/2).reshape(-1,1),np.repeat(self.tc[r_c_t-1],repeats = N/2).reshape(-1,1)],axis=0),dtype = tf.float32)
        #else:
            #mb_t_c = tf.constant(np.concatenate([np.repeat(self.tc[r_c_t],repeats = N/2).reshape(-1,1),np.repeat(self.tc[r_c_t+1],repeats = N/2).reshape(-1,1)],axis=0),dtype = tf.float32)

        mb_x_c = tf.convert_to_tensor(self.x_c[r_c_x].reshape((-1,1)), dtype=tf.float32)
        mb_t_c = tf.convert_to_tensor(self.t_c[r_c_t].reshape((-1,1)), dtype=tf.float32)

        return mb_t_c,mb_x_c

    def train(self, max_epochs, pretrain=True):
        epch, loss_p, loss_d, loss_val = [], [], [], []
        i = 0
        os.chdir(path_wts)

        if pretrain:
            self.load_weights(f"weight_f_{self.save_ext}")
            print("Weights have been loaded!")

        else:
            print("Training with Xavier initialization!")
        Coll_data = self.get_data(self.batch_size)
        L1, L2, = self.loss_function(Coll_data)


        widgets = ['| ', progressbar.Timer(),' | ',progressbar.Percentage(), ' ', progressbar.GranularBar(), ' ', progressbar.Counter(format='%(value)d/%(max_value)d'),' ',' | ',progressbar.ETA(),\
                   ' | ',progressbar.FormatLabel(""), progressbar.FormatLabel("")]

                   # ' ', progressbar.FormatLabel('Lp: %0.4f, Lv : %0.4f '%(L1,L2))]
        self.bar = progressbar.ProgressBar(max_value=max_epochs, widgets=widgets, term_width=150).start()
        while i < max_epochs and L2 > 0.00001:
            Coll_data = self.get_data(self.batch_size)

            g, L1, L2 = self.grad(Coll_data)
            self.train_op1.apply_gradients(zip(g, self.trainable_variables))
            epch.append(i)
            loss_p.append(L1)
            loss_val.append(L2)

            widgets[-2],widgets[-1] = progressbar.widgets.FormatLabel(' Lp : {0:.4f}'.format(L1)),\
                                                   progressbar.widgets.FormatLabel(', Lv : {0:.4f} |'.format(L2))
            self.bar.update(i + 1)
            i += 1

        self.bar.finish()

        os.chdir(path_save)
        np.save(f"L_{self.save_ext}.npy", np.vstack((loss_p, loss_val)))
        os.chdir(path_wts)
        self.save_weights(f"weight_f_{self.save_ext}")
        #self.loss_plot()

        return epch, (loss_p, loss_val)

    def predict(self, inputs):
        os.chdir(path_wts)
        self.load_weights(f"weight_f_{self.save_ext}")
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
    def plot_contour(self,t, x, C1, C2=np.array([0.0]), both=False):

        if both == False:

            fig, ax = plt.subplots(figsize=(12, 6))
            im = ax.imshow(C1.T, extent=[x.min(), x.max(), t.min(), t.max()], aspect='auto', cmap='viridis',
                           origin='lower')
            # ax.set_title('')
            ax.set_xlabel('X')
            ax.set_ylabel('T')
            cbar = fig.colorbar(im, ax=ax, orientation='vertical')
            cbar.set_label('Concentration')
            plt.tight_layout()
            plt.show()
        else:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

            # Plot the first subplot
            im1 = ax1.imshow(C1.T, extent=[x.min(), x.max(), t.min(), t.max()], aspect='auto', cmap='viridis',
                             origin='lower')
            ax1.set_title('Data')
            ax1.set_xlabel('X')
            ax1.set_ylabel('T')

            # Plot the second subplot
            im2 = ax2.imshow(C2, extent=[x.min(), x.max(), t.min(), t.max()], aspect='auto', cmap='viridis',
                             origin='lower')
            ax2.set_title('Predicted')
            ax2.set_xlabel('X')
            ax2.set_ylabel('T')

            # Add a shared colorbar
            cbar = fig.colorbar(im1, ax=[ax1, ax2], orientation='vertical')
            cbar.set_label('Concentration')

    def plot_anim(self,c_pred, x, t, c_data, double_anim = False):
        if double_anim:
            ## Plotting both videos simultaneously

            fig, ax = plt.subplots()
            fig.subplots_adjust(right=0.8)

            def animate_both(i):
                ax.clear()
                ax.set_ylim([0, 1.5])
                ax.plot(x, c_pred[i, :],label = "Predicted")
                ax.plot(x, c_data[i, :],label = "Data")
                ax.set_title('%03d' % (i))
                ax.set_xlabel("Location (x)")
                ax.set_ylabel("Concentration (C)")
                ax.legend()

            ani12 = animation.FuncAnimation(fig, animate_both, len(t), interval=10, blit=False)

            plt.show()

        else:
            fig, ax = plt.subplots()

            def animate(i):
                ax.set_ylim([0, 1.5])
                ax.plot(x, c_pred[i, :])
                ax.set_title('%03d' % (i))
                ax.set_xlabel("Location (x)")
                ax.set_ylabel("Concentration (C)")

            interval = 2
            ani = animation.FuncAnimation(fig, animate, len(t), interval=10, blit=False)
            plt.show()

    def plot_contour(self,t, x, C1, C2=np.array([0.0]), both=False):

        if both == False:

            fig, ax = plt.subplots(figsize=(12, 6))
            im = ax.imshow(C1.T, extent=[x.min(), x.max(), t.min(), t.max()], aspect='auto', cmap='viridis',
                           origin='lower')
            # ax.set_title('')
            ax.set_xlabel('X')
            ax.set_ylabel('T')
            cbar = fig.colorbar(im, ax=ax, orientation='vertical')
            cbar.set_label('Concentration')
            plt.tight_layout()
            plt.show()
        else:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

            # Plot the first subplot
            im1 = ax1.imshow(C1.T, extent=[x.min(), x.max(), t.min(), t.max()], aspect='auto', cmap='viridis',
                             origin='lower')
            ax1.set_title('Data')
            #ax1.set_xlabel('X')
            ax1.set_ylabel('T')

            # Plot the second subplot
            im2 = ax2.imshow(C2, extent=[x.min(), x.max(), t.min(), t.max()], aspect='auto', cmap='viridis',
                             origin='lower')
            ax2.set_title('Predicted')
            ax2.set_xlabel('X')
            ax2.set_ylabel('T')

            # Add a shared colorbar
            cbar = fig.colorbar(im1, ax=[ax1, ax2], orientation='vertical')
            cbar.set_label('Concentration')
            plt.show()


    def pred_plot(self, cpred, x, cdata):
        fig, ax = plt.subplots()
        ax.set_ylim([0, 1.5])
        ax.plot(x, cpred[-1, :],label = "Predicted")
        ax.plot(x, cdata[-1, :],label = "Data")
        ax.set_ylabel("Concentration (C)")
        ax.set_xlabel("Location (x)")
        ax.legend()
        plt.show()


# Training the model
if __name__ == "__main__":
    path_wts = r"/home/karvis/PINN/C_rxn_diff/NN/wts"
    path_data = r"/home/karvis/PINN/C_rxn_diff/Numerical"
    path_save = r"/home/karvis/PINN/C_rxn_diff/NN/results/graphs"

    model = PINN(8, 20	, tf.nn.tanh)


    #start = timer()
    #epch, L = model.train(max_epochs=10000, pretrain=False)
    #stop = timer()
    #print(f'\nTime taken for training is {stop - start} s')
    #model.loss_plot()



    t_p = t_data[::10]
    x_p = x_data
    X_p,T_p = np.meshgrid(x_p,t_p)

    c_p = c_data[::10,:]
    c_pred = model.predict(tf.concat([X_p.flatten().reshape(-1,1),T_p.flatten().reshape(-1,1)],axis=1))
    c_pred = np.reshape(c_pred,newshape=(len(t_p),len(x_p)))
    model.pred_plot(c_pred,x_p,c_p)

    model.plot_anim(c_pred,x_p,t_p,c_p,double_anim=True)
    model.plot_contour(t_p,x_p,c_p.T,c_pred,both=True)
    print("Success!")


