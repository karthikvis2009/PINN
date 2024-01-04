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


##Dense Network

class PINN(Model):
    def __init__(self, nhl, npl, act):

        #nhl : [nhl for x-model, nhl for t-model, nhl for shared model]
        super(PINN, self).__init__()

        self.nhl = nhl
        self.Mod = Sequential()

        for i in range(nhl):
            self.Mod.add(Dense(npl,activation=act,kernel_initializer=tf.keras.initializers.GlorotUniform(seed=i)))
        
        self.Mod.add(Dense(3,kernel_initializer=tf.keras.initializers.GlorotUniform(seed=nhl+1)))
        self.Mod.build([None,1])

        actDict = {tf.nn.tanh: "tanh", tf.nn.relu: "relu", tf.nn.sigmoid: "sigmoid", tf.nn.elu: "elu"}

        self.save_ext = f"{nhl}_{npl}_{actDict[act]}"

        self.batch_size = 32

        # lr=tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-2,decay_rate=0.09,decay_steps=100)

        self.train_op1 = tf.keras.optimizers.Adam(learning_rate=0.00001)

        #Collocation points
        self.tc = np.linspace(0,10,10001)

        #IC points
        self.t0 = tf.zeros(shape=(100,1))

        self.path_wts = os.getcwd() + r"/0D/NN/wts"
        self.path_loss = os.getcwd() + r"/0D/NN/loss"
        self.path_data = os.getcwd() + r"/0D/Num"

    def call(self, inputs):
        c = self.Mod(inputs)
        return c


    def loss_function(self, t):

        k1,k2,k3 = 1.0,0.9,0.3

        with tf.GradientTape(persistent=True) as tape:
            tape.watch((self.t0,t))
            C0 = self.call(self.t0)
            C = self.call(t)

            dAdt,dBdt,dCdt = tape.gradient((C[:,0],C[:,1],C[:,2]),(t,t,t))

            L_IC = tf.reduce_mean((C0[:,0]-1.0)**2) + tf.reduce_mean((C0[:,1]-0.5)**2) + tf.reduce_mean((C0[:,2]-0.2)**2)

            L_ODE = tf.reduce_mean((dAdt+k1*C[:,0]*C[:,1]-k2*C[:,1]*C[:,2]+k3*C[:,0]*C[:,2])**2) +\
            tf.reduce_mean((dBdt+k1*C[:,0]*C[:,1]+k2*C[:,1]*C[:,2]-k3*C[:,0]*C[:,2])**2)+\
            tf.reduce_mean((dAdt-k1*C[:,0]*C[:,1]+k2*C[:,1]*C[:,2]+k3*C[:,0]*C[:,2])**2)
        
        return L_IC + L_ODE
    

    def grad(self, mb_t):

        with tf.GradientTape() as grad:
            grad.watch((self.trainable_variables))
            L = self.loss_function(mb_t)

        g = grad.gradient(L, self.trainable_variables)

        return g, L
    

    def get_data(self, N):

        # Collocation points
        r_c = np.random.choice(np.arange(0, len(self.tc)), N, replace=False)

        mb_t = tf.convert_to_tensor(self.tc[r_c].reshape((-1,1)), dtype=tf.float32)

        return mb_t

    def train(self, max_epochs, pretrain=True):
        epch, loss_t = [], []
        i = 0
        os.chdir(self.path_wts)

        if pretrain:
            self.load_weights(f"weight_f_{self.save_ext}")
            print("Weights have been loaded!")

        else:
            print("Training with Xavier initialization!")

        mb_t = self.get_data(self.batch_size)
        L = self.loss_function(mb_t)

        widgets = ['| ', progressbar.Timer(),' | ',progressbar.Percentage(), ' ', progressbar.GranularBar(), ' ', progressbar.Counter(format='%(value)d/%(max_value)d'),' ',' | ',progressbar.ETA(),\
                   ' | ',progressbar.FormatLabel("")]

                   # ' ', progressbar.FormatLabel('Lp: %0.4f, Lv : %0.4f '%(L1,L2))]
        self.bar = progressbar.ProgressBar(max_value=max_epochs, widgets=widgets, term_width=150).start()

        while i < max_epochs and L > 1e-5:

            mb_t = self.get_data(self.batch_size)
            g, L = self.grad(mb_t)
            self.train_op1.apply_gradients(zip(g, self.trainable_variables))
            epch.append(i)
            loss_t.append(L)

            widgets[-1] = progressbar.widgets.FormatLabel(' Lp : {0:.4f}'.format(L))
            self.bar.update(i + 1)
            i += 1

        self.bar.finish()

        os.chdir(self.path_loss)
        np.save(f"L_{self.save_ext}.npy", loss_t)
        os.chdir(self.path_wts)
        self.save_weights(f"weight_f_{self.save_ext}")

        return epch, (loss_t)

    def predict(self, inputs):
        os.chdir(self.path_wts)
        self.load_weights(f"weight_f_{self.save_ext}")
        op = self.call(inputs)
        return (op)


    def loss_plot(self):
        os.chdir(self.path_loss)
        L = np.load(f"L_{self.save_ext}.npy")
        Epch = np.linspace(0, len(L), len(L))
        plt.plot(Epch, L)
        plt.xlabel("Iterations")
        plt.ylabel(r"Loss (Mean squared)")
        plt.title("Plot of Total Loss during training")
        plt.savefig(r"loss_plot" + self.save_ext + ".png")
        plt.show()


    def pred_plot(self, t, cpred, cdata):
        fig, ax = plt.subplots()
        ax.plot(t, cpred[:, 0],'g--',label = "A Predicted")
        ax.plot(t, cdata[:, 0],'g',label = "A Data")
        ax.plot(t, cpred[:, 1],'r--',label = "B Predicted")
        ax.plot(t, cdata[:, 1],'r',label = "B Data")
        ax.plot(t, cpred[:, 2],'b--',label = "C Predicted")
        ax.plot(t, cdata[:, 2],'b',label = "C Data")
        ax.set_ylabel("Concentration (C)")
        ax.set_xlabel("Time")
        ax.legend()
        plt.show()


# Training the model
if __name__ == "__main__":

    model = PINN(8, 100, tf.nn.relu)

    start = timer()
    epch, L = model.train(max_epochs=10000, pretrain=False)
    stop = timer()
    print(f'\nTime taken for training is {stop - start} s')
    model.loss_plot()

    #Predict and plot
    os.chdir(model.path_data)
    t = np.load('t.npy')[::100]
    Cd = np.load('C.npy')[::100,:] 
    Cp = model.predict(t)

    model.pred_plot(t,Cp,Cd)


