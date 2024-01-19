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
import gc


tf.get_logger().setLevel('ERROR')
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))


##Dense Network



class PINN(Model):
    def __init__(self, nhl, npl, act):

        super(PINN, self).__init__()

        self.Mod = self.create_model(nhl,npl,act)

        actDict = {tf.nn.tanh: "tanh", tf.nn.relu: "relu", tf.nn.sigmoid: "sigmoid", tf.nn.elu: "elu"}

        self.save_ext = f"{nhl}_{npl}_{actDict[act]}"

        self.batch_size = 32

        # lr=tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-2,decay_rate=0.09,decay_steps=100)
        
        self.train_op1 = tf.keras.optimizers.Adam(learning_rate=0.001)

        self.Mod.compile(optimizer= self.train_op1, loss = self.loss_function)


        #Collocation points
        self.tc = np.linspace(0,10,10001)

        #IC points
        self.t0 = tf.zeros(shape=(self.batch_size,1))

        self.ip = self.get_data(self.batch_size)


        self.path_wts = os.getcwd() + r"/0D/NN/wts"
        self.path_plots = os.getcwd() + r"/0D/NN/plots"
        self.path_data = os.getcwd() + r"/0D/Num"

    def create_model(self,nhl, npl, act):
        mod = Sequential()
        # mod.add(Dense(npl,activation=act,kernel_initializer=tf.keras.initializers.GlorotUniform(seed=i)))
        for i in range(nhl):
            mod.add(Dense(npl,activation=act,kernel_initializer=tf.keras.initializers.GlorotUniform(seed=i)))
        
        mod.add(Dense(3,kernel_initializer=tf.keras.initializers.GlorotUniform(seed=nhl+1)))
        mod.build([None,1])

        return mod


    def loss_function(self, y_true, y_pred):

        k1,k2,k3 = 1.0,0.9,0.3
        t = self.ip
        with tf.GradientTape(persistent=True) as tape:
            tape.watch((self.t0,t))
            C0 = self.Mod.call(self.t0)
            C = self.Mod.call(t)

            dAdt,dBdt,dCdt = tape.gradient((C[:,0],C[:,1],C[:,2]),(t,t,t))

        L_IC = tf.reduce_sum((C0[:,0]-1.0)**2) + tf.reduce_sum((C0[:,1]-0.5)**2) + tf.reduce_sum((C0[:,2]-0.2)**2)

        L_ODE = tf.reduce_sum((dAdt+k1*C[:,0]*C[:,1]-k2*C[:,1]*C[:,2]+k3*C[:,0]*C[:,2])**2) +\
                tf.reduce_sum((dBdt+k1*C[:,0]*C[:,1]+k2*C[:,1]*C[:,2]-k3*C[:,0]*C[:,2])**2)+\
                tf.reduce_sum((dCdt-k1*C[:,0]*C[:,1]+k2*C[:,1]*C[:,2]+k3*C[:,0]*C[:,2])**2)
        
        del tape
        
        return L_IC + L_ODE

    def get_data(self, N):

        # Collocation points
        r_c = np.random.choice(np.arange(0, len(self.tc)), N, replace=False)

        mb_t = tf.convert_to_tensor(self.tc[r_c].reshape((-1,1)), dtype=tf.float32)

        return mb_t

    def train(self,max_epochs):
        widgets = ['| ', progressbar.Timer(),' | ',progressbar.Percentage(), ' ', progressbar.GranularBar(), ' ', progressbar.Counter(format='%(value)d/%(max_value)d'),' ',' | ',progressbar.ETA(),\
                   ' | ',progressbar.FormatLabel(""), ' | ']

                   # ' ', progressbar.FormatLabel('Lp: %0.4f, Lv : %0.4f '%(L1,L2))]
        bar = progressbar.ProgressBar(max_value=max_epochs, widgets=widgets, term_width=150).start()

        for i in range(max_epochs):
            self.ip = self.get_data(self.batch_size)
            # print(tf.reduce_max(self.ip))
            h = self.Mod.fit(self.ip,tf.random.normal(shape=(32,1)),epochs=1,verbose=0)
            self.Mod.reset_states()
            widgets[-2] = progressbar.FormatLabel('L : {0:.4f}'.format(h.history['loss'][0]))
            bar.update(i+1)
            del h


        os.chdir(self.path_wts)
        self.save_weights(f"weight_{self.save_ext}")

        bar.finish()


    def predict_funct(self, inputs):
        os.chdir(self.path_wts)
        self.load_weights(f"weight_{self.save_ext}").expect_partial()
        op = self.Mod.predict(inputs)
        return (op)


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
        ax.set_title(f'Plot of concentrations vs time')
        ax.legend()
        os.chdir(self.path_plots)
        plt.savefig(f"Conc_{self.save_ext}.png")

    def clear_vars(self):
        vars = dir()
        for n in vars:
            if not n.startswith('__'):
                del n
        gc.collect()
        

# Training the model
if __name__ == "__main__":

    model = PINN(8, 50, tf.nn.tanh)
    model.train(max_epochs=100)

    #Predict and plot
    os.chdir(model.path_data)
    t = np.load('t.npy')[::100]
    Cd = np.load('C.npy')[::100,:] 
    Cp = model.predict_funct(t)

    model.pred_plot(t,Cp,Cd)


