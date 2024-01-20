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
    def __init__(self, nhl, npl, act, type):

        super(PINN, self).__init__()

        self.Mod = self.create_model(nhl,npl,act)

        actDict = {tf.nn.tanh: "tanh", tf.nn.relu: "relu", tf.nn.sigmoid: "sigmoid", tf.nn.elu: "elu"}

        self.save_ext = f"{type}_{nhl}_{npl}_{actDict[act]}"

        self.batch_size = 32

        self.type_of_network = type

        lr=tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-3,decay_rate=0.09,decay_steps=10000)
        
        self.train_op1 = tf.keras.optimizers.Adam(learning_rate=lr)


        self.path_wts = os.getcwd() + r"/1D_steady/NN/wts"
        self.path_plots = os.getcwd() + r"/1D_steady/NN/plots"
        self.path_data = os.getcwd() + r"/1D_steady/Num"

        os.chdir(self.path_data)

        self.u = np.round(np.arange(0.01,0.21,0.01),2)
        if self.type_of_network == 'NN':
            self.Mod.compile(optimizer= self.train_op1, loss = self.loss_function_NN)
            self.x_train = np.load('x.npy')
            self.C_train = np.load('C.npy')

        elif self.type_of_network == 'PINN':
            self.Mod.compile(optimizer= self.train_op1, loss = self.loss_function_PINN)

            self.x0_train = tf.zeros(shape = (self.batch_size,1))
            self.C0_train = tf.ones_like(self.x0_train)

            self.x_train = np.round(np.linspace(0,1,10001),4)
            self.C_train = tf.random.normal(shape=(self.batch_size,1))

        elif self.type_of_network == 'data-PINN':
            self.Mod.compile(optimizer= self.train_op1, loss = self.loss_function_dPINN)

            #Training data
            self.x_train_data = np.load('x.npy')
            self.C_train_data = np.load('C.npy')

            #Collocation data
            #BC
            self.x0_train = tf.zeros(shape = (self.batch_size,1))
            self.C0_train = tf.ones_like(self.x0_train)
            self.x_train = np.round(np.linspace(0,1,10001),4)

        else:
            raise(RuntimeError('Training type is incorrect. Either use "NN", "data-PINN" or "PINN"'))

        self.ip,self.op = self.get_data(self.batch_size,type)



    def create_model(self,nhl, npl, act):
        mod = Sequential()
        # mod.add(Dense(npl,activation=act,kernel_initializer=tf.keras.initializers.GlorotUniform(seed=i)))
        for i in range(nhl):
            mod.add(Dense(npl,activation=act,kernel_initializer=tf.keras.initializers.GlorotUniform(seed=i)))
        mod.add(Dense(1,kernel_initializer=tf.keras.initializers.GlorotUniform(seed=nhl+1)))
        mod.build([None,2])

        return mod
    
    def loss_function_NN(self, y_true, y_pred):
            
        Loss_NN = (1/y_pred.shape[0])*tf.reduce_sum((y_true-y_pred)**2)

        return Loss_NN

    def loss_function_PINN(self, y_true, y_pred):
        x,u = tf.split(self.ip,num_or_size_splits=2,axis=-1)
        k = 0.5
        with tf.GradientTape(persistent=True) as tape:
            tape.watch((x,u,self.x0_train))
            C0_pred = self.Mod.call(tf.concat([self.x0_train,u],axis=1))
            C_pred = self.Mod.call(tf.concat([x,u],axis=1))
            dCdx = tape.gradient(C_pred,x)
        L_PINN = (1/self.C0_train.shape[0])*tf.reduce_sum((self.C0_train-C0_pred)**2)   +   \
                 (1/C_pred.shape[0])*tf.reduce_sum((dCdx*u + k*C_pred)**2)
        del tape
        
        return L_PINN

    def loss_function_dPINN(self, y_true, y_pred):
        x,u = tf.split(self.ip,num_or_size_splits=2,axis=-1)
        k = 0.5
        with tf.GradientTape(persistent=True) as tape:
            tape.watch((x,u,self.mb_x_d,self.mb_u_d))
            C0_pred = self.Mod.call(tf.concat([self.x0_train,u],axis=1))
            C_pred = self.Mod.call(tf.concat([x,u],axis=1))
            C_pred_data = self.Mod.call(tf.concat([self.mb_x_d,self.mb_u_d],axis=1))
            dCdx = tape.gradient(C_pred,x)

        L_data = (1/C_pred_data.shape[0])*tf.reduce_sum((C_pred_data-self.mb_C_d)**2)
        L_PINN = (1/self.C0_train.shape[0])*tf.reduce_sum((self.C0_train-C0_pred)**2)   +   \
                 (1/C_pred.shape[0])*tf.reduce_sum((dCdx*u + k*C_pred)**2)
        
        del tape
        
        return L_data + L_PINN

    def get_data(self, N,type):

        # ru = np.random.choice(np.arange(0, len(self.u)), 1, replace=False)
        ru = np.random.randint(0,len(self.u))
        if type == 'NN':
            r = np.random.choice(np.arange(0, len(self.x_train)), N, replace=False)
            mb_x = tf.convert_to_tensor(self.x_train[r].reshape((-1,1)), dtype=tf.float32)
            mb_u = tf.convert_to_tensor(self.u[ru]*np.ones(shape=(N,1)), dtype=tf.float32)
            mb_C = tf.convert_to_tensor(self.C_train[r,ru].reshape((-1,1)), dtype=tf.float32)
            return (tf.concat([mb_x,mb_u],axis=1)),(mb_C)
        
        elif type == 'PINN':
            ##Coll data
            r = np.random.choice(np.arange(0, len(self.x_train)), N, replace=False)
            mb_x = tf.convert_to_tensor(self.x_train[r].reshape((-1,1)), dtype=tf.float32)
            mb_u = tf.convert_to_tensor(self.u[ru]*np.ones(shape=(N,1)), dtype=tf.float32)
            return (tf.concat([mb_x,mb_u],axis=1)),(tf.zeros_like(mb_x))

        
        elif type == 'data-PINN':

            ##Training data
            r = np.random.choice(np.arange(0, len(self.x_train)), N, replace=False)
            self.mb_x_d = tf.convert_to_tensor(self.x_train_data[r].reshape((-1,1)), dtype=tf.float32)
            self.mb_u_d = tf.convert_to_tensor(self.u[ru]*np.ones(shape=(N,1)), dtype=tf.float32)
            self.mb_C_d = tf.convert_to_tensor(self.C_train_data[r,ru].reshape((-1,1)), dtype=tf.float32)

            #Coll data
            mb_x = tf.convert_to_tensor(self.x_train[r].reshape((-1,1)), dtype=tf.float32)
            mb_u = tf.convert_to_tensor(self.u[ru]*np.ones(shape=(N,1)), dtype=tf.float32)
            return (tf.concat([mb_x,mb_u],axis=1)),(tf.zeros_like(mb_x))

        else:
            raise(RuntimeError('Incorrect type of network'))


    def train(self,max_epochs):
        widgets = ['| ', progressbar.Timer(),' | ',progressbar.Percentage(), ' ', progressbar.GranularBar(), ' ', progressbar.Counter(format='%(value)d/%(max_value)d'),' ',' | ',progressbar.ETA(),\
                   ' | ',progressbar.FormatLabel(""), ' | ']

                   # ' ', progressbar.FormatLabel('Lp: %0.4f, Lv : %0.4f '%(L1,L2))]
        bar = progressbar.ProgressBar(max_value=max_epochs, widgets=widgets, term_width=150).start()

        for i in range(max_epochs):
            self.ip,self.op = self.get_data(self.batch_size,self.type_of_network)
            h = self.Mod.fit(self.ip,self.op,epochs=1,verbose=0)
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


    def pred_plot(self, x, cpred, cdata):
        fig, ax = plt.subplots()
        ax.plot(self.x, cpred, label = 'Predicted')
        ax.plot(self.x, cdata, label = "Data")
        ax.set_xlabel('Location')
        ax.set_ylabel('Concentration')
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

    type = ['NN', 'PINN', 'data-PINN']

    model = PINN(8, 50, tf.nn.tanh,'data-PINN')
    model.train(max_epochs=10)

    #Predict and plot
    # os.chdir(model.path_data)
    # t = np.load('t.npy')[::100]
    # Cd = np.load('C.npy')[::100,:] 
    # Cp = model.predict_funct(t)

    # model.pred_plot(t,Cp,Cd)


