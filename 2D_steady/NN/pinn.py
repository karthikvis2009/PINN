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

        lr=tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=1e-3,decay_rate=0.09,decay_steps=10000)
        
        self.train_op1 = tf.keras.optimizers.Adam(learning_rate=lr)


        self.path_wts = os.getcwd() + r"/2D_steady/NN/wts"
        self.path_plots = os.getcwd() + r"/2D_steady/NN/plots"
        self.path_data = os.getcwd() + r"/2D_steady/Num"

        os.chdir(self.path_data)

        self.Mod.compile(optimizer= self.train_op1, loss = self.loss_function)

        #Training data

        ip_train_data = np.load('internal_ip.npy')
        op_train_data = np.load('internal_op.npy')


        self.x_train_data = ip_train_data[:,0]
        self.y_train_data = ip_train_data[:,1]

        self.u_train_data = op_train_data[:,0]
        self.v_train_data = op_train_data[:,1]
        self.p_train_data = op_train_data[:,2]

        #Collocation data
        #BC

        ##Inlet
        inlet_ip_coll = np.load('inlet_ip.npy')
        inlet_op_coll = np.load('inlet_op.npy')

        self.x_inlet_coll = inlet_ip_coll[:,0]
        self.y_inlet_coll = inlet_ip_coll[:,1]

        self.u_inlet_coll = inlet_op_coll[:,0]
        self.v_inlet_coll = inlet_op_coll[:,1]
        self.p_inlet_coll = inlet_op_coll[:,2]

        ##Outlet

        outlet_ip_coll = np.load('outlet_ip.npy')
        outlet_op_coll = np.load('outlet_op.npy')

        self.x_outlet_coll = outlet_ip_coll[:,0]
        self.y_outlet_coll = outlet_ip_coll[:,1]

        self.u_outlet_coll = outlet_op_coll[:,0]
        self.v_outlet_coll = outlet_op_coll[:,1]
        self.p_outlet_coll = outlet_op_coll[:,2]

        ##Open (Top)

        top_ip_coll = np.load('open_ip.npy')
        top_op_coll = np.load('open_op.npy')

        self.x_top_coll = top_ip_coll[:,0]
        self.y_top_coll = top_ip_coll[:,1]

        self.u_top_coll = top_op_coll[:,0]
        self.v_top_coll = top_op_coll[:,1]
        self.p_top_coll = top_op_coll[:,2]

        ## Wall (Bot)

        bot_ip_coll = np.load('wall_ip.npy')
        bot_op_coll = np.load('wall_op.npy')

        self.x_bot_coll = bot_ip_coll[:,0]
        self.y_bot_coll = bot_ip_coll[:,1]

        self.u_bot_coll = bot_op_coll[:,0]
        self.v_bot_coll = bot_op_coll[:,1]
        self.p_bot_coll = bot_op_coll[:,2]

        del ip_train_data,op_train_data,inlet_ip_coll,inlet_op_coll,\
            outlet_ip_coll,outlet_op_coll,top_ip_coll,\
            top_op_coll,bot_ip_coll,bot_op_coll
        

        self.ip,self.op = self.get_data(self.batch_size)



    def create_model(self,nhl, npl, act):
        mod = Sequential()
        # mod.add(Dense(npl,activation=act,kernel_initializer=tf.keras.initializers.GlorotUniform(seed=i)))
        for i in range(nhl):
            mod.add(Dense(npl,activation=act,kernel_initializer=tf.keras.initializers.GlorotUniform(seed=i)))
        mod.add(Dense(3,kernel_initializer=tf.keras.initializers.GlorotUniform(seed=nhl+1)))
        mod.build([None,2])
        return mod

    def loss_function(self, y_true, y_pred):
        xd,yd = tf.split(self.ip,num_or_size_splits=2,axis=-1)
        ud,vd,pd = tf.split(self.op,num_or_size_splits=3,axis=-1)
        x,y = self.mb_x,self.mb_y

        mu = 0.001
        rho = 1000.0



        with tf.GradientTape(persistent=True) as tape:
            tape.watch((x,y,xd,yd,self.mb_x_in,self.mb_y_in,\
                        self.mb_x_out,self.mb_y_out,self.mb_x_top,self.mb_y_top,\
                            self.mb_x_bot,self.mb_y_bot))
            
            opp = self.Mod.call(tf.concat([xd,yd],axis=1))
            op = self.Mod.call(tf.concat([x,y],axis=1))
            op_in = self.Mod.call(tf.concat([self.mb_x_in,self.mb_y_in],axis=1))
            op_out = self.Mod.call(tf.concat([self.mb_x_out,self.mb_y_out],axis=1))
            op_top = self.Mod.call(tf.concat([self.mb_x_top,self.mb_y_top],axis=1))
            op_bot = self.Mod.call(tf.concat([self.mb_x_bot,self.mb_y_bot],axis=1))

            ## Splitting the outputs
            up,vp,pp = tf.split(opp,num_or_size_splits=3,axis=1)
            u,v,p = tf.split(op,num_or_size_splits=3,axis=1)
            u_in,v_in,p_in = tf.split(op_in,num_or_size_splits=3,axis=1)
            u_out,v_out,p_out = tf.split(op_out,num_or_size_splits=3,axis=1)
            u_top,v_top,p_top = tf.split(op_top,num_or_size_splits=3,axis=1)
            u_bot,v_bot,p_bot = tf.split(op_bot,num_or_size_splits=3,axis=1)

            ##  Gradients
            ux,uy,vx,vy,px,py = tape.gradient((u,u,v,v,p,p),(x,y,x,y,x,y))

            uxx,uyy,vxx,vyy = tape.gradient((ux,uy,vx,vy),(x,y,x,y))
            u_out_x,u_top_y,v_out_x,v_top_y,p_in_x,p_bot_y = tape.gradient((u_out,u_top,v_out,v_top,p_in,p_bot),\
                                                                           (self.mb_x_out,self.mb_y_top,\
                                                                            self.mb_x_out,self.mb_y_top,\
                                                                            self.mb_x_in,self.mb_y_bot))



        L_data = (1/ud.shape[0])*tf.reduce_sum((ud-up)**2 + (vd-vp)**2 + (pd-pp)**2)
        

        L_int = (1/u.shape[0])*tf.reduce_sum(
                    (u*ux + v*uy + (1/rho)*px - (mu/rho)*(uxx+uyy))**2) +\
                 (1/v.shape[0])*tf.reduce_sum(
                    (u*vx + v*vy + (1/rho)*py - (mu/rho)*(vxx+vyy))**2
                    )
        
        L_in = (1/u_in.shape[0])*tf.reduce_sum((u_in-self.mb_u_in)**2 + (v_in-self.mb_v_in)**2 + (p_in_x)**2)
        L_out = (1/u_out.shape[0])*tf.reduce_sum((u_out_x)**2 + (v_out_x)**2 + (p_out-self.mb_p_out)**2)
        L_top = (1/u_top.shape[0])*tf.reduce_sum((u_top_y)**2 + (v_top_y)**2 + (p_top-self.mb_p_top)**2)
        L_bot = (1/u_bot.shape[0])*tf.reduce_sum((u_bot-self.mb_u_bot)**2 + (v_bot-self.mb_v_bot)**2 + (p_bot_y)**2)

        L_PINN = L_int + L_in + L_out + L_top + L_bot

        del tape
        
        return L_data + L_PINN

    

    def get_data(self, N):

        ##Training data
        r = np.random.choice(np.arange(0, len(self.x_train_data)), N, replace=False)
        mb_x_d = tf.convert_to_tensor(self.x_train_data[r].reshape((-1,1)), dtype=tf.float32)
        mb_y_d = tf.convert_to_tensor(self.y_train_data[r].reshape((-1,1)), dtype=tf.float32)

        mb_u_d = tf.convert_to_tensor(self.u_train_data[r].reshape((-1,1)), dtype=tf.float32)
        mb_v_d = tf.convert_to_tensor(self.v_train_data[r].reshape((-1,1)), dtype=tf.float32)
        mb_p_d = tf.convert_to_tensor(self.p_train_data[r].reshape((-1,1)), dtype=tf.float32)

        #Coll data
        r1 = np.random.choice(np.arange(0, len(self.x_train_data)), N, replace=False)
        self.mb_x = tf.convert_to_tensor(self.x_train_data[r1].reshape((-1,1)), dtype=tf.float32)
        self.mb_y = tf.convert_to_tensor(self.y_train_data[r1].reshape((-1,1)), dtype=tf.float32)

        #BC_inlet (left)
        rbc_in = np.random.choice(np.arange(0, len(self.x_inlet_coll)), N, replace=False)
        self.mb_x_in = tf.convert_to_tensor(self.x_inlet_coll[rbc_in].reshape((-1,1)), dtype=tf.float32)
        self.mb_y_in = tf.convert_to_tensor(self.y_inlet_coll[rbc_in].reshape((-1,1)), dtype=tf.float32)

        self.mb_u_in = tf.convert_to_tensor(self.u_inlet_coll[rbc_in].reshape((-1,1)), dtype=tf.float32)
        self.mb_v_in = tf.convert_to_tensor(self.v_inlet_coll[rbc_in].reshape((-1,1)), dtype=tf.float32)
        self.mb_p_in = tf.convert_to_tensor(self.p_inlet_coll[rbc_in].reshape((-1,1)), dtype=tf.float32)


        #BC_outlet (right)
        rbc_out = np.random.choice(np.arange(0, len(self.x_outlet_coll)), N, replace=False)
        self.mb_x_out = tf.convert_to_tensor(self.x_outlet_coll[rbc_out].reshape((-1,1)), dtype=tf.float32)
        self.mb_y_out = tf.convert_to_tensor(self.y_outlet_coll[rbc_out].reshape((-1,1)), dtype=tf.float32)

        self.mb_u_out = tf.convert_to_tensor(self.u_outlet_coll[rbc_out].reshape((-1,1)), dtype=tf.float32)
        self.mb_v_out = tf.convert_to_tensor(self.v_outlet_coll[rbc_out].reshape((-1,1)), dtype=tf.float32)
        self.mb_p_out = tf.convert_to_tensor(self.p_outlet_coll[rbc_out].reshape((-1,1)), dtype=tf.float32)


        #BC_open (top)
        rbc_top = np.random.choice(np.arange(0, len(self.x_top_coll)), N, replace=False)
        self.mb_x_top = tf.convert_to_tensor(self.x_top_coll[rbc_top].reshape((-1,1)), dtype=tf.float32)
        self.mb_y_top = tf.convert_to_tensor(self.y_top_coll[rbc_top].reshape((-1,1)), dtype=tf.float32)

        self.mb_u_top = tf.convert_to_tensor(self.u_top_coll[rbc_top].reshape((-1,1)), dtype=tf.float32)
        self.mb_v_top = tf.convert_to_tensor(self.v_top_coll[rbc_top].reshape((-1,1)), dtype=tf.float32)
        self.mb_p_top = tf.convert_to_tensor(self.p_top_coll[rbc_top].reshape((-1,1)), dtype=tf.float32)


        #BC_bot (right)
        rbc_bot = np.random.choice(np.arange(0, len(self.x_bot_coll)), N, replace=False)
        self.mb_x_bot = tf.convert_to_tensor(self.x_bot_coll[rbc_bot].reshape((-1,1)), dtype=tf.float32)
        self.mb_y_bot = tf.convert_to_tensor(self.y_bot_coll[rbc_bot].reshape((-1,1)), dtype=tf.float32)

        self.mb_u_bot = tf.convert_to_tensor(self.u_bot_coll[rbc_bot].reshape((-1,1)), dtype=tf.float32)
        self.mb_v_bot = tf.convert_to_tensor(self.v_bot_coll[rbc_bot].reshape((-1,1)), dtype=tf.float32)
        self.mb_p_bot = tf.convert_to_tensor(self.p_bot_coll[rbc_bot].reshape((-1,1)), dtype=tf.float32)



        return (tf.concat([mb_x_d,mb_y_d],axis=1)),(tf.concat([mb_u_d,mb_v_d,mb_p_d],axis=1))


    def train(self,max_epochs):
        widgets = ['| ', progressbar.Timer(),' | ',progressbar.Percentage(), ' ', progressbar.GranularBar(), ' ', progressbar.Counter(format='%(value)d/%(max_value)d'),' ',' | ',progressbar.ETA(),\
                   ' | ',progressbar.FormatLabel(""), ' | ']

                   # ' ', progressbar.FormatLabel('Lp: %0.4f, Lv : %0.4f '%(L1,L2))]
        bar = progressbar.ProgressBar(max_value=max_epochs, widgets=widgets, term_width=150).start()

        for i in range(max_epochs):
            self.ip,self.op = self.get_data(self.batch_size)
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


    def pred_plot(self, ip, op_pred, op, quiver = False):
	

        x,y = ip[:,0],ip[:,1]
        up,vp,pp = op_pred[:,0],op_pred[:,1],op_pred[:,2]
        ud,vd,pd = op[:,0],op[:,1],op[:,2]


        fig,[ax1,ax2] = plt.subplots(2,sharex=True)

        Up = np.sqrt(up**2+vp**2)
        Ud = np.sqrt(ud**2+vd**2)

        lvls = list(np.linspace(min(Up.min(),Ud.min()),max(Up.max(),Ud.max()),101))


        tcf_up = ax1.tricontourf(x,y,Up,cmap='jet',levels=lvls)
        tcf_ud = ax2.tricontourf(x,y,Ud,cmap='jet',levels=lvls)


        cb1=fig.colorbar(tcf_up,ax = [ax1,ax2])
        cb1.set_label('U (m/s)')
        # cb2=fig.colorbar(tcf_ur,cax=ax2)
        # cb2.set_label('U (m/s)')

        ax1.set_ylabel('y')
        ax1.set_title('Predicted')
        ax2.set_ylabel('y')
        ax2.set_xlabel('x')
        ax2.set_title('Ground Truth')
        fig.suptitle('Velocity profiles for prediction and ground truth')
        plt.show()

        os.chdir(self.path_plots)
        plt.savefig(f"vel_{self.save_ext}.png")


    def clear_vars(self):
        vars = dir()
        for n in vars:
            if not n.startswith('__'):
                del n
        gc.collect()
        

# Training the model
if __name__ == "__main__":

    model = PINN(16, 100, tf.nn.tanh)
    model.train(max_epochs=10000)

    # Predict and plot
    os.chdir(model.path_data)

    ip = np.load('val_ip.npy')[::10,:]
    op = np.load('val_op.npy')[::10,:]

    op_pred = model.predict_funct(ip)

    model.pred_plot(ip,op_pred,op)


