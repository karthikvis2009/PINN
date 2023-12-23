import os
import numpy as np
from sklearn.model_selection import train_test_split
os.chdir(os.getcwd()+'/2D_unsteady/Num')

r_og= np.load('r.npy')
th_og = np.load('th.npy')
t_og = np.load('t.npy')
T_al = np.load('al_T.npy')
T_br = np.load('br_T.npy')
T_cs = np.load('cs_T.npy')
T_cu = np.load('cu_T.npy')
T_ss = np.load('ss_T.npy')

k_const = [205.0,398.0,16.0,109.0,50.0]
rho_const = [2700.0,8960.0,8000.0,8530.0,7850.0]
Cp_const = [900.0,390.0,500.0,380.0,490.0]

k = np.concatenate([np.expand_dims(k_const[0]*np.ones_like(T_al),-1),\
                    np.expand_dims(k_const[1]*np.ones_like(T_br),-1),\
                    np.expand_dims(k_const[2]*np.ones_like(T_cs),-1),\
                    np.expand_dims(k_const[3]*np.ones_like(T_cu),-1),\
                    np.expand_dims(k_const[4]*np.ones_like(T_ss),-1)],axis=-1)

rho = np.concatenate([np.expand_dims(rho_const[0]*np.ones_like(T_al),-1),\
                    np.expand_dims(rho_const[1]*np.ones_like(T_br),-1),\
                    np.expand_dims(rho_const[2]*np.ones_like(T_cs),-1),\
                    np.expand_dims(rho_const[3]*np.ones_like(T_cu),-1),\
                    np.expand_dims(rho_const[4]*np.ones_like(T_ss),-1)],axis=-1)

Cp = np.concatenate([np.expand_dims(Cp_const[0]*np.ones_like(T_al),-1),\
                    np.expand_dims(Cp_const[1]*np.ones_like(T_br),-1),\
                    np.expand_dims(Cp_const[2]*np.ones_like(T_cs),-1),\
                    np.expand_dims(Cp_const[3]*np.ones_like(T_cu),-1),\
                    np.expand_dims(Cp_const[4]*np.ones_like(T_ss),-1)],axis=-1)


T = np.concatenate([np.expand_dims(T_al,-1),np.expand_dims(T_br,-1),np.expand_dims(T_cs,-1),np.expand_dims(T_cu,-1),np.expand_dims(T_ss,-1)],axis=-1)
t = np.broadcast_to(t_og.reshape(-1,1,1,1),shape=(T.shape))
r = np.broadcast_to(r_og.reshape(1,-1,1,1),shape=(T.shape))
th = np.broadcast_to(th_og.reshape(1,1,-1,1),shape=(T.shape))

ip = np.concatenate([t.flatten().reshape(-1,1),r.flatten().reshape(-1,1),th.flatten().reshape(-1,1),\
                     k.flatten().reshape(-1,1),rho.flatten().reshape(-1,1),Cp.flatten().reshape(-1,1)],axis = -1)
T1 = T.flatten().reshape(-1,1)




#100 IC data and 100 BC data for each

#IC
t_ic = np.zeros(shape=(100,1))
r_ic_ = np.linspace(0.04,0.05,10)
th_ic_ = np.linspace(0,np.pi/2,10)
th_ic,r_ic = np.meshgrid(th_ic_,r_ic_)
dummy_props = np.zeros(shape=(20,1))
k_conds = np.concatenate([k_const[0]*np.ones_like(dummy_props),k_const[1]*np.ones_like(dummy_props),\
                    k_const[2]*np.ones_like(dummy_props),k_const[3]*np.ones_like(dummy_props),\
                        k_const[4]*np.ones_like(dummy_props)],axis=1).flatten().reshape(-1,1)

rho_conds = np.concatenate([rho_const[0]*np.ones_like(dummy_props),rho_const[1]*np.ones_like(dummy_props),\
                    rho_const[2]*np.ones_like(dummy_props),rho_const[3]*np.ones_like(dummy_props),\
                        rho_const[4]*np.ones_like(dummy_props)],axis=1).flatten().reshape(-1,1)

Cp_conds = np.concatenate([Cp_const[0]*np.ones_like(dummy_props),Cp_const[1]*np.ones_like(dummy_props),\
                    Cp_const[2]*np.ones_like(dummy_props),Cp_const[3]*np.ones_like(dummy_props),\
                        Cp_const[4]*np.ones_like(dummy_props)],axis=1).flatten().reshape(-1,1)



t_bc = np.linspace(t_og[0],t_og[-1],100).reshape(-1,1)
r_bc1 = r_og[0]*np.ones_like(t_bc)
r_bc2 = r_og[-1]*np.ones_like(t_bc)
th_bc1 = np.zeros_like(t_bc)
th_bc2 = th_og[-1]*np.ones_like(t_bc)

ip_train,ip_test,T_train,T_test = train_test_split(ip,T1, test_size=0.2)
ip_ic = np.concatenate([t_ic,r_ic.reshape(t_ic.shape),th_ic.reshape(t_ic.shape),k_conds,rho_conds,Cp_conds],axis=-1)
ip_bc_r1 = np.concatenate([t_bc,r_bc1,th_ic.reshape(t_ic.shape),k_conds,rho_conds,Cp_conds],axis=-1)
ip_bc_r2 = np.concatenate([t_bc,r_bc2,th_ic.reshape(t_ic.shape),k_conds,rho_conds,Cp_conds],axis=-1)
ip_bc_th1 = np.concatenate([t_bc,r_ic.reshape(t_ic.shape),th_bc1,k_conds,rho_conds,Cp_conds],axis=-1)
ip_bc_th2 = np.concatenate([t_bc,r_ic.reshape(t_ic.shape),th_bc2,k_conds,rho_conds,Cp_conds],axis=-1)

T_ic = 25*np.ones_like(t_ic)
T_bc_r1 = 90*np.ones_like(t_ic)

print(ip_train.shape,ip_test.shape,ip_ic.shape,ip_bc_r1.shape,ip_bc_r2.shape,ip_bc_th1.shape,ip_bc_th2.shape)

os.chdir(r'../NN/Training_data')
np.save('ip_coll.npy',ip_train)
np.save('ip_val.npy',ip_test)
np.save('ip_ic.npy',ip_ic)
np.save('ip_bc_r1.npy',ip_bc_r1)
np.save('ip_bc_r2.npy',ip_bc_r2)
np.save('ip_bc_th1.npy',ip_bc_th1)
np.save('ip_bc_th2.npy',ip_bc_th2)

np.save('T_coll.npy',T_train)
np.save('T_val.npy',T_test)
np.save('T_ic.npy',T_ic)
np.save('T_bc_r1.npy',T_bc_r1)
