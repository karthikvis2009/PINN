from pinn import PINN as p
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import gc

type = ['NN', 'PINN', 'data-PINN']
nhl = 12
npl = 50
act = tf.nn.tanh

cwd = os.getcwd()
os.chdir(cwd+r'/1D_steady/Num')
x = np.load('x.npy')
C = np.load('C.npy')
u = np.arange(0.01,0.2,0.01)

ru = np.random.randint(0,len(u))
x = np.reshape(x[::100],(-1,1))
u = u[ru]*np.ones_like(x)
ip = tf.convert_to_tensor(np.concatenate([x,u],axis=1),dtype=tf.float32)
Cd = C[::100,ru]

err = np.zeros(shape=len(type))
Cp = []
for i in range(len(type)):
    os.chdir(cwd)
    mod = p(nhl,npl,act,type[i])
    print(f'\nTraining {type[i]} :\n')
    mod.train(max_epochs=10000)
    Cp.append(mod.predict_funct(ip))
    err[i]=np.linalg.norm(Cd-Cp[i],2)/np.linalg.norm(Cd,2)
    tf.keras.backend.clear_session()
    mod.clear_vars()
    del mod
    gc.collect()

def plot(x,u,type,cd,cp):
    fig,ax = plt.subplots()
    ax.set_title(f'Prediction and actual concentrations for u = {float(u):0.2f}')
    ax.plot(x,cd,label='Ground Truth')
    for i in range(len(type)):
        ax.plot(x,cp[i],label=type[i])

    ax.set_xlabel('Location')
    ax.set_ylabel('Concentrations')
    ax.legend()
    os.chdir(cwd+'/1D_steady/NN/plots/')
    plt.savefig('pred_plot.png')
    plt.show()


ind = np.where(err==err.min())
print(f'Prediction errors : {err}\n')

print(f'Minimum prediction error with value err = {float(err[ind]):.4f} \nfor type : {type[int(ind[0])]}')
os.chdir(cwd)
plot(x,u[ru],type,Cd,Cp)

os.chdir(cwd+'/1D_steady/')
with open('readme.txt','r+') as fobj:
    data = fobj.readlines()
    data[-3] = f'NN : {err[0]:0.4f}\n'
    data[-2] = f'PINN : {err[1]:0.4f}\n'
    data[-1] = f'data-PINN : {err[2]:0.4f}\n'
    fobj.seek(0)
    fobj.writelines(data)

