from pinn import PINN as p
from timeit import default_timer as timer
import tensorflow as tf
import os
import numpy as np
import gc

def train():


    nhl = [6,8,10,12,16]
    npl = [20,50,100]
    # nhl = [6]
    # npl = [50]
    act = [tf.nn.relu,tf.nn.tanh,tf.nn.sigmoid]
    actDict = {tf.nn.relu:"relu",tf.nn.tanh:"tanh",tf.nn.sigmoid:"sigmoid"}

    err = np.zeros(shape=(len(nhl),len(npl),len(act)))
    cwd = os.getcwd()
    os.chdir(cwd+r'/0D/Num/')
    t = np.load('t.npy')[::100]
    Cd = np.load('C.npy')[::100,:]


    for i in range(len(nhl)):
        for j in range(len(npl)):
            for k in range(len(act)):
                os.chdir(cwd)
                model = p(nhl[i], npl[j], act[k])
                print(f'\nTraining {nhl[i]} , {npl[j]}, {actDict[act[k]]}:\n')
                model.train(max_epochs=1000)
                Cp = model.predict_funct(t)
                err[i,j,k] = np.linalg.norm(Cd-Cp,2)/np.linalg.norm(Cd,2)
                tf.keras.backend.clear_session()
                model.clear_vars()
                del model,Cp
                gc.collect()


    ind = np.where(err==err.min())

    print(f'Minimum error with value err = {float(err[ind]):.4f} \nat nhl = {nhl[int(ind[0])]}, npl = {npl[int(ind[1])]} and act = {actDict[act[int(ind[2])]]}')

    os.chdir(cwd+r'/0D/NN/')
    np.save('err.npy',err)
    os.chdir(cwd)

    nhl_p = nhl[int(ind[0])]
    npl_p = npl[int(ind[1])]
    act_p = act[int(ind[2])]

    model = p(nhl_p,npl_p,act_p)
    Cp = model.predict_funct(t)
    model.pred_plot(t,Cp,Cd)

if __name__=="__main__":
    train()
