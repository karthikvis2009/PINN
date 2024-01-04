from pinn import PINN as p
from timeit import default_timer as timer
import tensorflow as tf
import os
import pickle
import numpy as np

nhl = [6,8,10,12,16]
npl = [20,50,100]
act = [tf.nn.relu,tf.nn.tanh]
actDict = {tf.nn.relu:"relu",tf.nn.tanh:"tanh"}

time_dict = {}
final_loss = {}
cwd = os.getcwd()
for i in range(len(npl)):
    for j in range(len(nhl)):
        for k in range(len(act)):
            os.chdir(cwd)
            model = p(nhl[j], npl[i], act[k])
            start = timer()
            epch, L = model.train(max_epochs=10000, pretrain=False)
            stop = timer()
            t = stop-start
            print(f'\nTime taken for training is {stop - start} s')
            time_dict.update({f"t_{nhl[j]}_{npl[i]}_{actDict[act[k]]}":round(t,4)})
            final_loss.update({f"L_{nhl[j]}_{npl[i]}_{actDict[act[k]]}":np.round(L[-1],4)})

print(time_dict)
os.chdir(model.path_loss)
with open('final_losses.pkl', 'wb') as f:
    pickle.dump(final_loss, f)
with open('final_losses.pkl', 'rb') as f:
    loaded_dict = pickle.load(f)
    print(loaded_dict)
