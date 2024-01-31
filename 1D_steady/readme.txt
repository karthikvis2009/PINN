1D convection reaction system to give steady state concentration profile along the length

Constant velocity, u = 0.1
First order reaction with rate Constant k = 0.5

u*dC/dx = -kC
C[0] = 1.0
u = 0.01 to 0.2 with du = 0.01

Neural Network ip : x,u ; op : C

Training difference between NN, partial PINNs and fully forward PINNs

NN : No physics
data PINN : physics + data
no data PINNs : Only physics

Prediction errors:-
NN : 7.7578
PINN : 7.6520
data-PINN : 8.2680
