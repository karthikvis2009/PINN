Radial heat transfer -2D case

mat = ['al','cu','ss','br','cs'] # Aluminium, Copper, Stainless Steel, Brass, Carbon Steel
k = [205.0,398.0,16.0,109.0,50.0]
rho = [2700.0,8960.0,8000.0,8530.0,7850.0]
Cp = [900.0,390.0,500.0,380.0,490.0]

r = 0.04 to 0.05
theta = 0 to pi/2

dT/dt = k/rhoCp * ((1/r)*d/dr(r*dT/dr) + (1/r2)*d2T/dth2)
BC  @ r = ro (outer boundary) : -kdT/dr = h(T-T_a)  # h = 70, T_a = 25 degC
    @ r = ri (inner boundary) : T = Ti (90 degC)
    @ th = 0 : dT/dth = 0
    @ th = pi/2 : dt/dth = 0