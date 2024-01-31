2D steady case flow past a bump
Bump is in the shape of a semicircle (Half-cylinder in 2D)


#################################################################################################################

Geometry:
-------------------------------------------------   ^
|                                               |   |
|                                               |   |
|                                               |   |   0.05
|       . .                                     |   |
|     .     .                                   |   |
-----.       .-----------------------------------   -

<---><-------><--------------------------------->
 0.01  0.02                 0.07

<----------------------------------------------->
                    0.1


Boundary Conditions:
@ x = 0, u = 0.01 m/s   (Inlet Velocity)
@ x = 0.1 (L), du/dx = 0, dv/dx = 0, p = 0      (Outlet pressure)
@ y = bottom_wall, u = 0, v = 0, dp/dy = 0
@ y = top_open, du/dy = 0, dv/dy = 0, dp/dy = 0

Initial condition:
u = 0
p = 0

#################################################################################################################

Continuity :-
du/dx + dv/dy = 0

Momentum :-

udu/dx + vdu/dy = -(1/rho)dP/dx + (1/rho)*(mu*(d2u/dx2+d2u/dy2))
udv/dx + vdv/dy = -(1/rho)dP/dy + (1/rho)*(mu*(d2v/dx2+d2v/dy2))

rho = 1000
mu = 0.001

Neural Network:

Input :- x,y ; Output :- u,v,p




