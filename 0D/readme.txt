0D system with only temporal variations and no spatial variations

A system of ODEs representing reactions between 3 components

A + B --k1--> C
B + C --k2--> A
A + C --k3--> B

dA/dt = -k1*A*B + k2*B*C - k3*A*C
dB/dt = -k1*A*B - k2*B*C + k3*A*C
dC/dt = k1*A*B - k2*B*C - k3*A*C

A[0] = 1.0
B[0] = 0.5
C[0] = 0.2

k1 = 1.0
k2 = 0.9
k3 = 0.3