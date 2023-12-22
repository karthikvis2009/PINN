import numpy as np
import matplotlib.pyplot as plt

# 1D SS convection and reaction

class oneD_SS:
	def __init__(self):

		self.x = np.linspace(0,1,10001)
		self.dx = self.x[1] - self.x[0]
		self.u = 0.1
		self.k_r = 0.5
		
	def solver(self,max_iter = 10001,a_tol=1e-6):
		#T = np.zeros_like(self.x)
		C = np.zeros_like(self.x)
		C[0] = 1.0
		#u*dC/dx = -k_r*C
		i=0
		err = 1.0
		
		while i<max_iter and err>a_tol:
			C_new = C.copy()
			C_new[1:] = C[:-1] - self.dx*(self.k_r/self.u)*C[:-1]
			C_new[0] = 1.0
			err = np.max(np.abs(C_new-C))
			C=C_new
			i+=1
			print(f"Iteration : {i}")
			
		return C_new
		
	def plot(self,C):
		fig,ax = plt.subplots()
		ax.plot(self.x, C)
		ax.set_xlabel('Location')
		ax.set_ylabel('Concentration')
		ax.set_xlim([self.x.min(),self.x.max()])
		ax.set_ylim([C.min(),C.max()])
		plt.show()
		
if __name__ == "__main__":
	s = oneD_SS()
	C = s.solver()
	s.plot(C)