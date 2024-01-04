import os
import numpy as np
import matplotlib.pyplot as plt
import progressbar
# 1D SS convection and reaction

class oneD_SS:
	def __init__(self,u):

		self.x = np.linspace(0,1,10001)
		self.dx = self.x[1] - self.x[0]
		self.u = u
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

	u = np.round(np.arange(0.01,0.21,0.01),2)

	path = os.getcwd() + r"/1D_steady/Num/"

	def gen_data(u,path):
		C = []

		widgets = ['| ', progressbar.Timer(), ' | ', progressbar.Percentage(), ' ', progressbar.GranularBar(), ' ', progressbar.Counter(format='%(value)d/%(max_value)d'), ' ', ' | ', progressbar.ETA(), ' | ',progressbar.FormatLabel(""), ' | ']
		bar = progressbar.ProgressBar(max_value=len(u), widgets=widgets, term_width=150).start()
		for i in range(len(u)):
			s = oneD_SS(u[i])
			C.append(s.solver().reshape(-1,1))
			widgets[-2] = progressbar.widgets.FormatLabel('Solving for u = {0:.2f}'.format(u[i]))
			bar.update(i+1)
		bar.finish()
		C1 = np.concatenate(C,axis=-1)
		os.chdir(path)
		np.save('C.npy',C1)
		np.save('x.npy',s.x)

	def plot(u,path):
		os.chdir(path)
		C = np.load('C.npy')
		x = np.load('x.npy')
		fig,ax = plt.subplots()

		for i in range(len(u)):
			ax.plot(x,C[:,i],label = f" u = {u[i]}")
		ax.set_xlabel('x')
		ax.set_ylabel('C')
		ax.set_title('Plot of concentrations for different velocities')
		ax.legend()
		plt.show()

	gen_data(u,path)
	plot(u,path)

