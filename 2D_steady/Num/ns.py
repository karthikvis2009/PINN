import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import progressbar

class NS2D():
	def __init__(self,mesh_dict,consts_dict,bc_dict,max_iter = 10000):
		self.Y,self.X = mesh_dict['Y'],mesh_dict['X']

		self.dx = mesh_dict['dx']
		self.dy = mesh_dict['dy']
		self.inds,self.m_ind = mesh_dict['inds'] # ind_in_main,ind_in_side,ind_out,ind_wall

		##Constants
		self.rho = consts_dict['rho']
		self.mu =consts_dict['mu']
		self.uin_main = bc_dict['u_in']
		self.vin_side = bc_dict['v_in']

		self.max_iterations = max_iter
		
		Re = max(abs(self.rho * self.uin_main * 0.6/self.mu),abs(self.rho * self.vin_side * 0.03/self.mu))

		print(f"Re : {Re}")
		self.path_save = os.getcwd() + r"/2D_steady/Num/"

	##Backward differences for first order derivatives
		
		
	def diff_x(self,f,scheme = 'central'):
		diff = np.zeros_like(f)
		if scheme=='backward':
			diff[1:-1,1:-1] = (f[1:-1,1:-1]-f[:-2,1:-1])/(self.dx)

		elif scheme=='forward':
			diff[1:-1,1:-1] = (f[2:,1:-1]-f[1:-1,1:-1])/(self.dx)


		elif scheme=='central':
			diff[1:-1,1:-1] = (f[2:,1:-1]-f[:-2,1:-1])/(2*self.dx)

		else:
			raise RuntimeError(r"Please specify a valid discretisation scheme - 'backward', 'forward' or 'central'")
		return diff


	def diff_y(self,f,scheme = 'central'):

		diff = np.zeros_like(f)
		if scheme =='backward':
			diff[1:-1,1:-1] = (f[1:-1,1:-1]-f[1:-1,:-2])/(self.dy)

		elif scheme =='forward':
			diff[1:-1,1:-1] = (f[2:,1:-1]-f[1:-1,1:-1])/(self.dy)

		elif scheme =='central':
			diff[1:-1,1:-1] = (f[2:,1:-1]-f[:-2,1:-1])/(2*self.dy)
		else:
			raise RuntimeError(r"Please specify a valid discretisation scheme - 'backward', 'forward' or 'central'")
		return diff
	
	def diff_xx(self,f):
		diff = np.zeros_like(f)
		diff[1:-1,1:-1] = (f[2:,1:-1] - 2*f[1:-1,1:-1] + f[:-2,1:-1])/self.dx**2
		return diff
	
	def diff_yy(self,f):
		diff = np.zeros_like(f)
		diff[1:-1,1:-1] = (f[1:-1,2:] - 2*f[1:-1,1:-1] + f[1:-1,:-2])/self.dy**2
		return diff


	# def laplacian(self,f):
	# 	diff = np.zeros_like(f)
	# 	diff[1:-1,1:-1] = (f[2:,1:-1] - 2*f[1:-1,1:-1] + f[:-2,1:-1])/self.dx**2 \
	# 			+ (f[1:-1,2:] - 2*f[1:-1,1:-1] + f[1:-1,:-2])/self.dy**2
	# 	return diff
	

	def vel_bc(self,f1,f2):
		# ind_int,ind_in_main,ind_in_side,ind_out,ind_wall
		f1[self.inds[0]] = self.uin_main		# Main inlet
		f1[self.inds[1]] = 0.0		# Side inlet
		f1[self.inds[2],self.inds[2][1]] = f1[self.inds[2][0]-1,self.inds[2][1]]		# Outlet
		f1[self.inds[3]] = 0.0		# Wall

		f2[self.inds[0]] = 0.0		# Main inlet
		f2[self.inds[1]] = self.vin_side		# Side inlet
		f2[self.inds[2][0],self.inds[2][1]] = f2[self.inds[2][0]-1,self.inds[2][1]]		# Outlet
		f2[self.inds[3]] = 0.0		# Wall


	def p_bc(self,p1):
		p1[self.inds[2]] = 0.0		# Outlet


	def solver(self,vel_scheme='central',p_scheme='forward',atol=1e-6):
		u = np.ma.masked_where(~self.m_ind,self.uin_main*np.ones_like(self.X))
		v = np.ma.masked_where(~self.m_ind,self.vin_side*np.ones_like(self.X))
		p = np.ma.masked_where(~self.m_ind,np.zeros_like(self.X))

		plt.pcolormesh(u.T)
		plt.show()

		print(f"Solving with {vel_scheme} scheme for velocity and {p_scheme} scheme for pressure")


		widgets = ['| ', progressbar.Timer(), ' | ', progressbar.Percentage(), ' ', progressbar.GranularBar(), ' ', progressbar.Counter(format='%(value)d/%(max_value)d'), ' ', ' | ', progressbar.ETA(), ' | ',progressbar.FormatLabel(""), progressbar.FormatLabel(""), ' | ']

		bar = progressbar.ProgressBar(max_value=self.max_iterations-1, widgets=widgets, term_width=150).start()

		i = 0
		err = 1.0

		while i<self.max_iterations and err>atol:
			u_new,v_new = u.copy(),v.copy()

			dudx = self.diff_x(u,vel_scheme)
			dudy = self.diff_y(u,vel_scheme)
			dvdx = self.diff_x(v,vel_scheme)

			dpdx,dpdy = self.diff_x(p,p_scheme),self.diff_y(p,p_scheme)

			uxx,uyy,vxx,uxy = self.diff_xx(u),self.diff_yy(u),self.diff_xx(v),self.diff_y(self.diff_x(-u))	#vyy = uxy from continuity

			u_new = (dudx**-1)*(-v*dudy-(1/self.rho)*dpdx+(self.mu/self.rho)*(uxx+uyy))
			v_new = (-dudx**-1)*(-u*dvdx-(1/self.rho)*dpdy+(self.mu/self.rho)*(vxx+uxy))

			###Applying the pressure-poisson equation
			

			#Apply boundary conditions for velocities and pressure
			self.vel_bc(u_new,v_new)
			self.p_bc(p)

			#Check absolute error
			err_u = np.max(np.abs(u-u_new))
			err_v = np.max(np.abs(u-u_new))

			widgets[-3] = progressbar.FormatLabel("Err u : {0:4f} ,".format(err_u))
			widgets[-2] = progressbar.FormatLabel("Err v : {0:4f}".format(err_v))
			bar.update(i+1)
			if np.isnan(err_u):
				raise ValueError("Solution did not converge in u")
			if  np.isnan(err_v):
				raise ValueError("Solution did not converge in v")
			elif err_u<=atol and err_v<=atol:
				break
			else:
				print(f"iteration {i}")
				i+=1
				u,v = u_new,v_new
				continue


		bar.finish()

		return(u,v)

		# self.save(t_sol,self.x,self.y,u_sol,v_sol,p_sol)



	def plot(self,u,v):

		fig, ax = plt.subplots()
		U = np.sqrt(u**2+v**2)
		im = ax.imshow(U.T,extent=[self.X.min(),self.X.max(),self.Y.min(),self.Y.max()],origin='lower',cmap='jet')
		ax.set_xlabel('X')
		ax.set_ylabel('Y')

		cbar = fig.colorbar(im, ax=[ax], orientation='vertical')
		cbar.set_label('Velocity Magnitude')
		plt.show()



	def save(self,t,x,y,u,v,p):
		os.chdir(self.path_save)
		np.save("t_ns.npy", t)
		np.save("x_ns.npy", x)
		np.save("y_ns.npy", y)
		np.save("u_ns.npy", u)
		np.save("v_ns.npy", v)
		np.save("p_ns.npy", p)

	def load(self):
		os.chdir(self.path_save)
		t1 = np.load("t_ns.npy")
		x1 = np.load("x_ns.npy")
		y1 = np.load("y_ns.npy")
		u1 = np.load("u_ns.npy")
		v1 = np.load("v_ns.npy")
		p1 = np.load("p_ns.npy")
		return (t1, x1, y1, u1, v1, p1)
				
		
		

# if __name__=="__main__":
# 	m = NS2D(nx=51,ny=21,nt=10001,T=10)
# 	for t,u,v,p in m.solver(10,vel_scheme='backward',p_scheme='central'):
# 		pass
# 	t1,x1,y1,u1,v1,p1 = m.load()
# 	m.plot(x1,y1,p1[-1,:,:],u1[-1,:,:],v1[-1,:,:])
# 	m.plot_anim(t1,x1,y1,p1,u1,v1)
# 	m.plot_anim_U(t1,x1,y1,u1,v1)

