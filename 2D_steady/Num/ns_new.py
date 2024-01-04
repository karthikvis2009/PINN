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

		if Re>=2400:
			raise ValueError(f"Turbulent flow obtained with Re = {Re}")

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


	def vel_bc(self,f1,f2):
		# ind_int,ind_in_main,ind_in_side,ind_out,ind_wall,[ind_in_main_1,ind_in_side_1,ind_out_1,ind_wall_1]
		
		f1[self.inds[0]] = self.uin_main		# Main inlet
		f1[self.inds[1]] = 0.0		# Side inlet

		f1[self.inds[2]] = f1[self.inds[-1][2]]		# Outlet
		f1[self.inds[3]] = 0.0		# Wall

		f2[self.inds[0]] = 0.0		# Main inlet
		f2[self.inds[1]] = self.vin_side		# Side inlet
		f2[self.inds[2]] = f2[self.inds[-1][2]]		# Outlet
		f2[self.inds[3]] = 0.0		# Wall


	def p_bc(self,p1):
		p1[self.inds[0]] = p1[self.inds[-1][0]]		# Main inlet
		p1[self.inds[1]] = p1[self.inds[-1][1]]		# Side Inlet
		p1[self.inds[2]] = 0.0		# Outlet
		p1[self.inds[3]] = p1[self.inds[-1][3]]		# Walls



	def solver(self,vel_scheme='central',p_scheme='backward',atol=1e-6):
		u = np.ma.masked_where(~self.m_ind,self.uin_main*np.ones_like(self.X))
		v = np.ma.masked_where(~self.m_ind,self.vin_side*np.ones_like(self.X))
		p = np.ma.masked_where(~self.m_ind,np.zeros_like(self.X))

		#Apply BC
		self.vel_bc(u,v)
		self.p_bc(p)

		print(f"Solving with {vel_scheme} scheme for velocity and {p_scheme} scheme for pressure")


		widgets = ['| ', progressbar.Timer(), ' | ', progressbar.Percentage(), ' ', progressbar.GranularBar(), ' ', 
		progressbar.Counter(format='%(value)d/%(max_value)d'), ' ', ' | ', progressbar.ETA(), ' | ',
		progressbar.FormatLabel(""), progressbar.FormatLabel(""), progressbar.FormatLabel(""), ' | ']

		bar = progressbar.ProgressBar(max_value=self.max_iterations-1, widgets=widgets, term_width=150).start()

		i = 0
		err = 1.0

		while i<self.max_iterations and err>atol :
			u_new,v_new = u.copy(),v.copy()

			dudx = self.diff_x(u_new,vel_scheme)
			dudy = self.diff_y(u_new,vel_scheme)
			dvdx = self.diff_x(v_new,vel_scheme)
			dvdy = self.diff_y(v_new,vel_scheme)

			uxx,uyy,vxx,vyy = self.diff_xx(u_new),self.diff_yy(u_new),self.diff_xx(v_new),self.diff_yy(v_new)	#vyy = uxy from continuity

			us = u_new * dudx + v_new * dudy - (self.mu/self.rho) * (uxx + uyy)
			vs = u_new * dvdx + v_new * dvdy - (self.mu/self.rho) * (vxx + vyy)

			#Apply u BC
			self.vel_bc(us,vs)

			###Applying the pressure-poisson equation

			usx,usy = self.diff_x(us,vel_scheme),self.diff_y(us,vel_scheme)
			vsx,vsy = self.diff_x(vs,vel_scheme),self.diff_y(vs,vel_scheme)

			p_new = p.copy()
			self.p_bc(p_new)

			p[1:-1,1:-1] = ((self.dx**2 + self.dy**2)/2)*(self.rho*(usx[1:-1,1:-1]**2 + 2*usy[1:-1,1:-1]*vsx[1:-1,1:-1] + vsy[1:-1,1:-1]**2) + \
									  (p_new[2:,1:-1]+p_new[:-2,1:-1])/self.dx**2 + \
										(p_new[1:-1,2:]+p_new[1:-1,:-2])/self.dy**2)
			
			px,py = self.diff_x(p,p_scheme),self.diff_y(p,p_scheme)

			u = us + px
			v = vs + py
			
			#Check absolute error
			err_u = np.max(np.abs(u-u_new))
			err_v = np.max(np.abs(v-v_new))
			err_p = np.max(np.abs(p-p_new))

			err = np.max([err_u,err_v,err_p])

			widgets[-4] = progressbar.FormatLabel("Err u : {0:4f} ,".format(err_u))
			widgets[-3] = progressbar.FormatLabel("Err v : {0:4f} ,".format(err_v))
			widgets[-2] = progressbar.FormatLabel("Err p : {0:4f}".format(err_p))

			bar.update(i+1)
			if np.isnan(err_u) or err_u>1000:
				raise ValueError("Solution did not converge in u")
			if  np.isnan(err_v) or err_v>1000:
				raise ValueError("Solution did not converge in v")
			if  np.isnan(err_p) or err_p>1000:
				raise ValueError("Solution did not converge in p")
			
			# elif err_u<=atol and err_v<=atol and err_p<=atol:
			# 	break
			else:
				i+=1
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
				
		
		

if __name__=="__main__":
    consts_dict = {'rho':1000.0,'mu':0.001}
    bc_dict = {'u_in':0.001,'v_in':-0.0001}
    mesh_dict = {'X':np.random.random((100,100)),'Y':np.random.random((100,100)),'dx' : 0.001,'dy' : 0.001,'inds':(4,5)}
    m = NS2D(mesh_dict,consts_dict,bc_dict)
    print("pass")

