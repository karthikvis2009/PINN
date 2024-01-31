import numpy as np
import pandas as pd
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import os


def gen_and_plot(dfc,dfr, quiver = False, interp = 'nearest'):
	####	Coarse
	xc = dfc['Points:0'].to_numpy()
	yc = dfc['Points:1'].to_numpy()
	uc = dfc['U:0'].to_numpy()
	vc = dfc['U:1'].to_numpy()
	pc = dfc['p'].to_numpy()

	#### 	Refined
	xr = dfr['Points:0'].to_numpy()
	yr = dfr['Points:1'].to_numpy()
	ur = dfr['U:0'].to_numpy()
	vr = dfr['U:1'].to_numpy()
	pr = dfr['p'].to_numpy()


	fig,[ax1,ax2] = plt.subplots(2,sharex=True)

	dlv = 0.0001
	Uc = np.sqrt(uc**2+vc**2)
	Ur = np.sqrt(ur**2+vr**2)

	lvls = list(np.linspace(min(Uc.min(),Ur.min()),max(Uc.max(),Ur.max()),101))


	tcf_uc = ax1.tricontourf(xc,yc,Uc,cmap='jet',levels=lvls)
	tcf_ur = ax2.tricontourf(xr,yr,Ur,cmap='jet',levels=lvls)


	cb1=fig.colorbar(tcf_uc,ax = [ax1,ax2])
	cb1.set_label('U (m/s)')
	# cb2=fig.colorbar(tcf_ur,cax=ax2)
	# cb2.set_label('U (m/s)')

	ax1.set_ylabel('y')
	ax1.set_title('Coarse mesh')
	ax2.set_ylabel('y')
	ax2.set_xlabel('x')
	ax2.set_title('Refined mesh')
	fig.suptitle('Velocity profiles for coarse and refined mesh')

	if quiver:
		x1 = np.linspace(xc.min(),xc.max(),101)
		y1 = np.linspace(yc.min(),yc.max(),101)
		xg,yg = np.meshgrid(x1,y1)
		u1 = griddata((xc,yc),uc,(xg,yg),method=interp)
		v1 = griddata((xc,yc),vc,(xg,yg),method=interp)
		ax1.quiver(xg[::5,::5],yg[::5,::5],u1[::5,::5],v1[::5,::5],scale = 4)
		u2 = griddata((xr,yr),ur,(xg,yg),method=interp)
		v2 = griddata((xr,yr),vr,(xg,yg),method=interp)
		ax2.quiver(xg[::5,::5],yg[::5,::5],u2[::5,::5],v2[::5,::5],scale = 4)

	plt.show()



#plot((x,y,u,v,p),quiver=True,interp = 'nearest')



#plot(df1,df2,quiver=True)

def save_data():
	os.chdir('/home/karvis/data_2D_SS/2D_SS/')
	df_inlet = pd.read_csv('inl.csv')
	df_outlet = pd.read_csv('outl.csv')
	df_open = pd.read_csv('open.csv')
	df_wall = pd.read_csv('wall.csv')
	df_int = pd.read_csv('data.csv')
	os.chdir('/home/karvis/data_2D_SS/2D_SS_refined/')
	df_val = pd.read_csv('data.csv')
	df = [df_inlet,df_outlet,df_open,df_wall,df_int,df_val]
	name = ['inlet','outlet','open','wall','internal','val']
	os.chdir('/home/karvis/Thesis/PINN/2D_steady/Num/')
	for d in range(len(df)):
		x = df[d]['Points:0'].to_numpy()
		y = df[d]['Points:1'].to_numpy()
		u = df[d]['U:0'].to_numpy()
		v = df[d]['U:1'].to_numpy()
		p = df[d]['p'].to_numpy()
		np.save(f'{name[d]}_ip.npy',np.concatenate([x.reshape(-1,1),y.reshape(-1,1)],axis=-1))
		np.save(f'{name[d]}_op.npy',np.concatenate([u.reshape(-1,1),v.reshape(-1,1),p.reshape(-1,1)],axis=-1))

save_data()

