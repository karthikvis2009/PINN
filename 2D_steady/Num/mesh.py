import numpy as np
import matplotlib.pyplot as plt

class msh:
    def __init__(self):
        xl = 2.87
        yl = 1.39
        self.dx = 0.005
        self.dy = 0.005

        self.x,self.y = np.arange(0,xl+self.dx,self.dx),np.arange(0,yl+self.dy,self.dy)
        Y_,X_ = np.round(np.meshgrid(self.y,self.x),3)

        self.ind = (((X_>=0)&(X_<=2.87)&(Y_>=0)&(Y_<=0.06))|
                       ((X_>=1.82)&(X_<=1.85)&(Y_>=0.06)&(Y_<=1.39))|
                       ((Y_==0.06)&(X_>=1.82)&(X_<=1.85)))

        self.X,self.Y = np.ma.masked_where(~self.ind,X_),np.ma.masked_where(~self.ind,Y_)     #Get the domain


        self.indices = self.gen_mesh()  #Get the indices for inlets, outlet and walls



    def gen_mesh(self):

        X,Y = self.X,self.Y

        ind_in_main = np.where(((X==0)&(Y>0)&(Y<0.06)))
        ind_in_side = np.where(((Y==1.39)&(X>1.82)&(X<1.85)))
        ind_out = np.where(((X==2.87)&(Y>=0)&(Y<=0.06)))
        ind_wall = np.where(((Y==0.06)&(X>=0)&(X<=1.82))|
                        ((X==1.82)&(Y>=0.06)&(Y<=1.39))|
                        ((X==1.85)&(Y>=0.06)&(Y<=1.39))|
                        ((Y==0.06)&(X>=1.85)&(X<=2.87))|
                        ((Y==0)&(X>=0)&(X<=2.87)))

        #Locations for Neumann Boundary conditions
        ind_in_main_1 = np.where(((X==self.dx)&(Y>0)&(Y<0.06)))
        ind_in_side_1 = np.where(((Y==1.39-self.dy)&(X>1.82)&(X<1.85)))
        ind_out_1 = np.where(((X==2.87-self.dx)&(Y>=0)&(Y<=0.06)))

        ind_wall_1 = np.where(((Y==0.06-self.dy)&(X>=0)&(X<=1.82))|
                        ((X==1.82+self.dx)&(Y>0.06)&(Y<=1.39))|
                        ((X>(1.85-2*self.dx))&(X<1.85)&(Y>0.06)&(Y<=1.39))|
                        ((Y==0.06-self.dy)&(X>=1.85)&(X<=2.87))|
                        ((Y==self.dy)&(X>=0)&(X<=2.87)))

        print(ind_wall[0].shape,ind_wall_1[0].shape)


        return(ind_in_main,ind_in_side,ind_out,ind_wall,[ind_in_main_1,ind_in_side_1,ind_out_1,ind_wall_1])



    def plot_mesh(self,inds):
        ind_in_main,ind_in_side,ind_out,ind_wall,_ = inds
        X,Y = self.X,self.Y
        fig,ax = plt.subplots(figsize=(12,12))
        ax.scatter(X,Y,color = 'r')
        ax.scatter(X[ind_in_main],Y[ind_in_main],color = 'b')
        ax.scatter(X[ind_in_side],Y[ind_in_side],color = 'b')
        ax.scatter(X[ind_out],Y[ind_out],color = 'g')
        ax.scatter(X[ind_wall],Y[ind_wall],color = 'y')
        plt.show()

if __name__ == "__main__":
    g = msh()
    # inds = g.gen_mesh()



