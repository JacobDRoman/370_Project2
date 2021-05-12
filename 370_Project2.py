import numpy as np
from matplotlib import pyplot as plt
from matplotlib import axes as ax
import sympy as sym
from numpy import linalg as LA
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import Axes3D
from numpy import percentile
import time
####Pyplot font settings
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=False)
plt.rcParams['font.size']=20

pi = np.pi
#No built-in pyplot to create waterfall plot
#The following function was taken from: https://stackoverflow.com/questions/46366461/matplotlib-3d-waterfall-plot-with-colored-heights
def waterfall_plot(fig,ax,X,Y,Z):
    '''
    Make a waterfall plot
    Input:
    fig,ax : matplotlib figure and axes to populate
    Z : n,m numpy array. Must be a 2d array even if only one line should be plotted
    X,Y : n,m array
    '''
    # Set normalization to the same values for all plots
    norm = plt.Normalize(Z.min().min(), Z.max().max())
    # Check sizes to loop always over the smallest dimension
    n,m = Z.shape
    if n>m:
        X=X.T
        Y=Y.T
        Z=Z.T
        m,n = n,m
    for j in range(n):
        # reshape the X,Z into pairs
        points = np.array([X[j,:], Z[j,:]]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap='plasma', norm=norm)
        # Set the values used for colormapping
        lc.set_array((Z[j,1:]+Z[j,:-1])/2)
        lc.set_linewidth(1.5) # set linewidth a little larger to see properly the colormap variation
        line = ax.add_collection3d(lc, zs=(Y[j,1:]+Y[j,:-1])/2, zdir='y') # add line to axes
    #fig.colorbar(lc) # add colorbar, as the normalization is the same for all, it doesent matter which of the lc objects we use
    #fig.colorbar(segments, cmap='plasma', norm=norm)
#Solve Heat eqn ove 2 < x < 15
#Wtih BCs u(2,t) = u(15,t) = 0
#and IC u(x,0) =0;
#This code does a convergence test in space
#Params for the problem
a=0
b=.65
d=.13
h=.05
mu = 5.78e-3
T_s = 66.3398
v = (T_s/mu)**(1/2)
ln = b-a
n_harm = 1
T = .6 #Final time to run to
#Make dt small in spatial convergence test so that time error
#doesn't pollute convergence
dt = 1e-4
#Exact solution
def sum (xl, tl, n):
    return ((2*h*L**2)/(pi**2*n**2*d*(L-d))*sym.sin((n*pi*d)/L))*sym.sin((n*pi*xl)/L)*sym.cos((n*pi*v*tl)/L)

def uex (xl, tl):
    u = 0
    for i in range(1,n_harm+1):
        u += sum(xl,tl,i)
    return (u)
#Initial condition
def eta(xl):
    return uex(xl, 0)
#Determine RHS using symbolic differentiation
t = sym.Symbol('T')
x = sym.Symbol('X')
uexa = sym.sin(t*sym.sin(6*pi*(x-a)/ln))
#RHS symbolic function
f = sym.diff(uexa, t) - (v**2)*sym.diff(sym.diff(uexa,x), x)
#Making rhs function callable
func = sym.lambdify((x, t), f)
def rhs_build(nl, xl, dx, tl):
    #Build RHS vector, g, by looping through each element
    b = np.zeros(nl-1)
    for jjj in range(nl-1):
        #--RHS (be very careful with indexing here!)
        #Define x from j-1 to j
        x_lft = np.arange(xl + (jjj) * dx, xl + (jjj + 1) * dx+0.1*dx, dx)
        #Define phij from j-1 to j (upward sloping part of phij)
        phij_lft = (1/dx)*(x_lft-xl-(jjj)*dx);
        #inner product contribution from j-1 to j:
        #Use trapz or any other quadrature rule
        b[jjj] = np.trapz( func(x_lft, tl)*phij_lft, x_lft)
        #Define x from j to j+1 (downward sloping part of phij)
        x_rgt = np.arange(xl+(jjj+1)*dx, xl+(jjj+2)*dx+ 0.1*dx, dx)
        #phij from j to j+1
        phij_rgt = (-1/dx)*(x_rgt-xl-(jjj+2)*dx);
        #Inner product contribution from j-1 to j
        b[jjj] = b[jjj] + np.trapz( func(x_rgt, tl)*phij_rgt, x_rgt)
    return b
#n values to be used
nvect = np.array([100])
dxvect = ln/nvect
#Initialize Error vector
err = np.zeros(len(nvect))
tvect = np.arange(dt, T+0.1*dt, dt)
nt = len(tvect)
# Snapshots to save
nsnaps = 100;
ind = max(1, np.floor(nt/nsnaps))
print("ind: ", ind)
tsv = np.linspace(dt, T, nsnaps)
ntsv = len(tsv)
# print("shape of tsv: ", np.shape(tsv), "length: ", len(tsv))
# print("tsv: ", tsv)

n = nvect[0] #n is the number of intervals.
#Array that saves values of u to be used in waterfall plot
u = np.zeros((n - 1, ntsv))
# up = np.zeros((n - 1, ntsv))
#Constructing grid
dx = ln/n;
xj = np.linspace(a, b, n+1)
print(xj)
# print(len(xj[1:-1]))
#Build L (matrix associated with u'' term):
L = (v**2)*(1/dx)*(-2*np.diag(np.ones(n-1), k=0) +np.diag(np.ones(n-2), k=-1) +np.diag(np.ones(n-2), k=1))
#Build N (matrix associated with u term):
N = dx*((2/3)*np.diag(np.ones(n-1), k=0) + (1/6)*np.diag(np.ones(n-2), k=-1) + (1/6)*np.diag(np.ones(n-2), k=1))
E = LA.solve(N, L)
#Build identity matrix of the same size as A
I = np.eye(n-1)
#Initialize for time stepping

def init(x):
    if(x < d):
        return((h*x)/d)
    else:
        return((h*(b-x))/(b-d))
uk = xj[1:-1]
for i in range(n-3):
    uk[i] = init(uk[i])
tk = 0
cnt = 0
#Do time stepping:
for jj in range(len(tvect)):
    tkp1 = tk+dt;
    gk = rhs_build(n, a, dx, tk)
    gkp1= rhs_build(n, a, dx, tkp1)
    f = LA.solve(N, L.dot(uk)+gk+gkp1)
    #Update solution at next time using trap method:
    ukp1 = LA.solve(I-0.5*dt*E, uk +0.5*dt*f)
    uk = ukp1;
    tk = tkp1;
    #Save u for finest grid:
    if ((tkp1>tsv[cnt])):
        u[:, cnt] = uk;
        # up[:,cnt] = uex(xj[1:-1], tkp1)
        cnt = cnt+1
#Error:
# err = LA.norm(ukp1 - uex(xj[1:-1], T))/LA.norm(uex(xj[1:-1], T))
# print("Error: ", err, "n: ", n)
# print("Cnt: ", cnt)
print(xj)
# Create waterfall plot
X, Tv = np.meshgrid(xj[1:-1], tsv)
np.savetxt('x.txt',X)
print("Shapes of X: ", np.shape(X), ", T: ", np.shape(Tv), ",u: ", np.shape(u))
plt.rcParams['font.size'] = 15
fig = plt.figure(figsize=(8, 15))
ax = fig.add_subplot(121, projection='3d')
waterfall_plot(fig, ax, X, Tv, u.transpose())
ax.set_xlabel('$x$');
ax.set_xlim3d(a, b)
ax.set_ylabel('$t$')
ax.set_ylim3d(tsv[0],tsv[ntsv-1])
ax.set_xticks(np.arange(0, .66, .13))
ax.set_yticks(np.arange(0, .6, 2))
ax.set_zlabel('$u$'); # ax.set_zlim3d(np.min(u),np.max(u))
ax.set_zlim3d(-1, 1)
ax.set_zticks(np.array([-1, 0, 1]))
plt.title('$u_{FD}$')
# ax = fig.add_subplot(122, projection='3d')
# waterfall_plot(fig, ax, X, Tv, up.transpose())
# ax.set_xlabel('$x$');
# ax.set_xlim3d(a, b)
# ax.set_ylabel('$t$'); # ax.set_ylim3d(tsv[0],tsv[ntsv-1])
# ax.set_xticks(np.arange(0, 15.1, 5))
# ax.set_yticks(np.arange(0, 6.1, 2))
# ax.set_zlabel('$u$'); # ax.set_zlim3d(np.min(u),np.max(u))
# ax.set_zlim3d(-1, 1)
# ax.set_zticks(np.array([-1, 0, 1]))
# plt.title('$u_{exact}$')
# plt.subplots_adjust(left=0.02, right=0.98, bottom=0.02, top=0.98, wspace=0.25)
plt.show()
# plt.savefig("Q4_waterfall.png", bbox_inches='tight', dpi=200)
# plt.clf()
# plt.rcParams['font.size'] = 20
# plt.figure(figsize=(8, 10))
####Error plot
# Plot dt^2 to show that error scales correctly
# n2 = np.shape(nvect)[0] - 1
# c1 = err[n2];
# c2 = c1 * ((dxvect[0] / dxvect[n2]) ** 2)
# xtemp = np.array([dxvect[n2], dxvect[0]]);
# carr = np.array([c1, c2])
# plt.loglog(xtemp, carr, linewidth=4, linestyle='--', color='k', label='$O(\Delta x^2)$')
# # Plot error
# plt.scatter(dxvect, err, s=100, color='b', label='$||e||_2$')
# plt.legend()
# plt.xlabel('$\Delta x$');
# plt.ylabel(r'$||e||_2$')
# # plt.xlim(0, nvect[len(nvect)-1]+1)
# plt.xlim(1e-2, 1e0)
# plt.gca().set_yscale('log');
# plt.gca().set_xscale('log')
# # xtick_arr = np.array([10, 100])
# # plt.xticks(xtick_arr)
# # plt.tick_params(axis="both")
# # plt.tick_params(right=True, top=True)
# plt.savefig('q4_error.png', bbox_inches='tight', dpi=100)

