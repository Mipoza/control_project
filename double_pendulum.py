import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp
from sympy import *

x, y = symbols('x y')

#u=[x1,x2,x3] and v=[y1,y2,y3]
def dot(u,v):
    return u[0]*v[0] + u[1]*v[1] + u[2]*v[2]

#Differentiate a vector-valued function
def diffv(f, s):
    return [diff(f[0], s), diff(f[1], s) ,diff(f[2], s)]

#Get the metric and its inverse from the parametrization
def metric(f):
    global x
    global y

    f_x = diffv(f, x)
    f_y = diffv(f, y)   
    g = [[simplify(dot(f_x,f_x)),simplify(dot(f_x,f_y))],[simplify(dot(f_x,f_y)),simplify(dot(f_y,f_y))]]
    det = g[0][0]*g[1][1] - g[0][1]**2
    inv_g = [[simplify(g[1][1]/det), simplify(- g[0][1]/det)],[simplify(- g[0][1]/det), simplify(g[0][0]/det)]] 
    return [g, inv_g]

#Get the christoffel symbols from the metric and its inverse
def christoffel(g, inv_g):
    global x
    global y

    gamma = [[],[]] # gamma[0]=[gamma^x_xx, gamma^x_xy, gamma^x_yy] and gamma[1]=[gamma^y_xx, gamma^y_xy, gamma^y_yy]

    for k in range(0,2):
        for l in range(0,3):
            i = x
            j = x
            if l >= 1:
                j = y
            if l == 2:
                i = y
            
            c = lambda a : 0 if a == x else 1
            cv = lambda a : x if a == 0 else y

            s = 0
            for m in range(0,2):
                s += inv_g[k][m]*(diff(g[c(j)][m], i)+diff(g[c(i)][m], j) - diff(g[c(i)][c(j)], cv(m)))

            gamma[k].append(simplify(0.5*s))

    return gamma

def F_gamma_u(X,gamma,U,inv_g):
    global x
    global y
    Y=[]

    for k in range(2,4):
        s = 0
        for i in range(0,2):
            for j in range(0,2):
                if i == 0:
                    s += X[i+2]*X[j+2]*gamma[k-2][j].subs([(x, X[0]), (y, X[1])]).evalf()
                else:
                    s += X[i+2]*X[j+2]*gamma[k-2][j+1].subs([(x, X[0]), (y, X[1])]).evalf()
        v=0
        v = (inv_g[k-2][0]*diff(U, x)+ inv_g[k-2][1]*diff(U,y)).subs([(x, X[0]), (y, X[1])]).evalf()
        #print(v)
        Y.append(-s-v)
    return [X[2],X[3],Y[0],Y[1]]



#Parameterization of the manifold as an embedded submanifold of R^3
#WARNING : Remember that a parametrization f need to be an immersion i.e. df never vanishes, if not it is possible that g^-1 diverge !
f = [(1+cos(y))*cos(x),(1+cos(y))*sin(x),sin(y)] #f : U -> R^3 with U an open subset of R^2

f_l = [lambdify([x,y], f[0]), lambdify([x,y], f[1]), lambdify([x,y], f[2])] #Converting f to a "numerical" function and not a formal one

T = 5 #Time horizon
D_f = 2*3.1415 #[-D_f,D_f]^2 is a square domain center at 0 where f is defined
X_0 = [1.57,0.2,0,0] #initial condition as [x_1,x_2,dx_1/dt,dx_2/dt] 

# g, inv_g = metric(f)

g = [[2,cos(x-y)],[cos(x-y),1]] #metric for the double pendulum
det = g[0][0]*g[1][1] - g[0][1]**2
inv_g = [[simplify(g[1][1]/det), simplify(- g[0][1]/det)],[simplify(- g[0][1]/det), simplify(g[0][0]/det)]] 

gamma = christoffel(g, inv_g)

U = 2*cos(x) + cos(y) #potential
F = lambda t, X : F_gamma_u(X,gamma,U,inv_g)

t_eval = np.arange(0, T, 0.01) #if x_1=x_2=0 be aware if not and in general to not leave the domain of f
sol = solve_ivp(F, [0, T], X_0, t_eval=t_eval)

theta_1 = sol.y[0]
theta_2 = sol.y[1]
x_1 = -np.sin(theta_1)
y_1 = np.cos(theta_1)
x_2=-np.sin(theta_1)-np.sin(theta_2)
y_2=np.cos(theta_2)+np.cos(theta_1)

fig, ax = plt.subplots()
ax.set_aspect('equal')
plt.xlim(-3,3)
plt.ylim(-3,3)
line1, = ax.plot([], [], marker='o', color='red')
line2, = ax.plot([], [], marker='o', color='blue')
ax.legend()

ax.axhline(0, color='black',linewidth=0.5)
ax.axvline(0, color='black',linewidth=0.5)

line_origin_to_point1, = ax.plot([], [], color='red', linestyle='-', linewidth=2)
line_point1_to_point2, = ax.plot([], [], color='red', linestyle='-', linewidth=2)
line_trace_of_point2, = ax.plot([], [], color='blue', linestyle='-', linewidth=2)


# Function to initialize the plot
def init():
    line1.set_data([], [])
    line2.set_data([], [])
    line_origin_to_point1.set_data([], [])
    line_point1_to_point2.set_data([], [])
    line_trace_of_point2.set_data([], [])
    return line1, line2, line_origin_to_point1, line_point1_to_point2, line_trace_of_point2

# Function to update the positions of the points in each frame
def update(frame):
    x1_val = x_1[frame]
    y1_val = y_1[frame]
    x2_val = x_2[frame]
    y2_val = y_2[frame]
    #print(sqrt(x1_val**2 + y1_val**2))
    line1.set_data(x1_val, y1_val)
    line2.set_data(x2_val, y2_val)

    line_origin_to_point1.set_data([0, x1_val], [0, y1_val])
    line_point1_to_point2.set_data([x1_val, x2_val], [y1_val, y2_val])
    line_trace_of_point2.set_data(x_2[:frame+1], y_2[:frame+1])
    
    return line1, line2, line_origin_to_point1, line_point1_to_point2, line_trace_of_point2

# Create the animation with a delay of 0.01 seconds between frames
frame_delay = 0.01  # in seconds
num_frames = len(x_1)

animation = FuncAnimation(
    fig, update, frames=range(num_frames), init_func=init, blit=True, repeat=False, interval=frame_delay * 1000
)

# Display the animation
plt.show()


#Now we want to display the path of the double pendulum on his configuration manifolds (torus)
path = np.array([f_l[0](sol.y[0],sol.y[1]),  f_l[1](sol.y[0],sol.y[1]), f_l[2](sol.y[0],sol.y[1])])

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.set_box_aspect([1,1,1])

X = path[0]
Y = path[1]
Z = path[2]

ax.plot(X, Y, Z, '-r', linewidth = 3)

U,V = np.meshgrid(np.linspace(-D_f, D_f, 30),np.linspace(-D_f, D_f, 30))

X = f_l[0](U,V)
Y = f_l[1](U,V)
Z = f_l[2](U,V)

ax.plot_surface(X, Y, Z, rstride=1, cstride=1, alpha=0.2, cmap='viridis', edgecolor='none') 

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

ax.set_xlim3d(-D_f/2,D_f/2)
ax.set_ylim3d(-D_f/2,D_f/2)
ax.set_zlim3d(-D_f/2,D_f/2)


plt.show()