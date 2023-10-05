import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp
from sympy import *

T = 20 #Time horizon
X_0 = [0.1,0.05,0,-0.1] #initial conditions

def u(t, X):
    return -10*X[0]+19/2*X[1]-4*X[2]+6*X[3]

def F(t, Xu): #dynamic, Xu = [\theta_1, \theta_2, \dot{\theta}_1, \dot{\theta}_2, x, \dot{x}] where x is the chariot position
    u_val = u(t, Xu[0:4])

    Y = [Xu[2],Xu[3],0,0,Xu[5],u_val]

    c_1 = cos(Xu[0])
    c_2 = cos(Xu[1])
    s_1 = sin(Xu[0])
    s_2 = sin(Xu[1])
    c_12 = cos(Xu[0]-Xu[1])
    s_12 = sin(Xu[0]-Xu[1])

    Y[2] = 1/(1-0.5*c_12**2) * (u_val * c_1 + s_1 - 0.5*s_12*Xu[3]**2 - 0.5*c_12*(u_val*c_2 + s_2 + s_12*Xu[2]**2))
    Y[3] = 1/(1-0.5*c_12**2) * (u_val * c_2 + s_2 + s_12*Xu[2]**2 - c_12*(u_val*c_1 + s_1 - 0.5*s_12*Xu[3]**2))

    return Y

t_eval = np.arange(0, T, 0.01)
sol = solve_ivp(F, [0, T], X_0 + [0,0], t_eval=t_eval)


theta_1 = sol.y[0]
theta_2 = sol.y[1]
x = sol.y[4]
x_1 = x-np.sin(theta_1)
y_1 = np.cos(theta_1)
x_2=x-np.sin(theta_1)-np.sin(theta_2)
y_2=np.cos(theta_2)+np.cos(theta_1)

fig, ax = plt.subplots()
ax.set_aspect('equal')
plt.xlim(-4,4)
plt.ylim(-4,4)
line1, = ax.plot([], [], marker='o', color='red')
line2, = ax.plot([], [], marker='o', color='blue')
line3, = ax.plot([], [], marker='s', color='red')
ax.legend()

ax.axhline(0, color='black',linewidth=0.5)

line_origin_to_point1, = ax.plot([], [], color='red', linestyle='-', linewidth=2)
line_point1_to_point2, = ax.plot([], [], color='red', linestyle='-', linewidth=2)
line_trace_of_point2, = ax.plot([], [], color='blue', linestyle='-', linewidth=2)


# Function to initialize the plot
def init():
    line1.set_data([], [])
    line2.set_data([], [])
    line3.set_data([], [])
    line_origin_to_point1.set_data([], [])
    line_point1_to_point2.set_data([], [])
    line_trace_of_point2.set_data([], [])
    return line1, line2, line3, line_origin_to_point1, line_point1_to_point2, line_trace_of_point2

# Function to update the positions of the points in each frame
def update(frame):
    x_val = x[frame]
    x1_val = x_1[frame]
    y1_val = y_1[frame]
    x2_val = x_2[frame]
    y2_val = y_2[frame]
    
    line1.set_data(x1_val, y1_val)
    line2.set_data(x2_val, y2_val)
    line3.set_data(x_val,0)

    line_origin_to_point1.set_data([x_val, x1_val], [0, y1_val])
    line_point1_to_point2.set_data([x1_val, x2_val], [y1_val, y2_val])
    line_trace_of_point2.set_data(x_2[:frame+1], y_2[:frame+1])
    
    return line1, line2, line3, line_origin_to_point1, line_point1_to_point2, line_trace_of_point2

# Create the animation with a delay of 0.01 seconds between frames
frame_delay = 0.01  # in seconds
num_frames = len(x_1)

animation = FuncAnimation(
    fig, update, frames=range(num_frames), init_func=init, blit=True, repeat=False, interval=frame_delay * 1000
)

# Display the animation
plt.show()