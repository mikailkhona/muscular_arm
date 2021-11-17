import numpy as np
import torch
import matplotlib.pyplot as plt
from muscular_Arm import muscular_arm

arm = muscular_arm(batch_size = 1, device = 'cpu')
arm.beta = 1.

steps = 400
#random network input (muscle layer)
u = torch.tensor([0,0,1,0,1,0])
x = np.zeros((steps,2))
thetas = np.zeros((steps,2))

joint_Coords_init = arm.joint_Coords()

for i in range(0,steps):
    arm.step(u)
    x[i,:] = arm.get_tipPosition()[0,0:2].detach().numpy()
    thetas[i,:] = arm.cur_j_state[0,0:2].detach().numpy()

fig, axes = plt.subplots(nrows = 1, ncols = 3,figsize = (4.5,12))
plt.subplot(311)
plt.scatter(x[:,0],x[:,1])
joint_Coords = arm.joint_Coords()
plt.plot(joint_Coords_init[:,0], joint_Coords_init[:,1], color = 'gray', label = 'arm at t=0')
plt.plot(joint_Coords[:,0], joint_Coords[:,1],color = 'black',label = 'arm at end')
plt.legend()
plt.xlim(-0.5,0.5)
plt.ylim(-0.5,0.5)
plt.title('drawing')

plt.subplot(312)
plt.plot(x[:,0])
plt.plot(x[:,1])
plt.xticks([])
plt.title('x(t) and y(t)')

plt.subplot(313)
plt.plot(thetas[:,0]*180/np.pi, label = 'shoulder')
plt.plot(thetas[:,1]*180/np.pi,label ='elbow')
plt.legend()

plt.title('theta(t) and phi(t)')
