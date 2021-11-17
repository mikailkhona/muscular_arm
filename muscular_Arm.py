"""
Created on Thu Jul 18 12:25:40 2019

@author: Hari - hariteja1992@gmail.com
@author: Mikail - mikailkhona@gmail.com (modifications)

This code implements the state equations for a 2-DOF planar arm with
6 muscle actuators and non-linearities.

In the tensor representation, the 3 dimensions represent :
time tensor dimension (T) - 1
sample number tensor dimension (C) - 2
features dimension (N) - 3

MOST IMPORTANT NOTE -
Do not ever perform tensor assignment in the middle of 'forward' method.
because torch.tensor does not remember or retain the graph unlike torch.cat.
This results in non-computation of gradient memory.

for example the wrong code is as follows -
h1 = ((-theta2_dot) * ((2*theta1_dot) + theta2_dot) * (self.a2 * torch.sin(theta2))) + (self.b11*theta1_dot) + (self.b12*theta2_dot)
        h2 = ((theta1_dot**2) * self.a2 * torch.sin(theta2)) + (self.b21*theta1_dot) + (self.b22*theta2_dot)
H = torch.tensor([[h1], [h2]])

above tensor assignment to H eliminates the gradient graph.

instead you should write the H assignment as

H = torch.cat((h1.unsqueeze(0), h2.unsqueeze(0)), 0)
This retains the grad_fn during the backprop
"""

import torch
use_cuda = 'false'
device = torch.device('cuda:0' if use_cuda else 'cpu')
device = 'cpu'
pi = torch.tensor(3.14)
class muscular_arm():
    def __init__(self, batch_size, device, dt = 0.01):
        super(muscular_arm, self).__init__()
        self.batch_size = batch_size

        # fixed Monkey arm parameters (1=shoulder; 2=elbow) (refer to Lillicrap et al 2013, Li&Todorov2007)
        self.i1 = torch.tensor([0.025]).to(device) # kg*m**2 shoulder inertia
        self.i2 = torch.tensor([0.045]).to(device) # kg*m**2 elbow inertia
        self.m1 = torch.tensor([0.2108]).to(device) # kg mass of shopulder link
        self.m2 = torch.tensor([0.1938]).to(device) # kg mass of elbow link
        self.l1 = torch.tensor([0.145]).to(device) # meter
        self.l2 = torch.tensor([0.284]).to(device) # meter
        self.s1 = torch.tensor([0.0749]).to(device)
        self.s2 = torch.tensor([0.0757]).to(device)

        # fixed joint-friction
        self.b11 = torch.tensor([0.5]).to(device)
        self.b22 = torch.tensor([0.5]).to(device)
        self.b21 = torch.tensor([0.1]).to(device)
        self.b12 = torch.tensor([0.1]).to(device)

        # inertial matrix tmp vars
        self.a1 = (self.i1 + self.i2) + (self.m2 * self.l1**2)
        self.a2 = self.m2 * self.l1 * self.s2
        self.a3 = self.i2

        # Moment arm param
        self.M = torch.tensor([[2.0, -2.0, 0.0, 0.0, 1.50, -2.0], [0.0, 0.0, 2.0, -2.0, 2.0, -1.50]]).to(device)

        # Muscle properties 
        #Angles to radians conversion
        self.theta0 = 0.01745*torch.tensor([[15.0, 4.88, 0.00, 0.00, 4.5, 2.12], [0.00, 0.00, 80.86, 109.32, 92.96, 91.52]]).to(device)
        self.L0 = torch.tensor([[7.32, 3.26, 6.4, 4.26, 5.95, 4.04]]).to(device)
        self.beta = 1.55
        self.omega = 0.81
        self.rho = 2.12
        self.Vmax = -7.39
        self.cv0 = -3.21
        self.cv1 = 4.17
        self.bv = 0.62
        self.av0 = -3.12
        self.av1 = 4.21
        self.av2 = -2.67

        # time-step of dynamics
        self.dt = dt
        
        self.cur_j_state = torch.zeros(batch_size, 4).to(device)
        self.FV = torch.zeros(batch_size, 6).to(device)

    def step(self, u):
        """
        Takes in muscle layer force inputs
        u (muscle input raw) shape:
        returns: nothing (Default), uncomment last line to return (theta,phi,thetadot,phidot)
        updates joint states using muscle output(incorporating length/velocity) -> torque input
        """

        # for linear muscle activation
        mus_inp = u
        # for non-linear muscle activation - add F-L/V property contribution
        fl_out, fv_out = self.muscleDyn()
        flv_computed = fl_out * fv_out
        mus_out = fl_out * fv_out * mus_inp
        #mus_out = mus_inp
        #muscle-force to joint-torque transformation (using M)
        self.tor = torch.mm(self.M, mus_out.transpose(0,1))
        # add external torque to the muscle generated torque
        self.tor = self.tor.transpose(0,1)
        # run the arm dynamics 
        net_command = self.tor
        x = self.armdyn(net_command)
        self.cur_j_state = x
        
        #constrain movement of arm to realistic configurations
        '''
        Upper bound angles
        self.cur_j_state[:,1] = pi - torch.relu(pi - self.cur_j_state[:,1])
        self.cur_j_state[:,1] = self.elbow_ub - torch.relu(self.elbow_ub - self.cur_j_state[:,1])
        
         Lower bound angles
        self.cur_j_state[:,1] = torch.relu(self.cur_j_state[:,1] + 80) - 80
        self.cur_j_state[:,0] = torch.relu(self.cur_j_state[:,0])
        '''
        #(Optional) compute cartesian-states from joint-states (Run armkinematics)
        #y = self.armkin(x)
        
        #return self.cur_j_state

    def armdyn(self, u):
        '''
        Helper function, used by step
        Takes in muscle layer torque outputs, updates theta, thetadot 
        return (theta,phi,thetadot,phidot)
        '''

        # extract joint angle states
        theta1 = self.cur_j_state[:, 0].clone().unsqueeze(1)
        theta2 = self.cur_j_state[:, 1].clone().unsqueeze(1)
        theta1_dot = self.cur_j_state[:, 2].clone().unsqueeze(1)
        theta2_dot = self.cur_j_state[:, 3].clone().unsqueeze(1)

        # compute inertia matrix
        I11 = self.a1 + (2*self.a2*(torch.cos(theta2)))
        I12 = self.a3 + (self.a2*(torch.cos(theta2)))
        I21 = self.a3 + (self.a2*(torch.cos(theta2)))
        I22 = self.a3
        I22 = I22.repeat(self.batch_size,1)

        # compute determinant of mass matrix 
        det = (I11 * I22) - (I12 * I12)

        # compute Inverse of inertia matrix
        Irow1 = torch.cat((I22, -I12), 1)
        Irow2 = torch.cat((-I21, I11), 1)

        Irow1 = (1/det) * Irow1
        Irow2 = (1/det) * Irow2
        # terms of Iinv matrix
        Iinv_11 = Irow1[:, 0].unsqueeze(1)
        Iinv_12 = Irow1[:, 1].unsqueeze(1)
        Iinv_21 = Irow2[:, 0].unsqueeze(1)
        Iinv_22 = Irow2[:, 1].unsqueeze(1)

        # compute extra torque H (coriolis, centripetal, friction)
        h1 = ((-theta2_dot) * ((2*theta1_dot) + theta2_dot) * (self.a2 * torch.sin(theta2))) + (self.b11*theta1_dot) + (self.b12*theta2_dot)
        h2 = ((theta1_dot**2) * self.a2 * torch.sin(theta2)) + (self.b21*theta1_dot) + (self.b22*theta2_dot)

        H = torch.cat((h1, h2), 1)

        # compute xdot = inv(M) * (u - H)
        torque = u - H
        
        # determione the terms in xdot matrix; xdot = [[dq1], [dq2], [ddq1], [ddq2]]
        dq1 = theta1_dot
        dq2 = theta2_dot
        dq = torch.cat((dq1, dq2), 1)

        # Update acceleration of shoulder and elbow joints - FWDDYN equations
        ddq1 = Iinv_11*torque[:, 0].unsqueeze(1) + Iinv_12*torque[:, 1].unsqueeze(1)
        ddq2 = Iinv_21*torque[:, 0].unsqueeze(1) + Iinv_22*torque[:, 1].unsqueeze(1)
        ddq = torch.cat((ddq1, ddq2), 1)

        # update xdot
        theta_dot = torch.cat((dq, ddq), 1)

        # step-update from x to x_next
        theta_next = self.cur_j_state + (self.dt * theta_dot)

        #out = x[:, 0:2]
        # above transposing is done to rearrange the state and output in a column
        # as is demanded by the tensor form in which we wrote our optimization code
        return theta_next

    def get_tipPosition(self):
        '''
        get position/velocity in (x,y,xdot,ydot) coordinates based on (theta, thetadot) joint coordinates
        '''
        
        theta1 = self.cur_j_state[:, 0].clone().unsqueeze(1)
        theta2 = self.cur_j_state[:, 1].clone().unsqueeze(1)
        theta1_dot = self.cur_j_state[:, 2].clone().unsqueeze(1)
        theta2_dot = self.cur_j_state[:, 3].clone().unsqueeze(1)

        x = (self.l1 * torch.cos(theta1)) + (self.l2 * torch.cos(theta1+theta2))
        y = (self.l1*torch.sin(theta1)) + (self.l2*torch.sin(theta1+theta2))
        
        xdot = - theta1_dot*((self.l1*torch.sin(theta1)) + (self.l2*torch.sin(theta1+theta2)))
        xdot = xdot - (theta2_dot*(self.l2*torch.sin(theta1+theta2)))
        
        ydot = theta1_dot*((self.l1*torch.cos(theta1)) + (self.l2*torch.cos(theta1+theta2)))
        ydot = ydot + (theta2_dot*(self.l2*torch.cos(theta1+theta2)))
        
        phase_space = torch.cat((x,y,xdot,ydot), 1)
        return phase_space

    def muscleDyn(self):
        '''
        Updates force-load and force-velocity states of muscles
        '''
        
        # F-L/V dependency
        mus_l = 1 + self.M[0,:] * (self.theta0[0,:] - self.cur_j_state[:, 0].unsqueeze(1))/self.L0 + self.M[1,:] * (self.theta0[1,:] - self.cur_j_state[:, 1].unsqueeze(1))/self.L0
        mus_l = torch.relu(mus_l)
        #If any of these are 0, the factor should be 0, otherwise 1
        if (torch.count_nonzero(mus_l)<6):
            factor = 0
        else:
            factor = 1
        mus_v = self.M[0, :] * self.cur_j_state[:, 2].unsqueeze(1)/self.L0 + self.M[1, :] * self.cur_j_state[:, 3].unsqueeze(1)/self.L0
        FL = factor*torch.exp(-torch.abs(((mus_l)**self.beta - 1)/self.omega)**self.rho)
        FV = self.FV.clone()

        #loop over number of muscles
        for i in range(0, 6):
            vel_i = mus_v[:, i]
            len_i = mus_l[:,i]
            FV[vel_i<=0, i] = (self.Vmax - vel_i[vel_i<=0])/(self.Vmax + vel_i[vel_i<=0]*(self.cv0 + self.cv1*len_i[vel_i<=0]))
            FV[vel_i >0, i] = (self.bv - vel_i[vel_i>0]*(self.av0+self.av1*len_i[vel_i>0]+self.av2*len_i[vel_i>0]**2))/(self.bv + vel_i[vel_i>0])

        return FL, FV
    
    def joint_Coords(self):
        '''
        Helper function, used by step
        Get (x,y) of each joint
        Used to draw the current configuration of the arm.
        In numpy for visualization.
        '''
        
        joint_Coords = np.array([[0,0],[arm.l1*np.cos(arm.cur_j_state[0,0].detach().numpy()),
                               arm.l1*np.sin(arm.cur_j_state[0,0].detach().numpy())],
                        [arm.l1*np.cos(arm.cur_j_state[0,0].detach().numpy()) + arm.l2*np.cos(arm.cur_j_state[0,0].detach().numpy() + arm.cur_j_state[0,1].detach().numpy()),
                               arm.l1*np.sin(arm.cur_j_state[0,0].detach().numpy()) + arm.l2*np.sin(arm.cur_j_state[0,0].detach().numpy() + arm.cur_j_state[0,1].detach().numpy())]])
        return joint_Coords
