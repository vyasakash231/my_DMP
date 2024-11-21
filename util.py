from math import *
import numpy as np

import numpy as np


def rk4_step(Y, A, B, dt):
    def f(Y):
        return A @ Y + B

    k1 = f(Y)
    k2 = f(Y + 0.5 * dt * k1)
    k3 = f(Y + 0.5 * dt * k2)
    k4 = f(Y + dt * k3)
    
    dY = (1 / 6) * (k1 + 2*k2 + 2*k3 + k4)
    return dY

'''%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%'''

def forward_euler(Y, A, B, dt):
    def f(Y):
        return A @ Y + B

    return f(Y)

'''%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%'''

def weight_Func(m,n,q_range,q,epsilon):
    const = 400
    We = np.zeros((m,m))
    for i in range(0,m):
        We[i,i] = 50

    Wc = np.zeros((n,n))
    for i in range(0,n):
        if q[0,i] < q_range[i,0]:
            Wc[i,i] = const
        elif q_range[i,0] <= q[0,i] <= (q_range[i,0] + epsilon[0,i]):
            Wc[i,i] = (const/2)*(1 + cos(pi*((q[0,i] - q_range[i,0])/epsilon[0,i])))
        elif (q_range[i,0] + epsilon[0,i]) < q[0,i] < (q_range[i,1] - epsilon[0,i]):
            Wc[i,i] = 0
        elif (q_range[i,1] - epsilon[0,i]) <= q[0,i] <= q_range[i,1]:
            Wc[i,i] = (const/2)*(1 + cos(pi*((q_range[i,1] - q[0,i])/epsilon[0,i])))
        else:
            Wc[i,i] = const

    Wv = np.zeros((n,n))
    for i in range(0,n):
        Wv[i,i] = 0.5
    return We, Wc, Wv

'''%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%'''

def cost_func(n,K,q,q_range,m):
    # Initiate
    c = np.zeros((n,))
    b = np.zeros((n,))
    del_phi_del_q = np.zeros((n,))
    q_c = np.mean(q_range,axis = 1); # column vector containing the mean of each row
    del_q = q_range[:,1] - q_range[:,0]; # Total working range of each joint

    for i in range(0,n):
        if q[0,i] >= q_c[i]:
            c[i] = pow((K[i,i]*((q[0,i] - q_c[i])/del_q[i])),m)
            b[i] = pow((K[i,i]*((q[0,i] - q_c[i])/del_q[i])),m-1)
        elif q[0,i] < q_c[i]:
            c[i] = pow((K[i,i]*((q_c[i] - q[0,i])/del_q[i])),m)
            b[i] = pow((K[i,i]*((q_c[i] - q[0,i])/del_q[i])),(m-1))

    L = np.sum(c)

    for j in range(0,n):
        if q[0,j] >= q_c[j]:
            del_phi_del_q[j] = pow(L,((1-m)/m))*b[j]*(K[j,j]/del_q[j])
        elif q[0,j] < q_c[j]:
            del_phi_del_q[j] = -pow(L,((1-m)/m))*b[j]*(K[j,j]/del_q[j])

    v = -del_phi_del_q
    return v