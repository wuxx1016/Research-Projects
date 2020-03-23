# -*- coding: utf-8 -*-
"""
Created on Wed Feb  6 18:38:18 2019

@author: wuxx1
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


# Assume I know the current S, I can generate 1000 S' , S_t = (Ptilde_t,Dtilde_t,P_t,D_t)
# Ptilde_t is the demanded weighted price at time t, Ptilde_2 = (p_0*d_0 + p_1*d_1+p_2*d_2)/(d_0+d_1+d_2)
# dtilde_t is the accumulated quantities at time t, Dtilde_2 = d_0+d_1+d_2
# P_t is the electricity spot price at time t
# Suppose the at time t=2, I have P_0=8, P_1=9, P_2=8, d_0=2. d_1=1, d_2=4, D_2=10, so Ptilde_2=8.1429, Dtilde=7
# S_2 = (8.1429, 7, 8, 10), suppose my action spaces = {0,1,...,14} i.e. 10+(4-2)*2 = 14, suppose P +-Pdelta with 1/2 prob
# Same assumptions apply to D+-Ddelta with 1/2 prob


# I will start with the boundray conditions first, so now I will work on t=3, P_3=7,d_3=3, D_3=11, so Ptilde_3 = 7.8000
# Dtilde = 10, i.e. S_3 = (7.8000,10,7,11). Based on the S_3, I am going to generate 1000 samples of my terminal states with (P_4,D_4)
# Since at the expiration date, the contract has to be realized and I have to match the real demand
# T = N = 4, N-1 = 3

Ptilde = np.zeros(4,)
Dtilde = np.zeros(4,)
P = np.zeros(4,)
D = np.zeros(4,)
S = np.zeros((4,4))
A = np.zeros(4,)

Ptilde[3]=7.8
Dtilde[3]=10
P[3]=7
D[3]=10
S[3] = [Ptilde[3], Dtilde[3],P[3], D[3]]
A[3] = 3

N = 5000
# Generate 1000 S_4 based on my S_3
price_change =[1,-1]
price_probabilities =[0.5,0.5]
P4 = np.zeros(N,)
P4 = P[3]+np.random.choice(price_change,N,p=price_probabilities)

demand_change =[1,-1]
demand_probabilities =[0.5,0.5]
D4 = D[3]+np.random.choice(demand_change,N,p=demand_probabilities)

S4 = np.mat([P4,D4])

P_regulated = 12
P_overhedge = 9

# Calcualted Q_4(S4), all my boundray conditions 
Q4 = np.zeros((len(D4-Dtilde[3]),3))
for i in range(len(D4-Dtilde[3])):
    Q4[i] = [S4[0,i],S4[1,i],P_regulated*D4[i]-Ptilde[3]*Dtilde[3]-P4[i]*max(D4[i]-Dtilde[3],0)-P_overhedge*max(Dtilde[3]-D4[i],0)]
    
# Backpropagation Q-learning
# Actually, I don't need to track the every updates for my Q_{N-1}(S,a).However, for now, I would like to understand the details of updating
# I will first recall a typical Q-learning method
"""
Initialize Q(S,a)
    Repeat (for each episode):
        Initialize S
        Repeat(for each step of episode):
            Choose a from S using policy derived from Q
            (e.g. epsilon-greedy)
            Take action a, Observe r,S'
            Q(S,a) <-- Q(S,a) + alpha*[r + gamma max_a' Q(S',a') - Q(S,a)]
            S <-- S'
            Unitl S is terminal
"""   
# My main function will be Q_{N-1}(S,a) = Q_{N-1}(S,a) + alpha*[0 + gamma* Q_N(S') - Q_{N-1}(S,a)] 

# Stepsize
alpha = 0.1

# Gamma for Q-learning
gamma = 0.9

Q = np.zeros((len(Q4),))
Q[0] = Q[0] + alpha * (0 + gamma*Q4[0,2]-Q[0])
for i in range(len(Q4)-1):
    Q[i+1] = Q[i]+ alpha *(0 + gamma*Q4[i+1,2] - Q[i])
    alpha = 1/(i+1)

if 0: 
# Q3 currently is highly depend on gamma,which I am not sure? 
# Now let us move to calculate Q2 first, S_2 = (8.1429, 7, 8, 10), based on this, I will generate S_3 and S_4
    S2=[8.1429, 7, 8, 10]
    # Let us suppose I will have different P & D samples. then based on different actions (0:16)
    
    #temp = 2*(S2[3]-S2[1])
    A2 = np.random.random_integers(0, temp, N)
    price_change =[1,-1]
    price_probabilities =[0.5,0.5]
    P3 = np.zeros(N,)
    P3 = S2[2]+np.random.choice(price_change,N,p=price_probabilities)
    
    demand_change =[1,-1]
    demand_probabilities =[0.5,0.5]
    D3 = S2[3]+np.random.choice(demand_change,N,p=demand_probabilities)
    
    S3 = np.zeros((N,4))
    for i in range(N):
        S3[i] = [(S2[0]*S2[1]+P3[i]*A2[i])/(S2[1]+A2[i]), S2[1]+A2[i], P3[i], D3[i]]
    
    # For every S3, I will generate 5000 S_4




    
