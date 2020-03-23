# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 21:54:17 2019

@author: wuxx1
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import array as arr
from numpy.linalg import inv
start_time = time.time()

N = 500
M = 1
P_regulated = 12
P_overhedge = 9

S2=[0.0001, 0.0001, 10, 5]
Q2 = np.zeros((M,6))
q3 = np.zeros((M,N,5))
q2 = np.zeros((M,1,5))
# Let us suppose I will have different P & D samples. then based on different actions (0:16)
price_change =[1,-1]
price_probabilities =[0.5,0.5]
PRAND = np.random.choice(price_change,N,p=price_probabilities)
#PRAND = [1,-1,1]
demand_change =[1,-1]
demand_probabilities =[0.5,0.5]
DRAND = np.random.choice(demand_change,N,p=demand_probabilities)
#DRAND = [-1,1,-1]

# Stepsize
alpha1 = 0.1
alpha2 = 0.1    
# Gamma for Q-learning
gamma = 1

#Number of basis function
K = 10

#Basis fuction
Phi4 = np.zeros(K,)
Phi3 = np.zeros(K,)
Phi2 = np.zeros(K,)
#Phit((Ptilde,Dtilde,P_t,D_t),A_t) = [Ptilde, Dtilde,P_t,D_t,A_t, P_t^2,D_t^2,Ptilde*Dtilde,P_t*D_t,A_t*P_t]

#Initilize Amatrix and bmatrix
Atilde3 = np.mat(np.zeros((K,K)))
Atilde2 = np.mat(np.zeros((K,K)))
btilde3 = np.transpose(np.mat(np.zeros(K,)))
btilde2 = np.transpose(np.mat(np.zeros(K,)))


for j in range(len(Q2)):
    istart_time = time.time()
    #temp = 2*(S2[3]-S2[1])
    A2 = int(3)
    #P3 = np.zeros(N,)
    P3 = S2[2]+PRAND
    #for l in range(len(PRAND)):
        #P3[l] = S2[2] + PRAND[l]
    
    #D3 = np.zeros(N,)
    #for l in range(len(DRAND)):
        #D3[l] = S2[3] + DRAND[l]
    D3 = S2[3]+DRAND
    
    S3 = np.zeros((N,4))
    for i in range(N):
        if S2[1]+A2 == 0:
            S3[i] = [0, 0, P3[i], D3[i]]
        else: 
            S3[i] = [(S2[0]*S2[1]+S2[2]*A2)/(S2[1]+A2), S2[1]+A2, P3[i], D3[i]]
        
    
    # For every S3, I will generate 500 S_4
    P4 = np.zeros((len(S3),N))
    D4 = np.zeros((len(S3),N))
    A3 = np.arange(0, 2*(abs(max(S3[:,1]-S3[:,3])))+2,1)
    S4 = np.zeros((len(S3),N,len(A3),4))
    
    for k in range(len(A3)): 
        for l in range(len(S3)):
                
                P4[l,:] = P3[l]+PRAND
                
                D4[l,:] = D3[l]+DRAND
                for i in range(N):
                    if S3[l,1]+A3[k] ==0:
                        S4[l,i,k,:] =[0,0,P4[l,i],D4[l,i]]
                    else:
                        S4[l,i,k,:] =[(S3[l,0]*S3[l,1]+S3[l,2]*A3[k])/(S3[l,1]+A3[k]),S3[l,1]+A3[k],P4[l,i],D4[l,i]]
                  
                    
    # Calcualted Q_4(S4), all my boundray conditions 
    Q4 = np.zeros((len(S3),N,len(A3),6))
    for k in range(len(A3)):
        for l in range(len(S3)):
            for i in range(N):
                Q4[l,i,k,:] = [S4[l,i,k,0],S4[l,i,k,1], S4[l,i,k,2],S4[l,i,k,3], A3[k],P_regulated*D4[l,i]-S4[l,i,k,0]*S4[l,i,k,1]-P4[l,i]*max(D4[l,i]-S4[l,i,k,1],0)-P_overhedge*max(-D4[l,i]+S4[l,i,k,1],0)]
                
    # Calculate Atilde3 and btilde3
#if False:
    #Wtilde3 = np.zeros((len(S3),len(A3),K))
    for k in range(len(A3)):
        for l in range(len(S3)):
            for i in range(N):
                Phi3 = np.transpose(np.mat([S3[i,0],S3[i,1],S3[i,2],S3[i,3],A3[k],S3[i,2]**2,S3[i,3]**2,S3[i,0]*S3[i,1],S3[i,2]*S3[i,3],A3[k]*S3[i,2]]))
                Phi4 = np.transpose(np.mat([S4[l,i,k,0],S4[l,i,k,1],S4[l,i,k,2],S4[l,i,k,3],D4[l,i]-S4[l,i,k,1],S4[l,i,k,2]**2,S4[l,i,k,3]**2,S4[l,i,k,0]*S4[l,i,k,1],S4[l,i,k,2]*S4[l,i,k,3],(D4[l,i]-S4[l,i,k,1])*S4[l,i,k,2]]))
                Atilde3 = Atilde3 + np.matmul(Phi3, np.transpose(Phi3-gamma*Phi4))
                btilde3 = btilde3 + Phi3*Q4[l,i,k,5]
    
    # Compute Wtile3
    Wtilde3 = inv(Atilde3)*btilde3   
    
    #Based on Wtilde3, Compute the approx Q3 values
    Q3 = np.zeros((len(S3),len(A3),6)) 
    for k in range(len(A3)):
        for l in range(len(S3)):
            for i in range(N):
                Phi3 = np.transpose(np.mat([S3[i,0],S3[i,1],S3[i,2],S3[i,3],A3[k],S3[i,2]**2,S3[i,3]**2,S3[i,0]*S3[i,1],S3[i,2]*S3[i,3],A3[k]*S3[i,2]]))
                Q3approx = np.matmul(np.transpose(Wtilde3),Phi3)
                Q3[l,k,:] = [S3[l,0],S3[l,1],S3[l,2],S3[l,3],A3[k],Q3approx[0]]  
#if False:
    for k in range(len(A3)):
        for l in range(len(S3)):
            Phi2 = np.transpose(np.mat([S2[0],S2[1],S2[2],S2[3],A2,S2[2]**2,S2[3]**2,S2[0]*S2[1],S2[2]*S2[3],A2*S2[2]]))
            Phi3 = np.transpose(np.mat([S3[l,0],S3[l,1],S3[l,2],S3[l,3],A3[k],S3[l,2]**2,S3[l,3]**2,S3[l,0]*S3[l,1],S3[l,2]*S3[l,3],A3[k]*S3[l,2]]))
            Atilde2 = Atilde2 + np.matmul(Phi2, np.transpose(Phi2-gamma*Phi3))
            btilde2 = btilde2 + Phi2*Q3[l,k,5]
            
    Wtilde2 = inv(Atilde2)*btilde2
    
    for l in range(N):
        Phi2 = np.transpose(np.mat([S2[0],S2[1],S2[2],S2[3],A2,S2[2]**2,S2[3]**2,S2[0]*S2[1],S2[2]*S2[3],A2*S2[2]]))
        #Q2approx = np.matmul(np.transpose(Wtilde2),Phi2)
        #Q2[j,:] = [S2[0],S2[1],S2[2],S2[3],A2,Q2approx[0]]
    
    for l in range(len(S3)):
        Index = np.argmax(Q3[l,:,5])
        q3[j,l,:]=[S3[l,0],S3[l,1],S3[l,2],S3[l,3],Q3[l,Index,4]]
            
            
    print("This iteration--- %s seconds ---" % (time.time() - istart_time)) 
#if False:
    for l in range(len(Q2)):
        Index = np.argmax(Q2[:,5])
        q2[j,:] = [S2[0],S2[1],S2[2],S2[3],Q2[Index,4]] 
    
print("--- %s seconds ---" % (time.time() - start_time)) 
        
                
            