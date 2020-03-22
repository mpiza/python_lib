# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 13:16:10 2020

@author: Apa
"""

import numpy as np

def net_f_df(z): # calculate f(z) and f'(z)
#    val=1/(1+np.exp(-z))
#    return(val,np.exp(-z)*(val**2)) # return both f and f'
    
    val=z*(z>0)
    return(val,z>0) # return both f and f'    
    
 
def forward_step(y,w,b): # calculate values in next layer, from input y
    
    z = np.dot(y,w)+b # w=weights, b=bias vector for next layer
    return(net_f_df(z)) # apply nonlinearity and return result
    
    
def apply_net(y_in,Weights, Biases, NumLayers,y_layer, df_layer ): # one forward pass through the network
#    global Weights, Biases, NumLayers
#    global y_layer, df_layer # for storing y-values and df/dz values
    
    y=y_in # start with input values
    y_layer[0]=y
    for j in range(NumLayers): # loop through all layers [not counting input]
        # j=0 corresponds to the first layer above the input
        y,df=forward_step(y,Weights[j],Biases[j]) # one step, into layer j
        df_layer[j]=df # store f'(z) [needed later in backprop]
        y_layer[j+1]=y # store f(z) [also needed in backprop]        
    return(y)    
    
    
def apply_net_simple(y_in,y_layer, Weights, Biases,NumLayers): # one forward pass through the network
    # no storage for backprop (this is used for simple tests)

    y=y_in # start with input values
    y_layer[0]=y
    for j in range(NumLayers): # loop through all layers
        # j=0 corresponds to the first layer above the input
        y,df=forward_step(y,Weights[j],Biases[j]) # one step
    return(y)
    
    
def backward_step(delta,w,df): 
    # delta at layer N, of batchsize x layersize(N))
    # w between N-1 and N [layersize(N-1) x layersize(N) matrix]
    # df = df/dz at layer N-1, of batchsize x layersize(N-1)
    return( np.dot(delta,np.transpose(w))*df ) 
    
def backprop(y_target,y_layer, df_layer, Weights, Biases, NumLayers,dw_layer, db_layer, batchsize): # one backward pass through the network
    # the result will be the 'dw_layer' matrices that contain
    # the derivatives of the cost function with respect to
    # the corresponding weight
#    global y_layer, df_layer, Weights, Biases, NumLayers
#    global dw_layer, db_layer # dCost/dw and dCost/db (w,b=weights,biases)
#    global batchsize
    
    delta=(y_layer[-1]-y_target)*df_layer[-1]
    dw_layer[-1]=np.dot(np.transpose(y_layer[-2]),delta)/batchsize
    db_layer[-1]=delta.sum(0)/batchsize
    for j in range(NumLayers-1):
        delta=backward_step(delta,Weights[-1-j],df_layer[-2-j])
        dw_layer[-2-j]=np.dot(np.transpose(y_layer[-3-j]),delta)/batchsize
        db_layer[-2-j]=delta.sum(0)/batchsize
        
        
def gradient_step(eta,NumLayers,dw_layer, db_layer, Weights, Biases): # update weights & biases (after backprop!)
#    global dw_layer, db_layer, Weights, Biases
    
    for j in range(NumLayers):
        Weights[j]-=eta*dw_layer[j]
        Biases[j]-=eta*db_layer[j]

def train_net(y_in,y_target,eta,y_layer, df_layer, Weights, Biases, NumLayers,dw_layer, db_layer, batchsize): # one full training batch
    # y_in is an array of size batchsize x (input-layer-size)
    # y_target is an array of size batchsize x (output-layer-size)
    # eta is the stepsize for the gradient descent
#    global y_out_result
    
    y_out_result=apply_net(y_in,Weights, Biases, NumLayers,y_layer, df_layer)
    backprop(y_target,y_layer, df_layer, Weights, Biases, NumLayers,dw_layer, db_layer, batchsize)
    gradient_step(eta,NumLayers,dw_layer, db_layer, Weights, Biases)
    cost=0.5*((y_target-y_out_result)**2).sum()/batchsize
    return(cost)        
    
    
 


