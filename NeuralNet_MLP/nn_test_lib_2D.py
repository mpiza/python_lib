# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 18:22:44 2020

@author: Apa
"""
import numpy as np
import matplotlib.pyplot as plt 
import nn_lib

#=======================================================================

# the function we want to have (desired outcome)
def myFunc(x0,x1):
    r2=x0**2+ x1**2
#    return(np.exp(0*r2)*abs(x1+x0))
    return(r2)
    
xrange=np.linspace(-0.5,0.5,100)
X0,X1=np.meshgrid(xrange,xrange)



fig=plt.figure(figsize=(10,10))
plt.imshow(myFunc(X0,X1),interpolation='nearest',origin='lower')
plt.colorbar()
plt.show()

def make_batch(batchsize):
   

    inputs=np.random.uniform(low=-0.5,high=+0.5,size=[batchsize,2])
    targets=np.zeros([batchsize,1]) # must have right dimensions
    targets[:,0]=myFunc(inputs[:,0],inputs[:,1])
    return(inputs,targets)

# set up all the weights and biases

LayerSizes=[2,30,30,30,30,1] # input-layer,hidden-1,hidden-2,...,output-layer
NumLayers = len(LayerSizes) - 1    # does not count input-layer (but does count output)

w_low = -0.5
w_high = +0.5
Weights=[np.random.uniform(low=w_low,high= w_high,size=[ LayerSizes[j],LayerSizes[j+1] ]) for j in range(NumLayers)]
Biases=[np.zeros(LayerSizes[j+1]) for j in range(NumLayers)]        
        
# set up all the helper variables

y_layer=[np.zeros(LayerSizes[j]) for j in range(NumLayers+1)]
df_layer=[np.zeros(LayerSizes[j+1]) for j in range(NumLayers)]
dw_layer=[np.zeros([LayerSizes[j],LayerSizes[j+1]]) for j in range(NumLayers)]
db_layer=[np.zeros(LayerSizes[j+1]) for j in range(NumLayers)]


   
# Now: the training! (and plot the cost function)
eta=.1

batchsize = 1000  # number of points taken in  "image" 
batches=500  # number of image samples
costs=np.zeros(batches)

for k in range(batches):
    y_in,y_target= make_batch(batchsize)
    costs[k]=nn_lib.train_net(y_in,y_target,eta, y_layer, df_layer, Weights, Biases, NumLayers,dw_layer, db_layer, batches)

fig=plt.figure(figsize=(10,10))
plt.plot(np.sqrt(costs))
plt.show()


# a 'test' batch that includes all the points on the image grid
test_batchsize=np.shape(X0)[0]*np.shape(X0)[1]
testsample=np.zeros([test_batchsize,2])
testsample[:,0]=X0.flatten()
testsample[:,1]=X1.flatten()

# show the output of this net
testoutput=nn_lib.apply_net_simple(testsample, y_layer,Weights, Biases, NumLayers)
fig=plt.figure(figsize=(10,10))
#myim=plt.imshow(np.reshape(testoutput,np.shape(X0)),origin='lower',interpolation='none')
plt.imshow(np.reshape(testoutput,np.shape(X0)),origin='lower',interpolation='none')
plt.colorbar()
plt.show()
        
