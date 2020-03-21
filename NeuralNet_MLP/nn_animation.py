# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 13:25:41 2020

@author: Apa
"""

import numpy as np
import matplotlib.pyplot as plt 
from matplotlib.animation import FuncAnimation



#Initialize parameters ======================

Nlayers = 5 # not counting the input layer & the output layer
LayerSize = 50

input_size = 2
output_size = 1

# weight and biases RANGES for each layer 

w_low_first = -1.0
w_hi_first = 1.0

w_low = -5
w_hi = 5

w_low_final = -1.0
w_hi_final = 1.0


b_low_first = -1.0
b_hi_first = 1.0

b_low = -1.0
b_hi = 1.0

b_low_final = -1.0
b_hi_final = 1.0



# for the first hidden layer (coming in from the input layer)
WeightsFirst = np.random.uniform(low = w_low_first,high = w_hi_first,size=[input_size,LayerSize])
BiasesFirst = np.random.uniform(low = b_low_first,high = b_hi_first,size=LayerSize)

# Middle layers
Weights = np.random.uniform(low = w_low,high = w_hi,size=[Nlayers,LayerSize,LayerSize])
Biases = np.random.uniform(low = b_low,high = b_hi,size=[Nlayers,LayerSize])


# for the final layer (i.e. the output neuron)
WeightsFinal = np.random.uniform(low = w_low_final, high = w_hi_final,size=[LayerSize,output_size])
BiasesFinal = np.random.uniform(low = b_low_final,high = b_hi_final ,size=output_size)


def activation(z):
     return(1/(1+np.exp(-z)))
#     return z*(z>0)
#      return z
     
     
# a function that applies a layer to of BATCH of inputs    takes a vextor y_in of dimmension (n_samples x n_in), a matrix w (n_in x n_out),vector  b (n_out)
# returns a MATRIX  of size (n_samples, n_out)
def apply_layer_new(y_in,w,b): 
    
    z=np.dot(y_in,w)+b # note different order in matrix product!
    return activation(z)

# apply neural net to y_in of dimmension (n_samples x n_in). Neural net represented by an array of  matrixes of weights and vector of biases. 

def apply_multi_net(y_in, Weights, Biases, WeightsFinal, BiasesFinal, Nlayers):
#    global Weights, Biases, WeightsFinal, BiasesFinal, Nlayers
    
    y=apply_layer_new(y_in,WeightsFirst,BiasesFirst)    
    for j in range(Nlayers):
        y=apply_layer_new(y,Weights[j,:,:],Biases[j,:])
    output=apply_layer_new(y,WeightsFinal,BiasesFinal)
    return(output)
    
    
# Generate a 'mesh grid', i.e. x,y values in an image
    
M=200   # image of dimension MxM
v0,v1=np.meshgrid(np.linspace(-0.5,0.5,M),np.linspace(-0.5,0.5,M))
batchsize=M**2 # number of samples = number of pixels = M^2
y_in =  np.zeros([batchsize,2])    
y_in[:,0]=v0.flatten() # fill first component (index 0)
y_in[:,1]=v1.flatten() # fill second component

# use the MxM input grid that we generated above 
y_out = apply_multi_net(y_in,Weights, Biases, WeightsFinal, BiasesFinal, Nlayers) # apply net to all these samples!


#y_in_2D = np.reshape(y_in[:,0],[M,M])
y_2D = np.reshape(y_out[:,0],[M,M]) # back to 2D image
#
#plt.figure(1)
#plt.subplot(211)
#plt.imshow(y_in_2D,origin='lower',extent=[-0.5,0.5,-0.5,0.5],interpolation='nearest',cmap='RdBu')
#plt.colorbar()
#
#plt.subplot(212)
#plt.imshow(y_2D,origin='lower',extent=[-0.5,0.5,-0.5,0.5],interpolation='nearest',cmap='RdBu')
#plt.colorbar()
#plt.show()



fig=plt.figure(figsize=(5,5)) # prepare figure
ax=plt.axes([0,0,1,1]) # fill everything with image (no border)
img=plt.imshow(y_2D,interpolation='nearest') # plot image
plt.axis('off') # no axes

t=0.0
dt=0.01

def update_wave(frame_number):
    global t, img
    global y_in, M
    global Weights, Biases, WeightsFinal, BiasesFinal, Nlayers
    
    Weights[1,1,1]=t
    y_out=apply_multi_net(y_in,Weights, Biases, WeightsFinal, BiasesFinal, Nlayers) # apply net to all these samples!
    y_2D=np.reshape(y_out[:,0],[M,M]) # back to 2D image
    img.set_data( y_2D )
    t+=dt
    return img

anim = FuncAnimation(fig, update_wave, interval=200)
plt.show()
    
    
    