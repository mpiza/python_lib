# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 13:25:41 2020

@author: Apa
"""

import numpy as np
import matplotlib.pyplot as plt 


Nlayers = 10 # not counting the input layer & the output layer
LayerSize = 50

input_size = 2
output_size = 1

# weight and biases ranges for each layer

w_low_first = -1.0
w_hi_first = 1.0

w_low = -3.5
w_hi = 3.5

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

# Mid layers
Weights = np.random.uniform(low = w_low,high = w_hi,size=[Nlayers,LayerSize,LayerSize])
Biases = np.random.uniform(low = b_low,high = b_hi,size=[Nlayers,LayerSize])


# for the final layer (i.e. the output neuron)
WeightsFinal = np.random.uniform(low = w_low_final, high = w_hi_final,size=[LayerSize,output_size])
BiasesFinal = np.random.uniform(low = b_low_final,high = b_hi_final ,size=output_size)


def activation(z):
    return(1/(1+np.exp(-z)))

def apply_layer_new(y_in,w,b): # a function that applies a layer    
    
    z=np.dot(y_in,w)+b # note different order in matrix product!
    return activation(z)

def apply_multi_net(y_in, Weights, Biases, WeightsFinal, BiasesFinal, Nlayers):
#    global Weights, Biases, WeightsFinal, BiasesFinal, Nlayers
    
    y=apply_layer_new(y_in,WeightsFirst,BiasesFirst)    
    for j in range(Nlayers):
        y=apply_layer_new(y,Weights[j,:,:],Biases[j,:])
    output=apply_layer_new(y,WeightsFinal,BiasesFinal)
    return(output)
    
    
# Generate a 'mesh grid', i.e. x,y values in an image
M=200
v0,v1=np.meshgrid(np.linspace(-0.5,0.5,M),np.linspace(-0.5,0.5,M))
batchsize=M**2 # number of samples = number of pixels = M^2
y_in =  np.zeros([batchsize,2])
y_in[:,0]=v0.flatten() # fill first component (index 0)
y_in[:,1]=v1.flatten() # fill second component

# use the MxM input grid that we generated above 
y_out = apply_multi_net(y_in,Weights, Biases, WeightsFinal, BiasesFinal, Nlayers) # apply net to all these samples!


y_in_2D = np.reshape(y_in[:,0],[M,M])
y_2D = np.reshape(y_out[:,0],[M,M]) # back to 2D image

plt.figure(1)
plt.subplot(211)
plt.imshow(y_in_2D,origin='lower',extent=[-0.5,0.5,-0.5,0.5],interpolation='nearest',cmap='RdBu')
plt.colorbar()

plt.subplot(212)
plt.imshow(y_2D,origin='lower',extent=[-0.5,0.5,-0.5,0.5],interpolation='nearest',cmap='RdBu')
plt.colorbar()
plt.show()

    
    
    