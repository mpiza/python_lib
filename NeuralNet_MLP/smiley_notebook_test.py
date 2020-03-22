# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 11:36:46 2020

@author: Apa
"""

from numpy import * # get the "numpy" library for linear algebra
from scipy import ndimage # for image loading/processing
from scipy import misc
import imageio

import matplotlib.pyplot as plt # for plotting

def net_f_df1(z): # calculate f(z) and f'(z)
    val=1/(1+exp(-z))
    return(val,exp(-z)*(val**2)) # return both f and f'
    
# implement a ReLU unit (rectified linear), which
# works better for training in this case
def net_f_df(z): # calculate f(z) and f'(z)
    val=z*(z>0)
    return(val,z>0) # return both f and f'    
    
def forward_step(y,w,b): # calculate values in next layer, from input y
    z=dot(y,w)+b # w=weights, b=bias vector for next layer
    return(net_f_df(z)) # apply nonlinearity and return result
    
def apply_net(y_in): # one forward pass through the network
    global Weights, Biases, NumLayers
    global y_layer, df_layer # for storing y-values and df/dz values
    
    y=y_in # start with input values
    y_layer[0]=y
    for j in range(NumLayers): # loop through all layers [not counting input]
        # j=0 corresponds to the first layer above the input
        y,df=forward_step(y,Weights[j],Biases[j]) # one step, into layer j
        df_layer[j]=df # store f'(z) [needed later in backprop]
        y_layer[j+1]=y # store f(z) [also needed in backprop]        
    return(y)
    
def apply_net_simple(y_in): # one forward pass through the network
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
    return( dot(delta,transpose(w))*df )
    
def backprop(y_target): # one backward pass through the network
    # the result will be the 'dw_layer' matrices that contain
    # the derivatives of the cost function with respect to
    # the corresponding weight
    global y_layer, df_layer, Weights, Biases, NumLayers
    global dw_layer, db_layer # dCost/dw and dCost/db (w,b=weights,biases)
    global batchsize
    
    delta=(y_layer[-1]-y_target)*df_layer[-1]
    dw_layer[-1]=dot(transpose(y_layer[-2]),delta)/batchsize
    db_layer[-1]=delta.sum(0)/batchsize
    for j in range(NumLayers-1):
        delta=backward_step(delta,Weights[-1-j],df_layer[-2-j])
        dw_layer[-2-j]=dot(transpose(y_layer[-3-j]),delta)/batchsize
        db_layer[-2-j]=delta.sum(0)/batchsize
        
        
def gradient_step(eta): # update weights & biases (after backprop!)
    global dw_layer, db_layer, Weights, Biases
    
    for j in range(NumLayers):
        Weights[j]-=eta*dw_layer[j]
        Biases[j]-=eta*db_layer[j]

#Train net to reproduce a 2D function
        
# set up all the weights and biases

NumLayers=4 # does not count input-layer (but does count output)
LayerSizes=[2,150,150,100,1] # input-layer,hidden-1,hidden-2,...,output-layer

Weights=[random.uniform(low=-0.1,high=+0.1,size=[ LayerSizes[j],LayerSizes[j+1] ]) for j in range(NumLayers)]
Biases=[zeros(LayerSizes[j+1]) for j in range(NumLayers)]

# set up all the helper variables

y_layer=[zeros(LayerSizes[j]) for j in range(NumLayers+1)]
df_layer=[zeros(LayerSizes[j+1]) for j in range(NumLayers)]
dw_layer=[zeros([LayerSizes[j],LayerSizes[j+1]]) for j in range(NumLayers)]
db_layer=[zeros(LayerSizes[j+1]) for j in range(NumLayers)]
        
# load the pixel image!
face = imageio.imread('imageio:horse.png')
pixel_image=transpose(face[:,:,0]) # have to transpose...
pixel_image=pixel_image[:,::-1] # and flip... to get the right view!
pixel_image-=pixel_image.min()
pixel_image=(pixel_image.astype(dtype='float'))/pixel_image.max() # normalize between 0 and 1!
Npixels=shape(pixel_image)[1] # assuming a square image!


# the function we want to have (desired outcome)
def myFunc(x0,x1):
    global pixel_image, Npixels
    # convert to integer coordinates (assuming input is 0..1)
    x0int=(x0*Npixels*0.9999).astype(dtype='int')
    x1int=(x1*Npixels*0.9999).astype(dtype='int')
    return(pixel_image[x0int,x1int]) # extract color values at these pixels

# check that this works:
Npixels_Test=200 # do the test output on a low-res grid! (saves time)
xrange=linspace(0,1,Npixels_Test)
X0,X1=meshgrid(xrange,xrange)
fig=plt.figure(figsize=(10,10))
plt.imshow(myFunc(X0,X1),interpolation='nearest',origin='lower')
plt.axis('off')
plt.colorbar()
plt.show()


def train_net(y_in,y_target,eta): # one full training batch
    # y_in is an array of size batchsize x (input-layer-size)
    # y_target is an array of size batchsize x (output-layer-size)
    # eta is the stepsize for the gradient descent
    global y_out_result
    
    y_out_result=apply_net(y_in)
    backprop(y_target)
    gradient_step(eta)
    cost=0.5*((y_target-y_out_result)**2).sum()/batchsize
    return(cost)

# pick 'batchsize' random positions in the 2D square
def make_batch():
    global batchsize

    inputs=random.uniform(low=0,high=1,size=[batchsize,2])
    targets=zeros([batchsize,1]) # must have right dimensions
    targets[:,0]=myFunc(inputs[:,0],inputs[:,1])
    return(inputs,targets)
    
    
# define the batchsize
batchsize=1000

# Now: the training! (and plot the cost function)
eta=0.2
batches=50000
costs=zeros(batches)

plot_step = 5000

random_id = random.randint(1,10000)

for k in range(batches):
    y_in,y_target=make_batch()
    costs[k]=train_net(y_in,y_target,eta)
    if (k%500 == 0):
        print(k, sqrt(costs[k]))
        
        if (k%(2*plot_step) == 0):
            fig=plt.figure(figsize=(10,10))
            plt.plot(sqrt(costs))
            plt.show()

# a 'test' batch that includes all the points on the image grid
        test_batchsize=shape(X0)[0]*shape(X0)[1]
        testsample=zeros([test_batchsize,2])
        testsample[:,0]=X0.flatten()
        testsample[:,1]=X1.flatten()


# check the output of this net
        testoutput=apply_net_simple(testsample)

# show this!
        if (k%plot_step == 0):
            fig=plt.figure(figsize=(10,10))
            myim=plt.imshow(reshape(testoutput,shape(X0)),origin='lower',interpolation='nearest',vmin=0.0,vmax=1.0)
            plt.axis('off')
            plt.colorbar()
            plt.show()
            
            fig_name = str(random_id) + " " +  str(k)
            plt.savefig(fig_name)

