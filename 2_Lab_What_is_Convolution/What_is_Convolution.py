import torch 
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage, misc

#Create a two-dimensional convolution object by using the constructor Conv2d, the parameter in_channels and out_channels will be used for this section, 
#and the parameter kernel_size will be three.
conv = nn.Conv2d(in_channels=1, out_channels=1,kernel_size=3)
conv

#Because the parameters in nn.Conv2d are randomly initialized and learned through training, give them some values.
conv.state_dict()['weight'][0][0]=torch.tensor([[1.0,0,-1.0],[2.0,0,-2.0],[1.0,0.0,-1.0]])
conv.state_dict()['bias'][0]=0.0
conv.state_dict()

#Create a dummy tensor to represent an image. The shape of the image is (1,1,5,5) where:
#(number of inputs, number of channels, number of rows, number of columns )
#Set the third column to 1:
image=torch.zeros(1,1,5,5)
image[0,0,:,2]=1
image

#Call the object conv on the tensor image as an input to perform the convolution and assign the result to the tensor z.
z=conv(image)
z

#Create a kernel of size 2:
K=2
conv1 = nn.Conv2d(in_channels=1, out_channels=1,kernel_size=K)
conv1.state_dict()['weight'][0][0]=torch.tensor([[1.0,1.0],[1.0,1.0]])
conv1.state_dict()['bias'][0]=0.0
conv1.state_dict()
conv1

#Create an image of size 2:
M=4
image1=torch.ones(1,1,M,M)

#Perform the convolution and verify the size is correct:
z1=conv1(image1)
print("z1:",z1)
print("shape:",z1.shape[2:4])

#Stride parameter---------------------------------------------------------------
#Create a convolution object with a stride of 2:
conv3 = nn.Conv2d(in_channels=1, out_channels=1,kernel_size=2,stride=2)

conv3.state_dict()['weight'][0][0]=torch.tensor([[1.0,1.0],[1.0,1.0]])
conv3.state_dict()['bias'][0]=0.0
conv3.state_dict()

#Perform the convolution and verify the size is correct
z3=conv3(image1)

print("z3:",z3)
print("shape:",z3.shape[2:4])

#Zero Padding---------------------------------------------------------------
#As you apply successive convolutions, the image will shrink. You can apply zero padding to keep the image at a reasonable size, 
#which also holds information at the borders.

#In addition, you might not get integer values for the size of the kernel. Consider the following image:
image1

#Try performing convolutions with the kernel_size=2 and a stride=3
conv4 = nn.Conv2d(in_channels=1, out_channels=1,kernel_size=2,stride=3)
conv4.state_dict()['weight'][0][0]=torch.tensor([[1.0,1.0],[1.0,1.0]])
conv4.state_dict()['bias'][0]=0.0
conv4.state_dict()
z4=conv4(image1)
print("z4:",z4)
print("z4:",z4.shape[2:4])

#Consider the following example:
conv5 = nn.Conv2d(in_channels=1, out_channels=1,kernel_size=2,stride=3,padding=1)

conv5.state_dict()['weight'][0][0]=torch.tensor([[1.0,1.0],[1.0,1.0]])
conv5.state_dict()['bias'][0]=0.0
conv5.state_dict()
z5=conv5(image1)
print("z5:",z5)
print("z5:",z4.shape[2:4])

#Practice Question--------------------------------------------------------------
#A kernel of zeros with a kernel size=3 is applied to the following image: 
Image=torch.randn((1,1,4,4))
Image

#Question: Without using the function, determine what the outputs values are as each element:
'''
As each element of the kernel is zero, and for every  output, the image is multiplied  by the  kernel, the result is always zero 
'''

#Question: Use the following convolution object to perform convolution on the tensor Image:
conv = nn.Conv2d(in_channels=1, out_channels=1,kernel_size=3)
conv.state_dict()['weight'][0][0]=torch.tensor([[0,0,0],[0,0,0],[0,0.0,0]])
conv.state_dict()['bias'][0]=0.0

conv(Image)

#Question: You have an image of size 4. The parameters are as follows kernel_size=2,stride=2. What is the size of the output?
'''
(M-K)/stride +1
(4-2)/2 +1
2
'''