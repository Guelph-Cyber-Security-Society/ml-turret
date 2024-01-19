##
## Path: src/lib/net.py
##

##
## Libraries
##   Note: Avoid importing entire libraries. Import only what is needed from them!
##
from torch.nn import Module, Sequential, Conv2d, ReLU, MaxPool2d
from torch import Tensor


##
## Basic Convolutional Neural Network Class
##
class BasicConvNet(Module):
    ##
    ## The constructor defines the layers of the network.
    ## The input parameters are the number of input channels,
    ## the number of output channels, and the kernel size.
    ##
    ## @param inchannels The number of input channels
    ## @param outchannels The number of output channels
    ## @param ksize The kernel size
    ##
    def __init__(
        self, inchannels: int, outchannels: int, ksize: int, poolksize: int = 2
    ):
        # Call the constructor of the parent class
        super(BasicConvNet, self).__init__()

        # Convolutional Layer
        self.sequence = Sequential(
            Conv2d(inchannels, outchannels, ksize),
            ReLU(),
            MaxPool2d(poolksize, poolksize),
            Conv2d(outchannels, outchannels, ksize),
            ReLU(),
            MaxPool2d(poolksize, poolksize),
        )

        ##
        ## End of constructor
        ##

    ##
    ## The forward pass function executes the sequence of layers
    ## in the order they were defined. The input is the input
    ## tensor to the network. The output is the output tensor
    ## of the network.
    ##
    ## @param x The input tensor to the network
    ## @return The output tensor of the network
    ##
    def forward(self, x) -> Tensor:
        x = self.sequence(x)
        return x

        ##
        ## End of 'forward' function
        ##

    ##
    ## End of class
    ##


##
## End of file
##
