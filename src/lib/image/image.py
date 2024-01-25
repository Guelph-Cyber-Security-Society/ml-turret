##
## Path: src/lib/image/image.py
##

##
## Libraries
##   Note: Avoid importing entire libraries. Import only what is needed from them!
##
from torch import Tensor


##
## Image Class
## This class represents an image and stores its name, width, height, and pixel data.
## The pixel data is stored in a Tensor so that it can be used with PyTorch.
##
class Image:
    ##
    ## The constructor defines the name, pixel data, width, and height of the image.
    ## We store this info in a class so that is can more easily be used in other
    ## parts of the program.
    ##
    ## @param name The name of the image
    ## @param image The pixel data of the image
    ## @param width The width of the image
    ## @param height The height of the image
    ##
    def __init__(self, name: str, tensor: Tensor, width: int, height: int) -> None:
        self.name = name
        self.width = width
        self.height = height
        self.tensor = tensor

        ##
        ## End of constructor
        ##

    ##
    ## End of class
    ##


##
## End of file
##
