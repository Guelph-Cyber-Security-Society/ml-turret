##
## Library Imports
##
from torch import Tensor


##
## Image Class
##
class Image:
    def __init__(self, name: str, image: Tensor, width: int, height: int):
        self.name = name
        self.width = width
        self.height = height
        self.image = image
