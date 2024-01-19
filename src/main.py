##
## Path: src/lib/net.py
##

##
## Libraries
##
from lib.basic_conv_net import BasicConvNet


##
## Main function
##
def main() -> None:
    # Create a network with 1 input channel, 6 output channels, and a kernel size of 5
    net = BasicConvNet(
        inchannels=1, outchannels=6, ksize=5, poolksize=2
    )  # Always have var=... for each parameter! This makes it easier to read.

    # Print the network
    print(net)


##
## Execute main
##
if __name__ == "__main__":
    main()
