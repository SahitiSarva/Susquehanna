import math
import numpy as np

class RBF:
    # RBF calculates the values of the radial basis function that determine the release 

    # Attributes
    #-------------------
    # numberOfRBF       : int 
    #                     number of radial basis functions, typically 2 more than the number of inputs
    # numberofInputs    : int
    # numberOfOutputs   : int
    # center            : list
    #                     list of center values from optimization
    # radius            : list
    #                     list of radius values from optimization
    # weights           : list
    #                     list of weights from optimization
    # out               : list
    #                     list of the same size as the number of RBFs that determines the control policy


    def __init__(self, numberOfRBF, numberOfInputs, numberOfOutputs, center, radius, weights):
        self.numberOfRBF = numberOfRBF
        self.numberOfInputs = numberOfInputs
        self.numberOfOutputs = numberOfOutputs
        self.radius = radius
        self.center = center
        self.weights = weights
        #self.inputRBF = inputRBF

        zero_array = [[0]*int((self.numberOfInputs)/2)]*self.numberOfRBF
        
        # TODO: Change the numberofinputs - 2 to a parameter. 
        # Q: Why is it that we need only half the values for center and radius?

        one_array = [[1.0]*int((self.numberOfInputs)/2)]*self.numberOfRBF
        
        self.center = np.array(self.center).reshape((self.numberOfRBF, int((self.numberOfInputs)/2)))
        self.center = np.concatenate((self.center, zero_array), axis=1)

        self.radius = np.array(self.radius).reshape((self.numberOfRBF, int((self.numberOfInputs)/2)))
        self.radius = np.concatenate((self.radius, one_array), axis=1)

        self.weights = np.array(self.weights).reshape(self.numberOfRBF, self.numberOfOutputs)

        ws = self.weights.sum(axis=0)

        for i in [np.where(ws == i)[0][0] for i in ws if i>10**-6] :
            self.weights[:,i]= self.weights[:,i]/ws[i]


    def rbf_control_law(self, inputRBF):

        center, radius, weights = self.center, self.radius, self.weights # calling the previous function with defau;t va;ues of input, output and number of RBF
        # phi=control parameters
        phi = np.exp(-((np.array(inputRBF) - center)**2/radius**2).sum(axis=1))
        out = (weights*(phi.reshape(self.numberOfRBF,1))).sum(axis=0)

        return out
