import numpy as np
import pandas as pd

class utils:
    
    def loadVector(file_name, row = type(None)):
        output = pd.read_csv(file_name, nrows = row, header = None)
        output = output[0].tolist()
        return output

    # TODO: Set default values to take all rows and all columns
    def loadMatrix( file_name, row, column):
        output = pd.read_csv(file_name, header = None, sep='   ',  nrows=row, usecols = range(0,column), engine='python')
        output = output.values
        return output

    def loadArrangeMatrix(file_name, rows, cols):
        arr = np.zeros((rows,cols))
        data = np.loadtxt(file_name)
        k = 0
        while (k < len(data)):
            for i in range(0, rows):
                for j in range(0, cols):
                    arr[i][j] = data[k]
                    k = k + 1
        return arr

    # def filterDictionaryPercentile(dictionary, percentile):
    #     percentile = np.percentile([v for k,v in dictionary.items()], percentile)
    #     #print(percentile)
    #     dictionary = dict(filter(lambda x: x[1] <= percentile, dictionary.items()))

    #     #print(dictionary)
    #     return dictionary

    def cubicFeetToAcreFeet( x):
        conv = 43560 #1 acre = 43560 feet2
        return x/conv
    
    def acreFeetToCubicFeet( x):
        conv = 43560 #1 acre = 43560 feet2
        #output = round(x*conv/100000,1)*100000
        return x*conv

    def acreToSquaredFeet(x):
        conv = 43560
        output = round(x*conv/100000,1)*100000
        return output

    def interpolate_linear(X, Y, x):
        # Finding the interpolate 
        dim = len(X)-1
        # if storage is less than 
        if x < X[0]:
            # if x is smaller than the smallest value on X, interpolate between the first two values
            y = (x-X[0])*(Y[1] - Y[0]) / ( X[1] - X[0]) + Y[0]
            return y 
        if x > X[dim]:
            # if x is larger than the largest value, interpolate between the the last two values
            y = (x-X[dim])*(Y[dim] - Y[dim-1]) / ( X[dim] - X[dim-1]) + Y[dim]
            return y
            #y = Y[dim]
        else:
            y = np.interp(x, X, Y)
        
            # delta = 0.0
            # j = -99
            # # Q: WOuld this max value work for python as well?
            # min_d = 1.7976931348623158e+308

            # for i in range(0,len(X)):
            #     if X[i] == x:
            #         y = Y[i]
            #         return y
                
            #     delta = abs(X[i] - x)
            #     if delta < min_d:
            #         min_d = delta
            #         j = i
                
            # if X[j] < x:
            #     k = j
            # else:
            #     k = j-1

            # a = (Y[k+1] - Y[k]) / (X[k+1] - X[k])
            # b = Y[k] - a*X[k]
            # y = a*x + b

            return y

    def inchesToFeet( x):
        conv = 0.08333
        return x*conv
    
    def cubicFeetToCubicMeters( x):
        conv = 0.0283
        return x*conv
    
    def feetToMeters( x):
        conv = 0.3048
        return x*conv
    
    def computeMean(x):
        return sum(x) / len(x)
    
    def computePercentile(x, percentile):

        return np.percentile(x,percentile)