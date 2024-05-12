import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sympy import *

def evalFuncs(a, currdict):
    """Evaluates an array of SymPy function with an input dictionary

    Args:
        param1 (np.array [n_eqs, n_obs] of type or nested list): SymPy functions to be evaluated.
        param2 (dict): Dictionary to evaluate simpy functions.

    Returns:
        array: evaluated expressions for array
    """
    a = Array(a).subs(currdict)
    return a
    
class affineTransformation:
    """
    This class forms an affine transformation. The initialization method 
    expects 1D numpy arrays, which are transformed into a 3D space to form 
    A matrix and calculate transformation values

    Args:
        x_c (1d np.array): X coordinates in comparator system
        y_c (1d np.array): Y coordinates in comparator system
        x_f (1d np.array): X coordinates for fedicial system
        y_f (1d np.array): Y coordinates for fedicial system
        verbose (bool): whether to print values on init. default = True

    Attributes:
        x_cap (np.array): estimates for coordinates in feducial system
        a (np.array): linearized value a
        b (np.array): linearized value b
        c (np.array): linearized value c
        d (np.array): linearized value d
        x_translation (np.array): value for translation in the x-direction
        y_translation (np.array): value for translation in the x-direction
        scalex (np.array): X scale value 
        scaley (np.array): Y scale value 
        rotation (np.array): rotation value 
        nonortho (np.array): nonorthogonality 
    """    
    def __init__(self, x_c, y_c, x_f, y_f, verbose=False):
        
        # CONVERT EVERYTHING TO NUMPY ARRAY AND INDEX AS DEPTH
        x_c = np.array(x_c)[np.newaxis, :]
        y_c = np.array(y_c)[np.newaxis, :]
        x_f = np.array(x_f)[np.newaxis, :]
        y_f = np.array(y_f)[np.newaxis, :]
        
        # CREATE L-MATRIX TO BE 2Px1
        l = np.dstack((x_f, y_f)).flatten()
        
        # CREATE A MATRIX
        a = np.vstack((x_c, y_c, np.ones(x_c.shape), np.zeros(x_c.shape), np.zeros(x_c.shape), np.zeros(x_c.shape))).T
        b = np.vstack((np.zeros(x_c.shape), np.zeros(x_c.shape), np.zeros(x_c.shape), x_c, y_c, np.ones(x_c.shape))).T

        outl = list()
        for i in range(x_c.size):
            outl.append(a[i, :])
            outl.append(b[i, :])
            
        # CONVERT LIST OF NP.ARRAYS TO NP.ARRAY
        A = np.array(outl)
        
        # CALCULATE X_CAP
        x_cap_temp = np.matmul(np.linalg.inv(np.matmul(A.T, A)), A.T)
        x_cap = np.matmul(x_cap_temp, l)
        
        # CALCULATE SAVE VALUES FOR TRANSFORMATION PARAMETERS
        self.x_cap = x_cap
        self.a = x_cap[0]
        self.b = x_cap[1]
        self.c = x_cap[3]
        self.d = x_cap[4]
        self.x_translation = x_cap[2]
        self.y_translation = x_cap[5]
        self.scalex = np.sqrt(self.a ** 2 + self.c ** 2)
        self.scaley = np.sqrt(self.b ** 2 + self.d ** 2)
        top = (self.a * self.b + self.c*self.d)
        bot = (self.a * self.b - self.c*self.d)
        self.nonortho = np.arctan(top / bot)
        self.rotation = np.arctan(self.c / self.a)
        
        self.matrix = np.array([[self.a, self.b, self.x_translation], 
                                [self.c, self.d, self.y_translation],
                                [     0,      0,                  1]])
        
        self.currdict = {
            "a" : self.a,
            "b" : self.b,
            "c" : self.c,
            "d" : self.d,
            "x_translation" : self.x_translation,
            "y_translation" : self.y_translation,
            "scalex" : self.scalex, 
            "scaley" : self.scaley,
            "nonortho" : self.nonortho,
            "rotation" : self.rotation
        }
        
        if verbose:
            print("\nAffine Transformation Parameters -------------------------")
            for k, v in self.currdict.items():
                print("{:<8} {:<15.3e}".format(k, v))
        
    def transform(self, x_c, y_c):
        # CREATE A MATRIX
        temp_matrix = np.array([[self.a, self.b], [self.c, self.d]])
        # MULTIPLY TIMES WEIGHTS AND ADD TRANSLATION 
        temp_out = temp_matrix @ np.vstack((x_c, y_c))
        temp_out = temp_out.T + np.array([self.x_translation, self.y_translation])
        # RETURN VALUES
        return temp_out[:, 0], temp_out[:, 1]
    
class similarityTransformation:
    
    """
    This class forms a similarity transformation. The initialization method 
    expects 1D numpy arrays, which are transformed into a 3D space to form 
    A matrix and calculate transformation values

    Args:
        x_c (1d np.array): X coordinates in comparator system
        y_c (1d np.array): Y coordinates in comparator system
        x_f (1d np.array): X coordinates for fedicial system
        y_f (1d np.array): Y coordinates for fedicial system
        verbose (boolean): Whether to print estimated parameters upon init, default=True

    Attributes:
        x_cap (np.array): estimates for coordinates in feducial system
        a (np.array): linearized value a
        b (np.array): linearized value b
        x_translation (np.array): value for translation in the x-direction
        y_translation (np.array): value for translation in the x-direction
        scale (np.array): scale value 
        rotation (np.array): rotation value 
    """
    
    def __init__(self, x_c, y_c, x_f, y_f, verbose=True):
        
        # CONVERT EVERYTHING TO NUMPY ARRAY AND INDEX AS DEPTH
        x_c = np.array(x_c)[np.newaxis, :]
        y_c = np.array(y_c)[np.newaxis, :]
        
        x_f = np.array(x_f)[np.newaxis, :]
        y_f = np.array(y_f)[np.newaxis, :]
        
        # CREATE L-MATRIX TO BE 2Px1
        l = np.dstack((x_f, y_f)).flatten()
        
        # CREATE A MATRIX
        a = np.vstack((x_c, -1 * y_c, np.ones(x_c.shape), np.zeros(x_c.shape))).T
        b = np.vstack((y_c, x_c, np.zeros(x_c.shape), np.ones(x_c.shape))).T

        outl = list()
        for i in range(x_c.size):
            outl.append(a[i, :])
            outl.append(b[i, :])
            
        # CONVERT LIST OF NP.ARRAYS TO NP.ARRAY
        A = np.array(outl)
        
        # CALCULATE X_CAP
        x_cap_temp = np.matmul(np.linalg.inv(np.matmul(A.T, A)), A.T)
        x_cap = np.matmul(x_cap_temp, l)
        
        # SAVE VALUES
        self.x_cap = x_cap
        self.a = x_cap[0]
        self.b = x_cap[1]
        self.x_translation = x_cap[2]
        self.y_translation = x_cap[3]
        self.scale = np.sqrt(self.a ** 2 + self.b ** 2)
        self.rotation = np.arctan(self.b / self.a)
        
        self.matrix = np.array([
            [self.a, -1 * self.b, self.x_translation], 
            [self.b, self.a, self.y_translation],
            [     0,      0,                  1]])

        self.currdict = {
            "a" : self.a,
            "b" : self.b,
            "x_translation" : self.x_translation,
            "y_translation" : self.y_translation,
            "scale" : self.scale,
            "rotation" : self.rotation
        }
        
        if verbose:
            print("\nSimilarity Transformation Parameters -------------------------")
            for k, v in self.currdict.items():
                print("{:<8} {:<15.3e}".format(k, v))
        
    def transform(self, x_c, y_c):
        # CREATE A MATRIX
        temp_matrix = np.array([[self.a,  -1 * self.b], [self.b, self.a]])
        
        # MULTIPLY TIMES WEIGHTS AND ADD TRANSLATION 
        temp_out = temp_matrix @ np.vstack((x_c, y_c))
        temp_out = temp_out.T + np.array([self.x_translation, self.y_translation])
        
        # RETURN VALUES
        return temp_out[:, 0], temp_out[:, 1]