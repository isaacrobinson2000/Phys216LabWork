#Physics plotting and least-squares fitting library
#to be imported 
#March 27, 2018
#Jaylene

import numpy as np
import matplotlib.pyplot as plt
import math

#define plotting and linear least squares fitting as functions

#goodness of fit parameter
def LLSFD2(x,y,dy,b,m):
    N = sum(((y-b-m*x)/dy)**2)
    return N
                      
def Delta(x, dy):
    D = (sum(1/dy**2))*(sum(x**2/dy**2))-(sum(x/dy**2))**2
    return D


def plotfit(x,y,dy,title,xlabel,ylabel,slopeunit):
    b,m=np.polynomial.polynomial.polyfit(x,y,1,w=dy)
    fit = b+m*x
    N = LLSFD2(x,y,dy,b,m)    
    D=Delta(x, dy)
    dm = math.sqrt(1/D*sum(1/dy**2)) #error in slope
    db = math.sqrt(1/D*sum(x**2/dy**2)) #error in intercept
    plt.scatter(x, y, color='red', marker='o')
    #Plot least squares fit line
    plt.plot(x, fit, color='green', linestyle='--')

    #Plot y error bars - do not need to modify
    plt.errorbar(x, y, yerr=dy, xerr=None, fmt="none")

    #Write x and y axes labels, title.  Modify these!
    plt.xlabel(xlabel) #x-axis label
    plt.ylabel(ylabel) #y-axis label
    plt.title(title) #plot title
    plt.show()
    print("Slope = ", m, " ", slopeunit)
    print("Error in slope = ", dm, " ", slopeunit)
    print("Goodness of fit = ", N)
    
    return m, dm, N
        