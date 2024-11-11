import numpy as np
import pandas as pd
from scipy import stats
import random

##-------------------------------------------------------------##
##--- Make the 3D Chondrule List ------------------------------##
##-------------------------------------------------------------##

def chd3DList(numberOfChondrules, zAxisLength, mu3D, sigma3D):
    chdList = []

    for i in range(numberOfChondrules):
        chdDiameter = np.random.lognormal(mu3D, sigma3D) # random chd diameter from a log-normal distribution
        chdCenterInter = zAxisLength * random.random() # random z-axis intercept
        chdTopInter = chdCenterInter + .5 * chdDiameter
        chdBottomInter = chdCenterInter - .5 * chdDiameter
        chdList.append([chdDiameter, chdCenterInter, chdTopInter, chdBottomInter])

    dfChdList = pd.DataFrame(chdList)
    return dfChdList.rename(columns = {0:'Chd Diameter', 1:'Chd Center z Intercept' , 2:'Chd Top z Intercept', 3:'Chd Bottom z Intercept'})
    

##---------------------------------------------------------------##
##--- Produce the list of the apparent 2D chondrule diameters ---##
##---------------------------------------------------------------##

def sectionedChd(zAxInter, chdList3D):
    zAxisIntercept = zAxInter
    
    # Select all chd visible at a specific z-axis intercept
    fil = (zAxisIntercept < chdList3D['Chd Top z Intercept']) & (zAxisIntercept > chdList3D['Chd Bottom z Intercept'])
    sectionedChdList = chdList3D[fil]

    # Calculate the diameters of all the sectioned chondrules at a specific zAxisIntercept
    appChdDiameterList = []
    for index, sectChd in sectionedChdList.iterrows():
        appChdDia = 2 * (((.5 * sectChd.iloc[0])**2 - (sectChd.iloc[1] - zAxisIntercept)**2)**.5)
        appChdDiameterList.append(appChdDia)

    return appChdDiameterList


##-------------------------------------------------------------##
##--- Fitting a distribution and calculating its mu & sigma ---##
##-------------------------------------------------------------##

def calcMuSigma(sample):
    shape, loc, scale = stats.lognorm.fit(sample, floc=0) # hold location to 0 while fitting
    m, s = np.log(scale), shape  # mu, sigma
    return (m, s)