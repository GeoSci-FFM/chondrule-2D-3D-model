import streamlit as st
import numpy as np
import pandas as pd
from scipy import stats
import random
import time
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle


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


##-----------------------##
##--- The 2D/3D model ---##
##-----------------------##

# Debounce input changes to prevent app crashes from rapid clicking.
def update_values():
    time.sleep(0.5)  # Small delay to prevent excessive updates
    for key in [
        "iniSigmastart", "iniSigmaend", "iniSigmastep",
        "minChdSize_rangestart", "minChdSize_rangeend", "minChdSize_rangestep",
        "maxChdSize_rangestart", "maxChdSize_rangeend", "maxChdSize_rangestep",
        "numberOfChondrules_power", "zAxisLength_power"
    ]:
        st.session_state[key] = st.session_state.get(key, None)
    

def run_model(mu3DList, iniSigma_List, minChdSize_range, maxChdSize_range, numberOfChondrules, zAxisLength):
    tot_loops = len(maxChdSize_range) * len(mu3DList) * len(iniSigma_List) * len(minChdSize_range)
    model_progress_bar = st.progress(0)
    loop_counter = 0
    
    parameters = []
    for maxChdSize in maxChdSize_range:
        for mu3D in mu3DList:
            # the minimum chondrule diameters as well as the sigma of the chondrule size distribution are varied
            for iniSigma in iniSigma_List:
                # clear_output(wait=True) # for progress indicator, which seems not to be working in streamlit
                # st.write(f'max Chd Size: {maxChdSize}')
                # st.write(f'mu3D: {mu3D}')
                # st.write(f'remaining iniSigma loops:  {len(iniSigma_List)-loop_counter}')
                dfChdList = chd3DList(numberOfChondrules, zAxisLength, mu3D, iniSigma) # noc, zAxisLen, mu, sigma
                
                for minChdSize in minChdSize_range:
                    filDiameter = (dfChdList['Chd Diameter'] > minChdSize) & (dfChdList['Chd Diameter'] < maxChdSize)
                    dfChdList = dfChdList[filDiameter]
                    appChdDiameterList = sectionedChd(0.5 * zAxisLength, dfChdList)
        
                    muFit3D, SigmaFit3D = calcMuSigma(dfChdList['Chd Diameter'])
                    muFit2D, sigmaFit2D = calcMuSigma(appChdDiameterList)
        
                    parameters3D = [muFit3D, SigmaFit3D, np.e**(muFit3D + .5 * SigmaFit3D**2), np.e**muFit3D, np.e**(muFit3D - SigmaFit3D**2)]   # 3D: mu, sigma, mean, median, mode
                    parameters2D = [muFit2D, sigmaFit2D, np.e**(muFit2D + .5 * sigmaFit2D**2), np.e**muFit2D, np.e**(muFit2D - sigmaFit2D**2)]
                    parameters.append([len(appChdDiameterList), minChdSize, maxChdSize, mu3D, iniSigma] + parameters3D + parameters2D)
                    loop_counter+=1
                    model_progress_bar.progress(loop_counter/tot_loops, text='working ...')
    model_progress_bar.empty()
    st.session_state.dfParameters = pd.DataFrame(parameters)

    st.session_state.dfParameters = st.session_state.dfParameters.rename(columns = {0:'Nr. of sect. Chd', 1:'min. Chd Diameter', 2:'max. Chd Diameter', 3:r'initial $\mu$ 3D', 4:r'initial $\sigma$ 3D'
                                ,5:r'$\mu$ Fit 3D', 6:r'$\sigma$ Fit 3D', 7:'mean 3D', 8:'median 3D', 9:'mode 3D'
                                ,10:r'$\mu$ Fit 2D', 11:r'$\sigma$ Fit 2D', 12:'mean 2D', 13:'median 2D', 14:'mode 2D'})


##--------------------------------##
##--- Plotting the 2D/3D model ---##
##--------------------------------##

def model_plot(max_chd_size, sel_mu3D, df_results):
    fil = (df_results['max. Chd Diameter'] == max_chd_size) & (df_results['initial $\mu$ 3D'] == sel_mu3D)
    dfParameters = df_results[fil]

    filBlue = dfParameters['mean 2D'] <= dfParameters['mean 3D']  # 2D < 3D
    filOrange = dfParameters['mean 2D'] > dfParameters['mean 3D']  # 2D > 3D
    
    # Clear previous plots
    plt.clf()

    plt.scatter(dfParameters[filBlue][r'initial $\sigma$ 3D'], dfParameters[filBlue]['min. Chd Diameter'], color = 'royalblue', s = 100)
    plt.scatter(dfParameters[filOrange][r'initial $\sigma$ 3D'], dfParameters[filOrange]['min. Chd Diameter'], color = 'sandybrown', s = 100)

    plt.scatter([.53, .46, .44],[90, 180, 245], color = 'black', edgecolors= 'white', linewidths = 3, s = 250)

    plt.xlabel(r'$\sigma$ of the 3D distribution')
    plt.ylabel('minimum sphere diameter (µm)')

    return st.pyplot(plt)


##-------------------------------------##
##--- Plot 3D chd size distribution ---##
##-------------------------------------##

def chd_3D_size_distribution(mu, sigma, x_max):
    x = np.linspace(1e-5, 2000, 1000)  # Start x from a small positive value to avoid division by zero
    pdf = (1 / (x * sigma * np.sqrt(2 * np.pi))) * np.exp(- (np.log(x) - mu)**2 / (2 * sigma**2))

    plt.clf()
    plt.figure(figsize=(4, 2))
    plt.plot(x, pdf)
    plt.xlabel('chondrule size, i.e., diameter (µm)', fontsize=6)
    plt.ylabel('frequency', fontsize=6)
    plt.xlim([0,x_max])
    plt.ylim([0, 1.1*max(pdf)])
    plt.xticks(fontsize=5)
    plt.yticks([])
    # plt.grid(True)
    return st.pyplot(plt)


##------------------------------------------------------------------------------------------------------------------------------------------------##
##--- Histograms illustrating the switch of the mean 2D located to the left of the mean 3D to the mean 2D located to the right of the mean 3D  ---##
##------------------------------------------------------------------------------------------------------------------------------------------------##

def switch_plot(mu3D, sigma3Dini, numberOfChondrules, zAxisLength, maxChdSize):
    plot_labels = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']

    model_progress_bar = st.progress(0)
    loop_counter = 0

    # Clear previous plots
    plt.clf()

    fig, axs = plt.subplots(3, 3, sharex=True, sharey=True, figsize = (12.5, 7.5))
    plt.subplots_adjust(hspace=0)
    plt.subplots_adjust(wspace=0)
    
    z = 0
    for i in range(3):
        for j in range(3):
            tot_loops = 9
            loop_counter+=1
            model_progress_bar.progress(loop_counter/tot_loops, text='calculating – this may take some time')
    # the mu of the parent 3D distribution is fixed at a typical value for chondrule size distributions
    #    mu3D = 6.2
            z+=1
            dfChdList = chd3DList(numberOfChondrules, zAxisLength, mu3D, sigma3Dini * z) # noc, zAxisLen, mu, sigma
            dfChdList = dfChdList[dfChdList['Chd Diameter'] < maxChdSize]
            appChdDiameterList = sectionedChd(.5 * zAxisLength, dfChdList)
            muFit, SigmaFit = calcMuSigma(appChdDiameterList)
            muFit3D, SigmaFit3D = calcMuSigma(dfChdList['Chd Diameter'])
        
            # plotting and formatting the figure
            xAxisMax = maxChdSize
        
            axs[i,j].hist(dfChdList['Chd Diameter'], 1000, density = True, alpha = .2, label = '3D')
            axs[i,j].hist(appChdDiameterList, 1000, density = True, alpha = .4, label = '2D')
            
            # ax.set_ylabel('frequency')
            axs[i,j].legend()
            
            axs[i,j].text(.76, .6, r'$\mu$ 3D: ' + str(mu3D), horizontalalignment = 'left', transform = axs[i,j].transAxes)
            axs[i,j].text(.76, .5, r'$\sigma$ 3D: ' + str(round(.1*z, 1)), horizontalalignment = 'left', color = 'purple', transform = axs[i,j].transAxes)
            # axs[i, j].text(0.5, 0.5, 'Text', ha='center', va='center', transform=axs[i, j].transAxes)
            axs[i,j].vlines(np.e**(muFit3D + 0.5 * SigmaFit3D**2), 0, .005, linestyles = 'dashed', colors = 'blue')
            axs[i,j].vlines(np.e**(muFit + 0.5 * SigmaFit**2), 0, .005, linestyles = 'dashed', colors = 'red')
        
            axs[i,j].set_xlim(0, xAxisMax)
            axs[i,j].set_ylim(0, .005)

            axs[i,j].text(50, .0045, plot_labels[z-1])

    model_progress_bar.empty()
    
    for j, ax in enumerate(axs[-1]):
        if j == 1:
            ax.set_xlabel('sphere size (µm)', fontsize=14)
        else:
            ax.set_xlabel('')  # Empty string for x-axis label
    
    for ax in axs[-1]:
        ax.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=True)

    st.write('preparing plots ... (this might take a few seconds)')
    for i, ax_row in enumerate(axs):
        for j, ax in enumerate(ax_row):
            if i == 1 and j == 0:
                ax.set_ylabel('frequency', rotation=90, fontsize=14) # or: normalised abundances
                ax.yaxis.set_ticks_position('none')  # Remove ticks
            else:
                ax.set_yticklabels([])  # Remove tick labels
                ax.set_yticks([])  # Remove ticks

    return st.pyplot(fig)


##-------------------------------##
##--- Plot sigma2D vs sigma3D ---##
##-------------------------------##

def sigma2D_vs_sigma3D(sel_mu3D, dfPara):
    # dfPara = pd.read_csv('chondrules 2D-3D distributions results file.csv')
    # dfPara = df_results
    # selParam = dfPara.columns.tolist()
    # xAxis = selParam[12]
    # yAxis = selParam[5]

    minChdSize_min = min(dfPara[r'min. Chd Diameter'])
    minChdSize_max = max(dfPara[r'min. Chd Diameter'])
    maxChdSize_min = min(dfPara[r'max. Chd Diameter'])
    maxChdSize_max = max(dfPara[r'max. Chd Diameter'])
    xAxis = r'$\sigma$ Fit 2D'
    yAxis = r'initial $\sigma$ 3D'
    
    plt.clf()
    # plt.text(.32, .97, 'a', fontsize=14)
    plt.text(.32, .97, f'µ 3D: {sel_mu3D}', c='dimgrey')
    plt.text(.83, .67, 'y = 1.19 * x - 0.27', rotation=31)

    x = np.linspace(0, 1.7, 50)
    plt.plot(x, 1.19 * x - .27, color = 'black', linestyle='--', lw = 1)

    for mark, maxChdSize in [['o', maxChdSize_min], ['d', maxChdSize_max]]: # [['o', 1000], ['v', 2000], ['d', 4000]]:
        for mu3D in [sel_mu3D]:
            for col, minChdSize in [['royalblue', minChdSize_min], ['sienna', minChdSize_max]]:
                fil = (dfPara['min. Chd Diameter'] == minChdSize) & (dfPara['max. Chd Diameter'] == maxChdSize) & (dfPara[r'initial $\mu$ 3D'] == mu3D)
                if maxChdSize == 2000:
                    plt.scatter(dfPara[fil][xAxis], dfPara[fil][yAxis], label=minChdSize, marker=mark, s=25, c=col)
                plt.scatter(dfPara[fil][xAxis], dfPara[fil][yAxis], label=minChdSize, marker=mark, s=25, c=col, alpha=.5)

    lower_rect = patches.Rectangle((0, 0), 0.65, 1.19 * .65 - .27, facecolor='black', alpha=.05)
    plt.gca().add_patch(lower_rect)

    plt.xlim(.3, 1)
    plt.ylim(0, 1.05)
    plt.xlabel('σ of the 2D size distribution', fontsize=14)
    plt.ylabel('σ of the 3D size distribution', fontsize=14)
    plt.legend(title=r'$\mu$ 3D', loc='lower right')
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)


    legend_handles = [
        plt.Line2D([], [], color='sienna', linestyle='-', label=f'min chd size: {minChdSize_max}'),
        plt.Line2D([], [], color='royalblue', linestyle='-', label=f'min chd size: {minChdSize_min}'),
        plt.Line2D([], [], color='black', marker='o', linestyle='None', markersize=5, label=f'max chd size: {maxChdSize_max}'),
        plt.Line2D([], [], color='black', marker='d', linestyle='None', markersize=5, label=f'max chd size: {maxChdSize_min}')
    ]

    plt.legend(handles=legend_handles, loc='lower right') #, title='legend title')
    # return st.write(dfPara[xAxis], dfPara[yAxis])
    return st.pyplot(plt)


##----------------------------##
##--- Plot mu2D vs mu3D ---##
##----------------------------##

def mu2D_vs_mu3D(sel_sigma, dfPara):
    # dfPara = pd.read_csv('chondrules 2D-3D distributions results file.csv')
    # dfPara = df_results
    # selParam = dfPara.columns.tolist()
    # xAxis = selParam[11]
    # yAxis = selParam[4]

    minChdSize_min = min(dfPara[r'min. Chd Diameter'])
    minChdSize_max = max(dfPara[r'min. Chd Diameter'])
    maxChdSize_min = min(dfPara[r'max. Chd Diameter'])
    maxChdSize_max = max(dfPara[r'max. Chd Diameter'])
    xAxis = r'$\mu$ Fit 2D'
    yAxis = r'initial $\mu$ 3D'

    plt.clf()
    # plt.text(5.84, 6.56, 'b', fontsize=14)
    plt.text(5.84, 6.56, f'$\sigma$ 3D: {sel_sigma}', c='dimgrey')
    plt.text(5.98, 6.02, '1:1 line', c='dimgrey', rotation=50)
    x = np.linspace(6, 6.6, 50)
    plt.plot(x, x, linestyle=':', c='grey', lw = 1)

    for sigma in [sel_sigma]: #[.02] + [x/10 for x in range(2, 20, 2)] + [1.98]:
        for l_style, maxChdSize in [['-', maxChdSize_min], ['-.', maxChdSize_max]]:
            for col, minChdSize in [['royalblue', minChdSize_min], ['sienna', minChdSize_max]]:
                fil = (dfPara['min. Chd Diameter'] == minChdSize) & (dfPara['max. Chd Diameter'] == maxChdSize) & (dfPara[r'initial $\sigma$ 3D'] == sigma)
                plt.plot(dfPara[fil][xAxis], dfPara[fil][yAxis], linestyle=l_style, color=col, label=sigma)

    plt.xlim(5.8, 6.8)
    plt.ylim(5.98, 6.62)
    plt.xlabel('μ of the 2D size distribution', fontsize=14)
    plt.ylabel('μ of the 3D size distribution', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    legend_handles = [
        plt.Line2D([], [], color='royalblue', linestyle='-', label=f'min chd size: {minChdSize_max}'),
        plt.Line2D([], [], color='sienna', linestyle='-', label=f'min chd size: {minChdSize_min}'),
        plt.Line2D([], [], color='black', linestyle='-', label=f'max chd size: {maxChdSize_max}'),
        plt.Line2D([], [], color='black', linestyle='-.', label=f'max chd size: {maxChdSize_min}')
    ]
    plt.legend(handles = legend_handles, loc='lower right')
    return st.pyplot(plt)