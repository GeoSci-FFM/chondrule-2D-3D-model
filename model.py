import streamlit as st
import utils.func

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
# import os
# current_dir = os.getcwd()

st.session_state.dfParameters = pd.read_csv('chondrules 2D-3D distributions results file.csv')

st.title('2D/3D model')

tab1, tab2, tab3 = st.tabs(['explore parameters', 'build your model', 'switch'])
# the mu of the parent 3D distribution is fixed at a typical value for chondrule size distributions
# taken from the metzler data fits below

with tab1:
    st.subheader('explore')

    df_results = pd.read_csv('chondrules 2D-3D distributions results file.csv')

    col1, col2 = st.columns(2)
    with col1:
        max_chd_size = st.selectbox('max. chd. diameter', df_results['max. Chd Diameter'].unique(), index=1)
    with col2:
        sel_mu3D = st.selectbox('initial $\mu$ 3D', df_results['initial $\mu$ 3D'].unique(), index=1)
    
    utils.func.model_plot(max_chd_size, sel_mu3D, df_results)

with tab2:
    col1, col2, col3 = st.columns(3)
    with col1:
        mu3DStart = st.number_input('**µ 3D mean** start', value=6.2, step=.2, key='mu3Dstart')
    with col2:
        mu3DSEnd = st.number_input('end', value=6.4, step=.2, key='mu3Dend')
    with col3:
        mu3Dstep = st.number_input('steps', value=.2, step=.2, key='mu3Dstep')
    mu3DList = [x/10 for x in range(int(mu3DStart*10), int((mu3DSEnd+.2)*10), int(mu3Dstep*10))]

    col1, col2, col3 = st.columns(3)
    with col1:
        iniSigmastart = st.number_input('**σ initial** start', value=.4, step=.05, key='iniSigmastart')
    with col2:
        iniSigmaend = st.number_input('end', value=.6, step=.05, key='iniSigmaend')
    with col3:
        iniSigmastep = st.number_input('steps', value=.1, step=.1, key='iniSigmastep')
    iniSigma_List = [x/10 for x in range(int(iniSigmastart*10), int((iniSigmaend+.1)*10), int(iniSigmastep*10))]

    col1, col2, col3 = st.columns(3)
    with col1:
        minChdSize_rangestart = st.number_input('**min chd size** start', value=100, step=50, key='minChdSize_rangestart')
    with col2:
        minChdSize_rangeend = st.number_input('end', value=200, step=50, key='minChdSize_rangeend')
    with col3:
        minChdSize_rangestep = st.number_input('steps', value=50, step=50, key='minChdSize_rangestep')
    minChdSize_range = [x/10 for x in range(int(minChdSize_rangestart*10), int((minChdSize_rangeend+.1)*10), int(minChdSize_rangestep*10))]

    col1, col2, col3 = st.columns(3)
    with col1:
        maxChdSize_rangestart = st.number_input('**max chd size** start', value=1500, step=100, key='maxChdSize_rangestart')
    with col2:
        maxChdSize_rangeend = st.number_input('end', value=2000, step=100, key='maxChdSize_rangeend')
    with col3:
        maxChdSize_rangestep = st.number_input('steps', value=500, step=100, key='maxChdSize_rangestep')
    maxChdSize_range = [x/10 for x in range(int(maxChdSize_rangestart*10), int((maxChdSize_rangeend+.1)*10), int(maxChdSize_rangestep*10))]
   
    col1, col2 = st.columns(2)
    with col1:
        numberOfChondrules_power = st.number_input('nr. of chondrules (10^x)', value=5, step=1)
    with col2:
        zAxisLength_power = st.number_input('length of z-axis (10^x)', value=4, step=1)
    numberOfChondrules = 10**numberOfChondrules_power
    zAxisLength = 10**zAxisLength_power

    model_parameter_names = ['µ 3D', 'σ initial', 'min chd size', 'max chd size', 'nr of chd', 'length y-axis']
    parameter_table = pd.DataFrame([mu3DList, iniSigma_List, minChdSize_range, maxChdSize_range, [numberOfChondrules], [zAxisLength]])
    parameter_table.insert(0, 'parameter', model_parameter_names)

    st.dataframe(parameter_table)


    if st.button('calculate'):
            utils.func.run_model(mu3DList, iniSigma_List, minChdSize_range, maxChdSize_range, numberOfChondrules, zAxisLength)
            # st.write('finished')

    with st.expander('Results Table'):
        st.dataframe(st.session_state.dfParameters)


    df_results_own = st.session_state.dfParameters

    col1, col2 = st.columns(2)
    with col1:
        max_chd_size_own_model = st.selectbox('max. chd. diameter', df_results_own['max. Chd Diameter'].unique(), index=1, key='max_chd_size_own_model')
    with col2:
        sel_mu3D_own_model = st.selectbox('initial $\mu$ 3D', df_results_own['initial $\mu$ 3D'].unique(), index=1, key='sel_mu3D_own_model')

    utils.func.model_plot(max_chd_size_own_model, sel_mu3D_own_model, df_results_own)


with tab3:
    st.subheader('Demonstrating the switch from the mean 3D being higher to mean 2D being higher')

    mu3D = 6.2
    sigma3Dini = .1
    numberOfChondrules = 10**6
    zAxisLength = 10**4
    maxChdSize = 1500

    if st.button('show plots'):
        utils.func.switch_plot(mu3D, sigma3Dini, numberOfChondrules, zAxisLength, maxChdSize)
