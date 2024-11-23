import streamlit as st
import utils.func
import pandas as pd

if 'dfParameters' not in st.session_state:
    st.session_state.dfParameters = pd.read_csv('chondrules 2D-3D distributions results file.csv')

st.title('Revealing the relationship between 2D and 3D size-frequency distributions')

tab1, tab2, tab3, tab4, tab5 = st.tabs(['Intro', 'pre-calculated parameters', 'own parameters', 'distributions', 'abstract'])
# the mu of the parent 3D distribution is fixed at a typical value for chondrule size distributions
# taken from the metzler data fits below

with tab1:
    st.markdown('''**This is the companion app to**  
                Hezel et al. 2025. Revealing the relationship between 2D and 3D
                chondrule size-frequency distribution in a meteorite. *Meteoritics & Planetary Sciences* (re-submitted)  
                The abstract of this paper can be found in the last tab of this app.
                ''')
    st.markdown('''
                ''')

    st.subheader('How this works')
    st.markdown('<div style="color: grey">Some more detailed explanations are in the works.</div>', unsafe_allow_html=True)
    st.markdown('<h5>pre-calculated parameters</h5>', unsafe_allow_html=True)
    st.markdown('See how Fig. 3 and Fig. 5a,b change in response to changing the max. chondrule diameter, the initial µ3D, and the inital $\sigma$.')

    st.markdown('<h5>own parameters</h5>', unsafe_allow_html=True)
    st.markdown('Here any parameter for initial chondrule size distribution can be defined, and then seen how this affects Fig. 3 and Fig. 5 a,b.')

    st.markdown('<h5>distributions</h5>', unsafe_allow_html=True)
    st.markdown('''Here the general shape of a log-normal distributin can be explored to see what parameters are sensible for a chondrule size-frequency distribution.  
                Using these parameters, Fig. 4 can then be produced to see how the mean of the 3D and corresponding 2D size-freuquency distributions change.''')


with tab2:
    st.markdown('''The two areas in which the mean of the 2D distribution is smaller (blue) or larger (orange)
                than the mean of the 3D distribution. Black points are data from measured ordinary chondrites.''')

    df_results = pd.read_csv('chondrules 2D-3D distributions results file.csv')

    col1, col2, col3 = st.columns(3)
    with col1:
        max_chd_size = st.selectbox('max. chd. diameter', df_results['max. Chd Diameter'].unique(), index=1)
    with col2:
        sel_mu3D = st.selectbox('initial $\mu$ 3D', df_results['initial $\mu$ 3D'].unique(), index=1)
    with col3:
        sel_sigma = st.selectbox('initial $\sigma$', df_results['initial $\sigma$ 3D'].unique(), index=11)

    col1, col2, col3 = st.columns(3)
    with col1:
        utils.func.model_plot(max_chd_size, sel_mu3D, df_results)
    with col2:
        utils.func.sigma2D_vs_sigma3D(sel_mu3D, df_results)
    with col3:
        utils.func.mu2D_vs_mu3D(sel_sigma, df_results)

with tab3:
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
        # st.session_state.df_results_own = st.session_state.dfParameters

    if st.session_state.dfParameters is not None:
        with st.expander('Results Table'):
            st.dataframe(st.session_state.dfParameters)

        col1, col2, col3 = st.columns(3)
        with col1:
            max_chd_size_own_model = st.selectbox('max. chd. diameter', st.session_state.dfParameters['max. Chd Diameter'].unique(), index=1, key='max_chd_size_own_model')
        with col2:
            sel_mu3D_own_model = st.selectbox('initial $\mu$ 3D', st.session_state.dfParameters['initial $\mu$ 3D'].unique(), index=1, key='sel_mu3D_own_model')
        with col3:
            sel_sigma_own_model = st.selectbox('initial $\sigma$', st.session_state.dfParameters['initial $\sigma$ 3D'].unique())

        col1, col2, col3 = st.columns(3)
        with col1:
            utils.func.model_plot(max_chd_size_own_model, sel_mu3D_own_model, st.session_state.dfParameters)
        with col2:
            utils.func.sigma2D_vs_sigma3D(sel_mu3D_own_model, st.session_state.dfParameters)
        with col3:
            utils.func.mu2D_vs_mu3D(sel_sigma_own_model, st.session_state.dfParameters)


with tab4:
    st.markdown('Explore how the 3D chondrule size distribution changes in response to its defining parameters 2 parameters **µ 3D** and **σ initial**.')

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        mu3D_switch = st.number_input('µ 3D', value=6.2, step=.2)
    with col2:
        sigma3Dini_switch = st.number_input('σ initial', value=.1, step=.1)
    with col3:
        maxChdSize_switch = st.number_input('max chd size', value=1600, step=200, max_value=2000)
    with col4:
        numberOfChondrules_switch = st.number_input('nr of chd 10^x', value=5, step=1)
        numberOfChondrules_switch = 10**numberOfChondrules_switch
    with col5:
        zAxisLength_switch = st.number_input('length of z-axis 10^x', value=4, step=1)
        zAxisLength_switch = 10**zAxisLength_switch

    utils.func.chd_3D_size_distribution(mu3D_switch, sigma3Dini_switch, maxChdSize_switch)

    st.divider()
    st.markdown('''Histograms illustrating the switch of the mean 2D located to the **left** of the mean 3D to the mean 2D located to the **right** of the mean 3D  
                As the following plots iterate through **σ initial**, **σ initial** is the only value not taken from the selections in the dropdown menus above.''')

    if st.button('show distributions'):
        utils.func.switch_plot(mu3D_switch, .1, numberOfChondrules_switch, zAxisLength_switch, maxChdSize_switch)

with tab5:
    st.subheader('Abstract')
    st.markdown('''	Chondrule size-frequency distributions provide important information to understand the origin of chondrules. Size-frequency distributions are often obtained as apparent 2D size-frequency distributions in thin sections, as determining a 3D size-frequency distribution is notoriously difficult. The relationship between a 2D size-frequency distribution and its corresponding 3D size-frequency distribution has been previously modelled, however, the results contradict measured results. Models so far predict a higher mean of the 2D size-frequency distribution than the corresponding mean of the 3D size-frequency distribution, while measurements of real chondrule populations show the opposite. Here we use a new model approach that agrees with these measurements and at the same time offers a solution, why models so far predicted the opposite. Our new model provides a tool with which the 3D chondrule size-frequency distribution can be determined from the fit of a measured 2D chondrule size-frequency distribution.
                ''')
    st.markdown('''Hezel et al. 2025. Revealing the relationship between 2D and 3D
                chondrule size-frequency distribution in a meteorite. *Meteoritics & Planetary Sciences* (re-submitted)''')