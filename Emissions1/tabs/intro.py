import streamlit as st

title = "Prediction of emissions"
sidebar_name = "Introduction"

def run():
    
    st.markdown('<style> section[tabindex="0"] div[data-testid="stVerticalBlock"] > div:nth-child(3) div[data-testid="stImage"] {border-top: 8px solid var(--red-color); border-right: 8px solid var(--red-color); border-top-right-radius: 23px; margin: auto;} section[tabindex="0"] div[data-testid="stVerticalBlock"] div[data-testid="stImage"] > img {border-top-right-radius: 15px;} section[tabindex="0"] div[data-testid="stVerticalBlock"] > div:nth-child(3) button[title="View fullscreen"] {display: none;} section[tabindex="0"] div[data-testid="stVerticalBlock"] div[data-testid="stImage"] {border-top: 3px solid var(--red-color); border-right: 3px solid var(--red-color); border-top-right-radius: 18px; margin: auto;}</style>', unsafe_allow_html=True)
    
    st.title('Predicting CO2 emissions')
    st.image('tabs/car emission.png')
    
    st.header('Emissions of cars in France in 2019')
    
    st.markdown(
        
        """ 
        
        ## Context
        ---
        The main goal of the project assigned is to determinate how all the characteristics presented in cars affect the emissions of CO2. 
        One must understand the features of cars as well as the relation among them.
        
        From an economic point of view, it is interesting to find the brands that generate more pollution to reduce the manufacturing so the companies will not get any kind of fine from the government.
        Scientifically speaking, it is desirable to have an eco-friendly environment because it is well known CO2 emissions are extremely dangerous for human health, then understanding the reason why one vehicle’s brand contaminates more than other, one will focus on those cars with a tendency of high pollution.
 
                
                
        
                
                
        
        """, unsafe_allow_html=True
                
                
                
                
                
                )
    
    st.markdown(
    """
    ## Objective
    ---
    
    The purpose of the project is to study the features of every brand of vehicle and to analyze how each one emits more emissions. Some of them will have a huge impact, 
    those are the main features to analyze.
    
    """
    )
    
    st.markdown(
    """
    ### Relevance
    
    
    The principal variable for the objective is the one which stores the emissions. Furthermore, the weight of vehicles, the width of wheels, 
    fuel consumption and others are highly correlated with emissions. 

    The target variable is ‘Ewltp (g/km)’ which indicates how many grams per kilometer a vehicle emits to the atmosphere.


    The only limitation one can find is that not all features offer information and others are completely empty.

    """
    )
    
    

    st.write('Part of the information can be found here:')
    st.markdown('[CO2 emissions datasets](https://ourworldindata.org/co2-dataset-sources)')
    
    st.markdown('''
                
                ## Sections
                
                This project is divided into three parts. The first section consists of
                investigating what features are more significant and relevant for the goal. The second part is reserved
                to train models in order to know whic one performs better. The last one is a section
                made up exclusively for the user. In this section it is provided features to select and models to train and predict 
                considering the variables of the car.
                
                ''')
