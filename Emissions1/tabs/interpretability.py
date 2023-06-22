import pandas as pd
import streamlit as st



title = 'Interpretability'
sidebar_name = 'Interpretability'



def run():
    
    st.header('Interpretability')
    
    st.markdown('''
                Interpretability in data science refers to the ability to understand and explain 
                the decisions and predictions made by machine learning models. It involves gaining 
                insights into how the model arrives at its conclusions, understanding the underlying 
                factors that contribute to those conclusions, and communicating those insights to stakeholders.
                
                Various techniques and tools are available for interpreting machine learning models, such as
                feature importance analysis, partial dependence plots, SHAP values, and surrogate models. 
                These techniques aim to provide a clear understanding of how the model works and enable 
                effective communication of insights to stakeholders.
                
                ''')
    
    df = pd.read_csv('tabs/df_r.csv',low_memory=False)
    df = df.sample(frac=0.1)
    df.dropna()
    
    st.header('What is it shap?')
    st.markdown('''
                In the field of machine learning and data science, understanding the factors that contribute to model predictions is essential. 
                SHAP (SHapley Additive exPlanations) is a powerful library that provides interpretability and explainability to machine learning models. 
                It helps us understand how different features or variables impact the model's predictions, offering insights into the model's decision-making process.

    In our web app, we are focused on predicting the "Ewltp (g/km)" variable, which represents the CO2 emissions of cars in France. By leveraging the SHAP library, 
    we can visualize and analyze the impact of various features on these emissions.

    One of the key features of SHAP is the ability to generate summary plots. These plots provide an overview of the most important features and their corresponding SHAP
    values. By examining the summary plot, we can quickly identify the features that have the greatest influence on CO2 emissions. This information can be crucial for 
    policymakers, car manufacturers, and individuals who are concerned about environmental sustainability and want to make informed decisions.

    Additionally, SHAP allows us to create individual feature importance plots, which provide a detailed view of how each feature contributes to the predicted CO2 
    emissions for a specific car. These plots help us understand the specific impact of each feature and can be used to identify areas for improvement or further 
    investigation.

    By integrating the SHAP library into our web app, we empower users to gain a deeper understanding of the factors driving CO2 emissions in the automotive sector. 
    This information can support informed decision-making and inspire actions towards reducing carbon footprints and promoting sustainable transportation.

    In summary, SHAP is a valuable tool for interpreting and explaining the predictions of machine learning models. By visualizing the contributions of different 
    features, we can gain insights into the factors influencing CO2 emissions in the context of car data in France.
                
                
                ''')
    
    st.markdown('With this information, it is time to show what features have more impact on the emissions')
    
    st.image('tabs/shap values.png')
    st.markdown('''
                The graph above show the impact of each feature to the target. 
                The color red indicates that it has important impact on the emissions, while the color
                blue indicates the contrary. Nevertheless, how can one interpret all this?
                
                The x-axis stands for SHAP value, while the y-axis represents all the features. Every 
                point on the chart indicates one SHAP value for a prediction. We can conclude that higher value of
                ‘Fuel consumption’ leads to higher prediction of emitting high volume of CO2. The same is for the 
                other features related to the mass of the car and its horsepower.
                The next grahp will clear it up
                
                ''')
    
    st.write('---')
    st.image('tabs/shap1.png')

    st.markdown('''
                According to the next figure, we can see the summary plot to see the importance of the variables 
                and how they influence the target. As expected, the fuel consumption has a big impact. Surprisingly, 
                the Mt variable is the second one with a huge impact. Mt stands for “mass in running order”, it corresponds
                to the mass with passengers or any object inside the car. Of course, the more weight the car carries, the more
                pollution it emits.
                
                ''')
    
   
    
    
    
    
