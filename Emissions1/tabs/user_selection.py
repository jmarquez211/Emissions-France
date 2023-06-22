import streamlit as st, pandas as pd, numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso

from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score

import plotly.express as px
import plotly.graph_objects as go

from joblib import dump, load


title = "The user selection"
sidebar_name = "User selection"






# Mostrar la interfaz en la web app
def run():
    st.title("Model Selection")
    
    st.markdown(''' 
                Not it's time for the user to select what model wants to use and its parameters.
                The fisrt part consist of using several parameters for Ridge and Lasso regressions. The other models 
                do not need to specify it.
                ''')
    
    st.write('---')
    
    df = pd.read_csv('df_r.csv')
    #df_r is the data set clean and preprocessed
    #df = df.sample(frac=0.7, random_state=42)
    target = df['Ewltp (g/km)']
    feats = df.drop('Ewltp (g/km)', axis=1)
    
    X_train, X_test, y_train, y_test = train_test_split(feats, target, test_size=0.2,random_state=42)

    # alphas for Ridge & Lasso
    alphas_ridge = [0.01, 0.05, 0.1, 0.3, 0.8, 1, 5, 10, 15, 30, 50]
    alphas_lasso = [10, 1, 0.1, 0.01, 0.001, 0.0005]



    # a function to select the model
    def get_model(model_name):
        
        if model_name == "Ridge":
            alpha = st.selectbox("Select alpha", alphas_ridge)
            return Ridge(alpha=alpha)
        elif model_name == "Lasso":
            alpha = st.selectbox("Select alpha", alphas_lasso)
            return Lasso(alpha=alpha)
        

    # Function to obtain the metric selected by the user
    def get_scoring_metric(model_name):
        if model_name in ["Ridge", "Lasso"]:
            
            return st.radio("Select scoring metric", ["score", "mean_squared_error"])
        
    
    # Button to select the model
    model_name = st.selectbox("Select model", ["Ridge", "Lasso"])
    
    # obtain the model selected
    model = get_model(model_name)
    
    if model_name in ["Ridge", "Lasso"]:
        # showing the metrics for each model
        scoring_metric = get_scoring_metric(model_name)
        
        if scoring_metric:
            st.write("Selected alpha:", model.alpha)
            st.write("Selected scoring metric:", scoring_metric)
        
    # Button to show results on screen
    if st.button("Show Results"):
        
        st.write("Results:")
        
        model.fit(X_train,y_train)
        
        predictions = model.predict(X_test)
        
        if scoring_metric:
            if scoring_metric == "score" or scoring_metric == "mean_squared_error":
                score = model.score(X_test, y_test)
                st.write("Score:", score)
                
                # this is a dataframe, to show it better
                df_pred = pd.DataFrame({"Predictions": predictions})
        
                # a little changes for representation
                df_pred = df_pred.rename(columns={"Predictions": "Predictions (g/km)"})
                st.write('The predictions for every car are:')
                st.write(df_pred)
                
            elif scoring_metric == "r2_score":
                r2 = r2_score(y_test, predictions)
                st.write("R2 Score:", r2)
            
    
    st.write('---')
    
    st.markdown('''
                You can now select the features to predict the emissions:
                
                ''')
    
    # Now we can use only the principal features to predict emissions
    
    data = pd.read_csv('tabs/df_r.csv', na_values=['-', 'N/A', 'NaN'])
    data = pd.read_csv('tabs/df_r.csv', keep_default_na=False)
    data = pd.read_csv('tabs/df_r.csv', usecols=['m (kg)', 'Mt', 'W (mm)', 'ec (cm3)', 'year',
                                                      'Fuel consumption ','ep (KW)', 'Ewltp (g/km)'])
    data.rename({'Fuel consumption':'Consumption'},axis=1,inplace=True)
    
    
    data = data.dropna()
    x_d = data.drop('Ewltp (g/km)', axis=1)
    y_d = data[['Ewltp (g/km)']]
    X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(x_d,y_d, test_size=0.2, random_state=42)
    # with _d to differentiate between this and the other one
    
    def get_model2(model_name2):
        if model_name2 == 'Ridge':
            return Ridge()
        elif model_name2 == 'Lasso':
            return Lasso()
        #elif model_name2 == 'Random Forest Regressor':
           # return RandomForestRegressor()
    
    model_name2 = st.selectbox("Select model",["Ridge", "Lasso"],key='models')

    alpha = None
    
    model2 = get_model2(model_name2)
    
    if model2 in ['Ridge', 'Lasso']:
        if model2 == 'Ridge':
            alpha_deafult = 0.01
        elif model2 == 'Lasso':
            alpha_deafult = 0.0005
        alpha = st.number_input("Alpha", value=alpha_deafult)
    
    
    
    # range of values to select from the features, from minimum to maximum
    w_min, w_max = 0, 3750
    m_min, m_max = 0, 2678
    ec_min, ec_max = 0, 5038
    ep_min, ep_max = 0, 471
    mt_min, mt_max = 0, 2814
    fc_min, fc_max = 0.0,8.0
    ft_pmin, ft_pmax = 0,1

    # sliders for selecting the values
    w_value = st.slider('W (mm)', w_min, w_max)
    m_value = st.slider('m (kg)', m_min, m_max)
    ec_value = st.slider('ec (cm3)', ec_min, ec_max)
    ep_value = st.slider('ep (KW)', ep_min, ep_max)
    mt_value = st.slider('Mt', mt_min, mt_max)
    fc_value = st.slider('Fuel consumption',fc_min,fc_max,step=0.2)
    ftp = st.slider('Ft_Petrol',ft_pmin,ft_pmax)
    
    
    if st.button("Predict"):
        model2.fit(X_train_d,y_train_d)
          

        # the user will select the values, this command show the sliders
        input_data = [[w_value, m_value, ec_value, ep_value, mt_value,fc_value,ftp]]

        # making the prediction and selecting the first one
        predicted_emissions = model2.predict(input_data)[0]
        
        predicted_emissions = float(predicted_emissions)
        formatted_emissions =f"{predicted_emissions:.3f}"
        #little modification to show it better
        

        # the result
        st.write('Prediction of emissions Ewltp (g/km):', formatted_emissions)
    
    st.write('---')
    
    
    


    




                
            
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

'''
the table of contents


tablefig = go.Figure(data=[go.Table(
    header=dict(values=list(df.columns),
                fill_color='paleturquoise',
                align='left'),
    cells=dict(values=[df['m (kg)'], df['Ewltp (g/km)'], df['W (mm)'],df['At1 (mm)'], df['At2 (mm)'], df['ep (KW)'],
                       df['Mt'],df['Erwltp (g/km)'],df['ec (cm3)'],df['year']],
               fill_color='lavender',
               align='left'))
])

tablefig.show()

'''








    


