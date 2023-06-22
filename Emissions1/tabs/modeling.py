import streamlit as st, pandas as pd, numpy as np

from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.linear_model import Ridge, LassoCV
from sklearn.metrics import mean_squared_error, r2_score

import plotly.graph_objects as go



title = "Testing the model"
sidebar_name = "Models"

#comment


def run():
    
    st.subheader('Training the models')
    st.write('---')
    st.markdown(''' 
                As commented before, the main gol of this project is to predict emissions. 
                In order to do so, we have implemented two models of regression: Lasso and Ridge regression.
                Both techniques help prevent overfitting and improve model performance. The choice depends on the 
                problem and desired model characteristics. In this project we have proved both and compare them. 
                
                Lasso regression adds a penalty term to the loss function using L1 regularization. It encourages sparsity by driving 
                some coefficients to exactly zero, effectively performing feature selection.
                Ridge regression adds a penalty term to the loss function using L2 regularization. It shrinks the coefficients towards 
                zero without forcing them to be exactly zero.
                
                ''') 
    
    st.markdown('''
                Continue with the models, we are going to display the performance of each model.
                ''')
    
    df = pd.read_csv('Emissions1/tabs/df_r.csv')
    df = df.sample(frac=0.5)
    target = df['Ewltp (g/km)']
    feats = df.drop('Ewltp (g/km)', axis=1)
    
    X_train, X_test, y_train, y_test = train_test_split(feats, target, test_size=0.2, random_state=42)

    #Create a function rmse_cv, which for a given model calculates the Root Mean Square Error (RMSE) obtained by
#cross-validation at 5 samples, using the function cross_val score of sklearn.model_selection.

    def rmse_cv(model, data, target):
        rmse= np.sqrt(-cross_val_score(model, target, feats, scoring="neg_mean_squared_error", cv=5))
        return(rmse)
    
    coefs = []
    rmse  = []
    alphas = [0.01, 0.05 , 0.1, 0.3, 0.8, 1, 5,10, 15, 30,50]
    for a in alphas:
        ridge = Ridge(alpha=a, fit_intercept=True, random_state=42)
        ridge.fit(X_train, y_train)
        pred = ridge.predict(X_test)
        coefs.append(ridge.coef_)
        rmse.append(mean_squared_error(y_test, pred))
        
    st.write('---')
    st.markdown('The scoring of the model *Ridge* without using a technique to select the best parameter are:')
        
    train_score_ridge = ridge.score(X_train, y_train)
    test_score_ridge = ridge.score(X_test,y_test)
    
    #to modify how it appears in the screen, we add the next commands
    #it's CSS style, and we add the format to see three decimal numbers. The command unsafe_allow_html is for showing the result
        
    st.write("The train score for ridge model is <span style='color:#00cc00;'>{:.3%}</span>".format(train_score_ridge), unsafe_allow_html=True)
    st.write("The test score for ridge model is <span style='color:#00cc00;'>{:.3%}</span>".format(test_score_ridge), unsafe_allow_html=True)
    
    cv_ridge = pd.Series(rmse, index=alphas)
    fig_ridge = go.Figure(data=go.Scatter(x=alphas, y=cv_ridge, mode='lines'))
    fig_ridge.update_layout(title="rmse in function of alpha", xaxis_title="alpha", yaxis_title="rmse")
    
    
    st.plotly_chart(fig_ridge)
    
    st.markdown(''' 
                The higher the value of alpha, the higher the total error.
                The optimal parameter here is $\\alpha$ = 5. This is the one that gives the smallest RMSE error.

                ''')
    
    model_ridge = Ridge(alpha = 5).fit(X_train, y_train)
    train_score_model_ridge = model_ridge.score(X_train, y_train)
    test_score_model_ridge = model_ridge.score(X_test, y_test)
    st.write("The train score for ridge model is <span style='color:#00cc00;'>{:.3%}</span>".format(train_score_model_ridge), unsafe_allow_html=True)
    st.write("The test score for ridge model is <span style='color:#00cc00;'>{:.3%}</span>".format(test_score_model_ridge), unsafe_allow_html=True)
    
    #st.markdown('Lets see the performance of the Lasso regression.')
    
    #Create a Lasso regression model that will choose the parameter a by cross validation from [10, 1, 0.1, 0.1,
    #0.001, 0.0005], using the LassoCV function
    lasso_reg = LassoCV(alphas = [10, 1, 0.1, 0.1, 0.001, 0.0005], random_state=0).fit(X_train, y_train)
    
    #st.write("The train score for Lasso model is <span style='color:#00cc00;'> {:.3%}</span>".format(lasso_reg.score(X_train,y_train)), unsafe_allow_html=True)
    #st.write("The test score for Lasso model is <span style='color:#00cc00;'>{:.3%}</span>".format(lasso_reg.score(X_test,y_test)), unsafe_allow_html=True)

    from sklearn.linear_model import Lasso

    alphas_lasso = [10, 1, 0.1, 0.1, 0.001, 0.0005]
    for a in alphas_lasso:
        model_lasso = Lasso(alpha=a)
        model_lasso.fit(X_train, y_train)
        
    st.write('---')
    
    st.markdown(''' 
                What about Lasso regression? We know this model regression eliminate some variables kept and the
                number of variables that are significantly important to the target.
                ''')
    
    alphas = [10, 1, 0.1, 0.1, 0.001, 0.0005]
    for a in alphas:
        model_lasso = Lasso(alpha = a).fit(X_train, y_train)
        coef_lasso = pd.Series(model_lasso.coef_, index = X_train.columns)
    
    st.write("Lasso picked " + str(sum(coef_lasso != 0)) + " variables and eliminated the other " + str(sum(coef_lasso == 0)) + " variables")
    
    imp_coef = pd.concat([coef_lasso.sort_values().head(10),
                    coef_lasso.sort_values().tail(10)])
    
    fig_lasso = go.Figure(data=[go.Bar(y=imp_coef.index, x=imp_coef.values, orientation='h')])
    fig_lasso.update_layout(title="Coefficients in the Lasso Model", yaxis=dict(title="Variables"))
    fig_lasso.update_layout(width=800, height=700)
    
    st.plotly_chart(fig_lasso)
    
    st.subheader('Which is the best model?')
    
    st.markdown('''
                To answer this question, we need to use a technique for calculating the best parameters.
                We use **GridSearchCV**.
                GridSearchCV is a powerful tool in machine learning for hyperparameter tuning. 
                It systematically searches through a grid of hyperparameter values to find the optimal 
                combination that results in the best model performance. Applying GridSearchCV to Ridge 
                and Lasso regressions allows us to find the best regularization parameter ($\\alpha$) that 
                balances model complexity and performance, improving the predictive accuracy and interpretability
                of the models.
                ''')
    
    from sklearn.model_selection import GridSearchCV
    
    lasso_params = {'alpha':[10,1,0.1,0.1,0.001,0.0005]}
    ridge_params = {'alpha':[0.01,0.05,0.1,0.3,0.8,1,5,10,15,30,50]}

    grid1 = GridSearchCV(Lasso(),param_grid=lasso_params).fit(X_train,y_train)

    grid2 = GridSearchCV(Ridge(),param_grid=ridge_params).fit(X_train,y_train)

    models2 = {'Lasso': grid1,
            'Ridge' : grid2}
    
    st.write('Best parameters for Lasso: ', grid1.best_estimator_)
    st.write('\n')
    st.write('Best parameters for Ridge: ', grid2.best_estimator_)
    st.write('Then, the best score for both models are: ')
    
    st.write("Best score for Lasso: <span style='color:#00cc00;'>{:.4%}</span>" .format(grid1.best_score_),unsafe_allow_html=True)
    st.write('\n')
    st.write("Best score for Ridge: <span style='color:#00cc00;'>{:.4%}</span>".format(grid2.best_score_), unsafe_allow_html=True)
    
    st.write('---')
    
    st.markdown('''
                The predictions for both models are represented in the next two columns. 
                The column on the left are the predictions for the *ridge* model. The
                column on the right are the predictions for the *Lasso* model.
                ''')
    
    y_pred_lasso = model_lasso.predict(X_test)
    y_pred_ridge = model_ridge.predict(X_test)
    
    
    predictions_df = pd.DataFrame({
    'Index': X_test.index,
    'Lasso Prediction': y_pred_lasso,
    'Ridge Prediction': y_pred_ridge
})
    #the command above is for customizing the results

    col_r, col_l = st.columns(2)

    col_r.write(predictions_df['Ridge Prediction'])
    col_l.write(predictions_df['Lasso Prediction'])
    
    #st.write(target)
    
    st.write('---')
    
    st.write('Result and precision of predictions:')
    
    

    # metrics for Lasso prediction
    rmse_lasso = np.sqrt(mean_squared_error(y_test, y_pred_lasso))
    r2_lasso = r2_score(y_test, y_pred_lasso)

    # metrics for Ridge prediction
    rmse_ridge = np.sqrt(mean_squared_error(y_test, y_pred_ridge))
    r2_ridge = r2_score(y_test, y_pred_ridge)

    # showing the metrics
    
    st.write("Lasso model - RMSE: <span style='color:#00cc00;'>{:.4}</span>".format(rmse_lasso), unsafe_allow_html=True)
    st.write("Lasso model - score: <span style='color:#00cc00;'>{:.3%}</span>".format(r2_lasso), unsafe_allow_html=True)
    st.write("Ridge model - RMSE: <span style='color:#00cc00;'>{:.4}</span>".format(rmse_ridge), unsafe_allow_html=True)
    st.write("Ridge model - score: <span style='color:#00cc00;'>{:.3%}</span>".format(r2_ridge), unsafe_allow_html=True)
    
    st.write('---')
    st.markdown('''
                To sump up, both models are great to make predictions in this dataset. We select Ridge
                because it has more precision than Lasso, although it is only slightly better, this model is 
                better for dataset that have several features correlated and that can be important, almost most of them.
                
                ''')
    
    

    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
   
    
     
     
        
       

          
                
            
                
        
       
           
        
       
                
        
            
                
                
