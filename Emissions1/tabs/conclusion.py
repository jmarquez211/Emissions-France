import streamlit as st

title = "Conclusion"

sidebar_name = "Conclusion"

def run():
    
    
    st.markdown(
        
    """
    
    In conclusion, our project focused on the prediction of CO2 emissions in France and its implications 
    in the automotive industry. We discovered that France stands out as one of the countries with lower 
    pollution emissions compared to its neighboring countries and major global powers.

    Through our analysis, we found that while the overall quality of car manufacturers is commendable, 
    there are instances where certain car models exceed the permissible emission limits. However, it is 
    worth noting that, based on the classification we established using labels such as A, B, C, etc., 
    the majority of cars in France comply with European regulations.

    We also conducted a comparison between the Lasso and Ridge regression models to predict CO2 emissions. 
    Both models demonstrated their effectiveness in capturing the relationships between various features and 
    emission levels. However, the Lasso regression model, with its ability to perform feature selection, 
    provided valuable insights into the most influential factors affecting emissions.

    Furthermore, our analysis revealed a positive trend in the automotive industry, with an increasing 
    number of cars being designed and manufactured to emit lower levels of pollutants. This indicates a 
    proactive approach by car manufacturers to contribute to environmental sustainability and address the challenges of climate change.

    In summary, our project shed light on the importance of monitoring and regulating CO2 emissions in the automotive sector. 
    It highlighted the need for continued efforts from both regulatory bodies and car manufacturers to ensure compliance with
    emission standards and promote the development of eco-friendly vehicles. By leveraging predictive modeling techniques, 
    such as Lasso and Ridge regressions, we can gain valuable insights for better decision-making and contribute to a cleaner
    and greener future.
    
    
    ### THAT'S ALL FOLKS
    
    
    
    
    """, unsafe_allow_html=True
    )