import streamlit as st, pandas as pd, numpy as np
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns

sidebar_name = "Comparation"

def run():
    st.markdown('''
            
            In this section we are going to compare the emission from other countries 
            with France
            
            ''')

    

    df = pd.read_csv('tabs/annual-share-of-co2-emissions.csv')
    
    df.rename({'Entity':'Country'},axis=1, inplace=True)
    
    #df['Annual CO₂ emissions (zero filled)'] = df['Annual CO₂ emissions (zero filled)'].apply(lambda x: '{:.0f}'.format(x))
    df['Year'] = df['Year'].apply(lambda x: '{:.0f}'.format(x))
    
    '''
    The line above is to change the format of the year, because streamlit uses scientific notation by default
    '''
    
    df.dropna(inplace=True)
    
    st.write(df.head(8))
    
    
    '''
    We have to compare meaningful values, so we create a new dataset
    
    We select the values of year since the 50's decades because before that, is irrelevant.
    We apply this for the most important countries with high industry
    
    '''
    
    df['Year'] = df['Year'].astype(int)
    df_good= df.loc[(df['Year'] >= 1955) & (df['Year'] <= 2020)]
    
    st.write('---')
    
    df_france = df_good.loc[(df_good['Country'] == 'France')]
    df_france['Year'] = df_france['Year'].apply(lambda x: '{:.0f}'.format(x)) #for changing the scientific notation format
    
    # we select the relevant years

    
    # 
    st.markdown('''Let's focus on the values for France ''')
    st.write(df_france)

    

    
    figf = go.Figure(data=go.Scatter(
    x=df_france['Year'],
    y=df_france['Annual emissions'],
    mode='markers',
    marker=dict(
        color='gold'  
    )
    ))

    # This is for changing the title of the figure and the axis labels
    
    figf.update_layout(
        title='Annual CO2 emissions in Francia',
        xaxis=dict(title='Year'),
        yaxis=dict(title='Annual C02 emissions')
    )
    
    df_spain = df_good.loc[(df_good['Country'] == 'Spain')]
    df_germany = df_good.loc[(df_good['Country'] == 'Germany')]
    df_usa = df_good.loc[(df_good['Country'] == 'United States')]
    df_india = df_good.loc[(df_good['Country'] == 'India')]
    df_rusia = df_good.loc[(df_good['Country'] == 'Russia')]
    df_arab = df_good.loc[(df_good['Country'] == 'United Arab Emirates')]
    df_eng = df_good.loc[(df_good['Country'] == 'United Kingdom')]
    df_france = df_good.loc[(df_good['Country'] == 'France')]
    
    '''
    We create more dataset for every country to display in the next plots
    '''
    
    #The next commands are not necessary
    

    # Showing the graph
    st.write(figf)
    st.markdown('''
                The emissions are in billions of tons. Now let's compare France with
                other countries.
                ''')
    
    
    
    
    df_EU = df_good.loc[(df_good['Country'] == 'France') | (df_good['Country'] == 'Germany') | (df_good['Country'] == 'Spain') | (df_good['Country'] == 'United Kingdom') | (df_good['Country'] == 'Italy')]
    
    #the line above is for selecting several country and use them in one dataset
    
    # scatterplot for the df_EU
    fig = px.scatter(df_EU, x='Year', y='Annual emissions', color='Country')

    # Commnads for changing the title and the axis
    fig.update_layout(
        title='Annual CO2 emissions for european countries',
        xaxis_title='Year',
        yaxis_title='Annual CO2 emissions'
    )

    # legend
    fig.update_layout(legend=dict(title='Country'))
    fig.update_layout(width=800, height=500)


    #showing the graph
    st.plotly_chart(fig)
    
    st.markdown('''
                
                We can see that over the decades, countries have been emitting less pollution. This is largely due to government 
                regulations that are committed to environmental care. Additionally, it is worth noting that technology has improved in 
                recent years, making combustion engines more efficient and environmentally friendly.

                We can observe that in 2020, CO2 emissions reached the lowest values in history, which should not come as a surprise as it was the year 
                when lockdown measures were implemented due to the COVID-19 pandemic.
                ''')
    
    st.text('Example')
    

    # This commnad is used for the palette of colors for the countries
    colors = px.colors.qualitative.Plotly

    # list of countries to plot
    traces = []
    countries = ['France', 'Spain', 'Germany', 'Italy','United Kingdom']  #Include any country

    for i, country in enumerate(countries):
        df_country = df[df['Country'] == country]
        trace = go.Box(
            x=df_country['Country'],
            y=df_country['Annual emissions'],
            name=country,
            marker=dict(color=colors[i % len(colors)])  # this assigns the color for each country
        )
        traces.append(trace)

    # Creating the figure and changing the title
    fig = go.Figure(data=traces)
    fig.update_layout(
        title='Distribution of CO2 emissions by country',
        xaxis_title='Country',
        yaxis_title='Annual CO2 emissions'
    )

    # the graph
    st.plotly_chart(fig)
    
    st.markdown('''
                As we can see, some countries have emitted more than others. Considering the fact that involves the 
                second industrial revolution and the countries involved in the WW2, it is quite normal to have these values
                ''')
    
    st.markdown('''
                Now let's show the classification based on the emissions according to the
                to the regularization in Europe.
                ''')
    
    
    data = pd.read_csv('tabs/data_2019.csv')
    data = data.sample(frac=0.1)
    #Let's clean it as we did it before
    NaN_percentage = (data.isnull().sum()/data.isnull().count()*100).sort_values(ascending = False)

    drop_list = sorted(list(NaN_percentage[NaN_percentage > 70].index))
    data.drop(labels=drop_list, axis=1, inplace=True)
    #Replacing Nans with Median in Numerical features
    data.fillna(data.median(numeric_only=None), inplace=True)
    #Dropping the rest of all missing values since they are in categorical features. We do not want to 
    # use the mode since NaN could be the most occuring variable.
    data.dropna(inplace=True)
    
    European_Classification = pd.cut(x = data['Ewltp (g/km)'],
                 bins = [100, 120, 140, 160, 200, 250, 500,700],
                 labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G'])
    clasf = pd.crosstab(data['Mh'], European_Classification)
    st.write(clasf)
    st.write('---')
    
    st.markdown('''The representation of the classification is shown below. 
                The feature 'Mh' stands for the brand of the car.
                ''')
    fig = go.Figure()

    for classification in clasf.columns:
        fig.add_trace(go.Bar(x=clasf.index, y=clasf[classification], name=classification))

    fig.update_layout(
        title='Classification of CO2 emissions',
        xaxis=dict(title='Mh'),
        yaxis=dict(title='Proportion'),
        barmode='stack',
        legend_title='European Classification'
    )
    fig.update_layout(width=800, height=600)

    st.write(fig)
    st.markdown(''' The label 'A' corresponds to the the lowest value of emission. If a car is classified with
                that label, the emissions (g/km) are less than 100. The label 'F' corresponds 
                to 500 (g/km) emissions.''')


    '''
    # Create a list for countries
    traces = []
    countries = ['France', 'Spain', 'Germany','United Kingdom']  # Agrega aquí los países que deseas incluir

    for country in countries:
        df_country = df[df['Country'] == country]
        trace = go.Box(
            x=df_country['Country'],
            y=df_country['Annual emissions'],
            name=country,
            marker=dict(color='Country')  # Asigna un color distinto para cada país
        )
        traces.append(trace)

    # Create figure and customize the name
    fig = go.Figure(data=traces)
    fig.update_layout(
        title='Distribución de emisiones de CO2 por país',
        xaxis_title='País',
        yaxis_title='Emisiones CO2 anuales'
    )
    fig.update_layout(width=800, height=500)

    # Mostrar el gráfico
    st.plotly_chart(fig)
    '''





