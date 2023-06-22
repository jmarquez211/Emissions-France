import streamlit as st 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

import plotly.express as px
import plotly.graph_objects as go




'''
This file is created for the visualization
'''

sidebar_name = "Data visualization"


def run():
    
    st.markdown("""First, let's take a look to the dataset
                and then, we can explore the features
                
                """)
    st.subheader('CO2 emissions dataset')
    
    
    df = pd.read_csv('tabs/data2019_r1.csv')
    df = df.sample(frac=0.04)
    st.write(df.head())
    
    st.markdown('''
                Here are the features. The feature 'Ewltp (g/km) is the emissions
                of C02, this is the target. The rest of the features are characteristics of cars and
                brand of cars.
                
                We can see the distribution of the emissions so we can observe what is the
                average.
                ''')
    
    st.subheader('Distribution of emissions')
    distribution_co2 = pd.DataFrame(df['Ewltp (g/km)'].value_counts()).head(50)
    st.bar_chart(distribution_co2)
    
    
    col1, col2 = st.columns(2)
    
    #max_depth = col1.slider('hey',min_value=10, max_value=100,value=20, step=10)
    st.markdown('''
                Value count for cars produced with respect to Fuel Type.
                Petrol and diesel produce the more. Now let's take a look of the kinf of fuel
                used in this dataset.''')
    
    
    fig_count = px.bar(df,x='Ft',color='Ft',title='Number of cars with its kind of fuel type')
    st.write(fig_count)
    
    st.markdown('''
                It is not a surprised most of people use cars with organic 
                combustible. There are few of them that are electric of hybrid,
                nevertheless, that kind were not considered on the models studied.
                
                The next graph shows us the value count for cars produced with respect to country of origin. It is 
                interesting to see France and Germany are the highest producers.
                ''')
    
   
    
    fig_count_country = px.bar(df,x='Country',color='Country',
                               title='Value counts by country')
    st.write(fig_count_country)
    
    '''
    Little modifications for plotting
    '''
    df['Mk'] = df['Mk'].str.upper()
    df['Mh'] = df['Mh'].replace(['BMW AG', 'BMW GMBH'],'BMW')
    df['Mh'] = df['Mh'].replace(['PSA', 'AUTOMOBILES PEUGEOT'],'AUTOMOBILES PEUGEOT')
    df['Mh'] = df['Mh'].replace(['MAZDA EUROPE', 'MAZDA'],'MAZDA')
    df['Mh'] = df['Mh'].replace(['SUZUKI MOTOR CORPORATION', 'MAGYAR SUZUKI'],'SUZUKI MOTOR CORPORATION')
    df['Mh'] = df['Mh'].replace(['AUDI AG', 'AUDI HUNGARIA'],'AUDI')
    df['Mh'] = df['Mh'].replace(['FORD WERKE GMBH', 'FORD MOTOR COMPANY', 'CNG TECHNIK'],'FORD')
    df['Mh'] = df['Mh'].replace(['HYUNDAI CZECH', 'HYUNDAI ASSAN', 'HYUNDAI'],'HYUNDAI')
    df['Mh'] = df['Mh'].replace(['KIA SLOVAKIA', 'KIA'],'KIA')
    df['Mh'] = df['Mh'].replace(['MITSUBISHI MOTORS THAILAND', 'MITSUBISHI MOTORS CORPORATION'],'MITSUBISHI')
    df['Mh'] = df['Mh'].replace(['NISSAN', 'NISSAN AUTOMOTIVE EUROPE'],'NISSAN')
    
    df['Mk'] = df['Mk'].replace(['VOLKSWAGEN, VW','VOLKSWAGEN','VOLKSWAGEN  VW','VOLKSWAGEN VW',
                             'VOLKSWAGEN,VW','VOLKSWAGEN. VW','VW'],'VOLKSWAGEN')
    
    df['Mk'] = df['Mk'].replace(['MERCEDES-BENZ', 'MERCEDES BENZ', 'MERCEDES'],'MERCEDES-BENZ')
    df['Mk'] = df['Mk'].replace(['FORD', 'FORD-CNG-TECHNIK', 'FORD - CNG-TECHNIK','FORD-CNG TECHNIK'],'FORD')
    df['Mk'] = df['Mk'].replace(['HYUNDAI', 'HYUNDAI, GENESIS'],'HYUNDAI')
    df['Mk'] = df['Mk'].replace(['OPEL / VAUXHALL', 'OPEL'],'OPEL')
    df['Mk'] = df['Mk'].replace(['CITROEN / DS', 'DS AUTOMOBILES', 'DS'],'DS')
    df['Mk'] = df['Mk'].replace(['SOCIETE DES AUTOMOBILES ALPINE(SAA)', 'ALPINE'],'ALPINE')
    
    
    df['Make_Type'] = df['Mk'].replace(['BMW', 'AUDI', 'MERCEDES-BENZ', 'PORSCHE', 
                                           'CUPRA', 'JEEP', 'LAND ROVER','LEXUS', 'JAGUAR','Audi','Mercedes-Benz','Jaguar','Land Rover'],'Luxury')
    
    df['Make_Type'] = df['Make_Type'].replace(['VOLKSWAGEN', 'PEUGEOT', 'RENAULT', 'SKODA', 'CITROEN', 
                                           'OPEL', 'DACIA', 'TOYOTA','FORD', 'SEAT', 'MINI', 'KIA',
                                          'HYUNDAI', 'FIAT', 'SUZUKI', 'NISSAN', 'MAZDA', 'MITSUBISHI',
                                          'DS', 'HONDA', 'ALFA ROMEO', 'ŠKODA', 'SUBARU', 'ALPINE',
                                          'API CZ', 'KEAT','Skoda','Mazda','Opel/ Vauxhall','Honda','Seat'],'General')
    
    df.drop(['Mk'],inplace=True, axis=1)
    
    df['Make_Type'].unique()
    df['Make_Type'] = df['Make_Type'].astype(str)
    
    

    
    
    
    st.write('---')
    st.markdown(
        """
        From here, we are going to display some graphs so the user can see interesting comparisons.
        We can deduce what features are the highest impact on the feature target. 
    
        """
    )
    
    
   
    st.markdown(
        """
        Relation between CO2 emissions and engine power
        
        """
    )
    
    
    
    fig_scatter = px.scatter(df1,x=df1['ep (KW)'],y=df1['Ewltp (g/km)'],
                             color=df1['Ewltp (g/km)'])
    st.write(fig_scatter)
    
    st.write('---')
    st.markdown(
        """
        Relation between CO2 emissions and mass by running order.
        Cars with high mass by running order have high CO2 emission
        """
    )
    
    
    #Relation between CO2 Emmission and Mass by running order
    
 
    fig_scatter0 = px.scatter(df,x=df['Mt'],y=df['Ewltp (g/km)'],
                              color=df['Mt'],
                              title='Distribution of mass of cars')
    st.write(fig_scatter0)
    
    
    
    
    st.markdown(
        """
        Relation between CO2 emissions and wheel base.
        Vehicles with big wheel base also produce high CO2 emission
        """
    )
    
    
    
    st.write('---')
    
    fig_sc = px.scatter(df,x=df['W (mm)'],y=df['Ewltp (g/km)'],
                        color=df['W (mm)'],title='Emissions by wheel base')
    st.write(fig_sc)
    
    st.write('---')
    st.markdown(
        """
        Relation between CO2 emissions and wheel base.
        Vehicles with big wheel base also produce high CO2 emission
        """
    )
    
 
    

    
    #df['Mk'] = df['Mk'].astype(str)

    
    
    st.write('---')
    
    figbox = px.box(df,x=df['Make_Type'],y='Ewltp (g/km)',
                    color=df['Make_Type'],title='Distribution of emissions according to the brand')
    
    
    st.write(figbox)
    
    st.write('---')
    
    st.subheader('Is France emitting too much pollution?')
    st.markdown(''' 
                To answer this question we are going to compare it with other countries.
                To do so, we have to inspect the data set of emission for evert country.
                The column 'Annual emissions' is the total of emissions in that year, represented
                in billions of tons.
                ''')
    
    df = pd.read_csv('tabs/annual-share-of-co2-emissions.csv')
    
    df.rename({'Entity':'Country'},axis=1, inplace=True)
    
    #df['Annual CO₂ emissions (zero filled)'] = df['Annual CO₂ emissions (zero filled)'].apply(lambda x: '{:.0f}'.format(x))
    df['Year'] = df['Year'].apply(lambda x: '{:.0f}'.format(x))
    
    '''
    The line above is to change the format of the year, because streamlit uses scientific notation by default
    '''
    
    df.dropna(inplace=True)
    #st.write(df.head())
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
    
    st.markdown('''Let's focus on the values for France. We reduce the data set
                to the years more relevant, since the decade of 60's. This reason to do
                this is because other periods of history are influenced by great events like wars
                and development of techonology. So, the data to study is shown below:
                ''')
    st.write(df_france)
    
    st.markdown(' ')
    
    
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
    df_china = df_good.loc[(df_good['Country'] == 'China')]
    
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
    
    
    

    '''
    This graph isn't necessary indeed
    
    
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
    fig_EU = go.Figure(data=traces)
    fig_EU.update_layout(
        title='Distribution of CO2 emissions by country in UE',
        xaxis_title='Country',
        yaxis_title='Annual CO2 emissions'
    )
    
    # the graph
    st.plotly_chart(fig_EU)
    '''
    
    st.markdown(''' 
                
                Now we compare France with the most powerful and richest countries:
                
                '''
                )
    
    
    
    colors = px.colors.qualitative.Plotly
    
    # list of countries to plot
    traces_r = []
    countries_r = ['France', 'United States of America', 'India', 'Russia','United Arab Emirates','China'] 
    
    for i, country in enumerate(countries_r):
        df_country = df[df['Country'] == country]
        trace = go.Box(
            x=df_country['Country'],
            y=df_country['Annual emissions'],
            name=country,
            marker=dict(color=colors[i % len(colors)])  # this assigns the color for each country
        )
        traces_r.append(trace)
    
    fig_r = go.Figure()

    #We could make another dataset with the richest countries

    # Agregar la primera traza (Francia)
    fig_r.add_trace(go.Scatter(
        x=df_india['Year'],
        y=df_india['Annual emissions'],
        mode='markers',
        name='India'
    ))

    # Agregar la segunda traza (España)
    fig_r.add_trace(go.Scatter(
        x=df_rusia['Year'],
        y=df_rusia['Annual emissions'],
        mode='markers',
        name='Russia'
    ))

    # Agregar la tercera traza (Alemania)

    fig_r.add_trace(go.Scatter(
        x=df_arab['Year'],
        y=df_arab['Annual emissions'],
        mode='markers',
        name='United Arab Emirates'
    )) 

    fig_r.add_trace(go.Scatter(
        x=df_usa['Year'],
        y=df_usa['Annual emissions'],
        mode='markers',
        name='USA'
    ))
    
    fig_r.add_trace(go.Scatter(
    x=df_china['Year'],
    y=df_china['Annual emissions'],
    mode='markers',
    name='China'
)) 

    fig_r.add_trace(go.Scatter(
        x=df_france['Year'],
        y=df_france['Annual emissions'],
        mode='markers',
        name='France'
    )) 


    # Establecer título y nombres de los ejes
    fig_r.update_layout(
        title='Annual emissions, countries with the highest emissions',
        xaxis=dict(title='Year'),
        yaxis=dict(title='Annual CO2 emissions')
    )
    fig_r.update_layout(width=800, height=600)
    
    st.plotly_chart(fig_r)

    
    st.write('---')
    
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
    fig_cls = go.Figure()
    
    for classification in clasf.columns:
        fig_cls.add_trace(go.Bar(x=clasf.index, y=clasf[classification], name=classification))

    fig_cls.update_layout(
        title='Classification of CO2 emissions',
        xaxis=dict(title='Mh'),
        yaxis=dict(title='Proportion'),
        barmode='stack',
        legend_title='European Classification'
    )
    fig_cls.update_layout(width=800, height=600)

    st.write(fig_cls)
    st.markdown(''' The label 'A' corresponds to the the lowest value of emission. If a car is classified with
                that label, the emissions (g/km) are less than 100. The label 'F' corresponds 
                to 500 (g/km) emissions.''')
    
    
    '''
    df['Mk1'] = df['Mk'].replace(['BMW', 'AUDI', 'MERCEDES-BENZ', 'PORSCHE', 
                                           'CUPRA', 'JEEP', 'LAND ROVER','LEXUS', 'JAGUAR'],'Luxury')
    
    df['Mk1'] = df['Mk1'].replace(['VOLKSWAGEN', 'PEUGEOT', 'RENAULT', 'SKODA', 'CITROEN', 
                                           'OPEL', 'DACIA', 'TOYOTA','FORD', 'SEAT', 'MINI', 'KIA',
                                          'HYUNDAI', 'FIAT', 'SUZUKI', 'NISSAN', 'MAZDA', 'MITSUBISHI',
                                          'DS', 'HONDA', 'ALFA ROMEO', 'ŠKODA', 'SUBARU', 'ALPINE',
                                          'API CZ', 'KEAT'],'General')
   '''
