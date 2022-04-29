import plotly.express as px
import seaborn as sns
from matplotlib import pyplot as pltpwd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
 

# loading Dataset 
df1 = pd.read_csv('usedcars_dataset.csv')
st.title("Analysis of used cars")

list = ['Introduction','Relationship between feature and price', 'Normalized-Losses and Make', 'Risk Level', 'Speed and the weight of the car', 'Intercorrelation Heatmap']

pageview = st.sidebar.radio('Select the page you want to view', list)

from PIL import Image
img1 = Image.open("used cars.jpg")



if pageview == 'Introduction':
    
    st.write("Used cars market is increasing day by day and a consumer always wants the best resale price of his car. Using our dataset of used car, we intended to clarify some of these unknown variables and provide a convenient environment for those searching for a car.")
    
    
    st.image(img1)

    

if pageview == 'Relationship between feature and price':
    st.subheader("Co-relationship between price and other features") 
    
    fp = ['horsepower','highway-mpg', 'symboling', 'engine-size']
    feature1 = st.selectbox("Select feature 1",(fp))
    scatterplot = alt.Chart(df1).mark_circle(size=60).encode(
    x=alt.X(feature1),
    y=alt.Y('price'),
    tooltip=[feature1, 'price'],
        #color='blue'
    ).interactive()
    
    st.write(scatterplot)
 

    
if pageview == 'Normalized-Losses and Make':
    
    df2 =df1[df1['normalized-losses'].notna()]
    df3 = df2.groupby(["make"]).mean()
    df3=df3.sort_values(by=['normalized-losses'],ascending=False)
    df3 = df3.reset_index()
    
    st.subheader("Normalized-Losses and Make") 
    radioButton = st.radio(
     "What do you want to show?",
     ('Introduction', 'Visualization'))
    if radioButton == 'Introduction':
        st.write("Normalized-losses is the relative average loss payment per insured vehicle year. This value is normalized for all autos within a particular size classification, and represents the average loss per car per year. ")
    if radioButton == 'Visualization':
        fig1 = px.bar(df3, x='make', y='normalized-losses', color='make', title="Bar chart of normalized losses and make")
        st.plotly_chart(fig1)
        

        fig2 = px.scatter(df1, 
                 x="normalized-losses", 
                 y="make",
                 size="price", 
                 color="make",
                 hover_name="normalized-losses", 
                 size_max=60)
        st.plotly_chart(fig2)

        
        box_fig = px.box(df2, x='make', y='normalized-losses', title="Box plot of normalized losses and make")
        st.write(box_fig)

        
        
if pageview == 'Risk Level':
    st.subheader("Insurance Risk Level")
    st.markdown("**Utilizing these charts, we could find the _low risk engine-type/ make/ body-style_**")
    f1 = st.radio("Select one among these:", ['engine-type', 'make','body-style'])
    f2 = st.selectbox("Select a feature on which you want to evaluate the risk level:", ['symboling', 'price'])
    
   
    
    if f1 == 'make':
        dfe = df1.groupby([f1]).mean()
        dfe=dfe.sort_values(by=[f2],ascending=False)
        dfe = dfe.reset_index()
        fige = px.bar(dfe, x=f1, y=f2, color=f1, title="Bar chart of "+f1+" and "+f2)
        st.plotly_chart(fige)
    
    if f1 == 'engine-type':
        dfe = df1.groupby([f1]).mean()
        dfe=dfe.sort_values(by=[f2],ascending=False)
        dfe = dfe.reset_index()
        fige = px.bar(dfe, x=f1, y=f2, color=f1, title="Bar chart of "+f1+" and "+f2)
        st.plotly_chart(fige)
     
    if f1 == 'body-style':
        dfe = df1.groupby([f1]).mean()
        dfe=dfe.sort_values(by=[f2],ascending=False)
        dfe = dfe.reset_index()
        fige = px.bar(dfe, x=f1, y=f2, color=f1,title="Bar chart of "+f1+" and "+f2)
        st.plotly_chart(fige)
        
        

# Heatmap Visualization 
if pageview == 'City gas mileage prediction':
    st.subheader("city gas mileage prediction")
    st.markdown("**_Under_ Progress**")
    corr = df.corr()
    sns.set_context("notebook", font_scale=1.0, rc={"lines.linewidth": 2.5})
    plt.figure(figsize=(13,7))
    a = sns.heatmap(corr, annot=True, fmt='.2f')
    rotx = a.set_xticklabels(a.get_xticklabels(), rotation=90)
    roty = a.set_yticklabels(a.get_yticklabels(), rotation=30)
    st.write(a)

if pageview =='Intercorrelation Heatmap':
    st.set_option('deprecation.showPyplotGlobalUse', False)
    st.header('Intercorrelation Matrix Heatmap')
    corr = df1.corr()
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    with sns.axes_style("white"):
        f,ax = plt.subplots(figsize=(7, 5))
        ax = sns.heatmap(corr, mask=mask, vmax=1, square=True)
    st.pyplot()
    
    
if pageview == 'Speed and the weight of the car':
    
    radioButton = st.radio("What do you want to show?",('Higest mpg', 'Lowest mpg'))
    st.set_option('deprecation.showPyplotGlobalUse', False)
    
    if radioButton == 'Higest mpg':
         g = sns.lmplot('highway-mpg',"curb-weight", df1, hue="make",fit_reg=False);
         
         st.pyplot()
    if radioButton == 'Lowest mpg':
         g = sns.lmplot('city-mpg',"curb-weight", df1, hue="make",fit_reg=False);
         st.pyplot()
   
    
    

