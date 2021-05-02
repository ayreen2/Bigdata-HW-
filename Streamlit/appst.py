import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import plotly.express as px
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import math

st.title('Gapminder')

#@st.cache
def load_data():
    data = pd.read_csv("gapminder_data.csv")
    return data.assign(Year = lambda d: pd.to_datetime(d['Year'])) 
df = load_data()

def gdp_life(df):
    gdp_year = st.sidebar.slider("Select Year ",min_value=1960, max_value=2019, step=1, key = 'gdp')
    gdp_Region = st.sidebar.selectbox("Select Region",
       df.Region.unique(),key = 'gdp'
    )

    # Filtering the dataframe.
    gy_df = df[(df['Year'] == gdp_year) & (df['Region'] == gdp_Region)] 
    gdp_le = px.scatter(gy_df, x="GDP", y="Life", text="Country", log_x=True, size='Population')
    gdp_le.update_traces(textposition='top center')
    

    st.title(f'GDP per capita and Life Expectancy in {gdp_Region} in {gdp_year}')
    st.plotly_chart(gdp_le)



if __name__ == "__main__":
    
    @st.cache
    def load_data():
        df = pd.read_csv('gapminder_data.csv')
        return df

    df = load_data()
    
    st.sidebar.title(f"Selection")
    
    gdp_life(df)

