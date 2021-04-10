#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
pd.set_option('display.max_columns', None)
get_ipython().run_line_magic('matplotlib', 'inline')
import json
import plotly.express as px
import plotly.graph_objects as go
import cufflinks as cf
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
init_notebook_mode(connected=True)
cf.go_offline()


# In[2]:


import os 
import folium
from folium import plugins
import rasterio as rio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import earthpy as et
import webbrowser


# In[3]:


df = pd.read_csv("Chicago_Crimes_2012_to_2017.csv")


# In[4]:


df.head()


# In[5]:


df.info()


# In[6]:


df.columns


# In[7]:


len(df['Unnamed: 0'].unique())


# In[8]:


df.drop(['Unnamed: 0', 'IUCR', 'Block', 'Beat', 'Ward', 'Community Area', 'FBI Code', 
        'X Coordinate', 'Y Coordinate', 'Updated On'], axis=1, inplace=True)


# In[9]:


df[df.duplicated()]


# In[10]:


df['Primary Type'].unique()


# In[11]:


df.drop_duplicates(inplace=True)


# Seeking for duplicates, in for example 'Case Number', we see that there are some cases, but I believe it is possible in cases such as charges for more than one crime commited by one person - maybe 'Case number' is assigned to a criminal. The 'Primary type' being different within each repeating 'Case Number' may lead to that conclusion.

# In[12]:


df[df['Case Number'].duplicated(keep=False)]


# In[13]:


df[df.isnull()['District']==True] #seeking for null value in District column


# In[14]:


df.dropna(axis = 0, subset = ['District'], inplace = True) # subset defines in which columns to look for missing values
df[df.isnull()['District']==True]


# In[15]:


df['District'] = df['District'].map(lambda i: str(int(i))) # swapping floats in District column to strings


# # Yearly stats

# ### Overall number of crimes for each year
# As we can see, the numbers have slightly decreased over the years with exception of 2015-2016 where there was a minimal growth.
# Data for 2017 is incomplete, therefore there are significantly less crimes that year.

# In[16]:


df['Year'].value_counts().to_frame().reset_index().rename(columns = {'index':'Year','Year':'Number of Crimes'})


# In[17]:


fig = px.histogram(df,
                    x = 'Year',
                    title = "Number of crimes throught years", color='Arrest') # creates countplot for 'Year' column

fig.update_layout(
    bargap=0.2) ## sets gap size between bins

fig.show()


# It turns out that significantly more crimes that are reported do not result in arrests. We will dive into that deeper later.

# In[87]:


C = df.groupby(['Primary Type', 'Arrest', 'Year']).count()['ID'].to_frame().reset_index().rename(columns = {'ID':'Number of crimes'})


# In[19]:


C[C['Primary Type'] == 'ARSON']


# In[88]:


fig = px.sunburst(C, path=['Year', 'Arrest'], values='Number of crimes', title = 'Number of crimes and arrests ratio within each year')
fig.show()


# In[21]:


from datetime import datetime
import calendar
df['Month'] = [calendar.month_name[datetime.strptime(i[:-3],'%m/%d/%Y %H:%M:%S').month] for i in df['Date']]


# In[89]:


D = df.groupby(['Year', 'Month', 'Primary Type']).count()['ID'].to_frame().reset_index().rename(columns = {'ID':'Number of crimes'})
D[D['Primary Type'] == 'ARSON'].head()


# Plot of the number of crimes as time series shows two things. Firstly, numbers are decreasing and secondly, there seems to be some seasonality involved. 

# In[23]:


by_month = pd.to_datetime(df['Date']).dt.to_period('M').value_counts().sort_index()
by_month.index = pd.PeriodIndex(by_month.index)


# In[24]:


df_month = by_month.rename_axis('month').reset_index(name='counts')
df_month['month'] = df_month['month'].astype(dtype=str) # necessary because px.line does not let y/m format


# In[25]:


fig = px.line(df_month, x='month', y="counts", title = "Monthly number of crimes throughout the years")
fig.show()


# Similarly to overall number of crimes, thefts seem to occur more often seasonally. The peaks are usually around July / August and lows in February. Trend is also decreasing.

# In[26]:


theft_month = pd.to_datetime(df[df['Primary Type'] == 'THEFT']['Date']).dt.to_period('M').value_counts().sort_index()
theft_month.index = pd.PeriodIndex(theft_month.index)


# In[27]:


df_month_theft = theft_month.rename_axis('month').reset_index(name='counts')
df_month_theft['month'] = df_month_theft['month'].astype(dtype=str)


# In[28]:


fig = px.line(df_month_theft, x='month', y="counts", title = "Monthly number of thefts throughout the years")
fig.show()


# When it comes to narcotics, at first glance there seems not to be any seasonality, but number of cases is decreasing on definitely faster rate than previous crimes.

# In[29]:


narcotics_month = pd.to_datetime(df[df['Primary Type'] == 'NARCOTICS']['Date']).dt.to_period('M').value_counts().sort_index()
narcotics_month.index = pd.PeriodIndex(narcotics_month.index)


# In[30]:


df_month_narcotics = narcotics_month.rename_axis('month').reset_index(name='counts')
df_month_narcotics['month'] = df_month_narcotics['month'].astype(dtype=str)


# In[31]:


fig = px.line(df_month_narcotics, x='month', y="counts", title = "Monthly number of crimes related to narcotics throughout the years")
fig.show()


# Assaults seem to be happening more often around May. What's interesting, the trend does not seem to be decreasing - it's kind of periodic.

# In[32]:


assaults_month = pd.to_datetime(df[df['Primary Type'] == 'ASSAULT']['Date']).dt.to_period('M').value_counts().sort_index()
assaults_month.index = pd.PeriodIndex(assaults_month.index)


# In[33]:


df_month_assaults = assaults_month.rename_axis('month').reset_index(name='counts')
df_month_assaults['month'] = df_month_assaults['month'].astype(dtype=str)


# In[34]:


fig = px.line(df_month_assaults, x='month', y="counts", title = "Monthly number of assaults throughout the years")
fig.show()


# # Overall stats

# In[92]:


list(df['Primary Type'].value_counts().iloc[:10].index)


# In[36]:


fig = px.bar(df['Primary Type'].value_counts().iloc[:20].to_frame().reset_index().rename(columns = {'index':'Primary Type', 'Primary Type':'Number of Crimes'}),
                x = 'Primary Type', y = 'Number of Crimes',
                title="Number of 20 most common crimes during 2012-2017 period")
fig.show()


# In[38]:


fig = px.histogram(df[df['Primary Type'].isin(list(df['Primary Type'].value_counts().iloc[:10].index))],
                x = 'Primary Type', color='Year',
                title="Top 10 crimes", barmode='group',
                category_orders = {"Year":[2012,2013,2014,2015,2016,2017]}).update_xaxes(categoryorder="total descending")
fig.show() 
# category_orders is for order of the color - without it, it was random
# update_xaxes is for order of the x axis - without it it was random


# ### Number of 20 most common crime venues during 2012-2017 period and top 10 in each individual year

# In[39]:


df['Location Description'].value_counts().iloc[:20].index


# In[40]:


fig = px.bar(df['Location Description'].value_counts().iloc[:20].to_frame().reset_index().rename(columns = {'index':'Location Description', 'Location Description':'Number of Crimes'}),
                x = 'Location Description', y = 'Number of Crimes',
                title="Number of 20 most common locations of crimes during 2012-2017 period")
fig.show()


# In[41]:


fig = px.histogram(df[df['Location Description'].isin(list(df['Location Description'].value_counts().iloc[:10].index))],
                x = 'Location Description', color='Year',
                title="Top 10 Locations", barmode='group',
                category_orders = {"Year":[2012,2013,2014,2015,2016,2017]}).update_xaxes(categoryorder="total descending")
fig.show()


# In[42]:


fig = px.histogram(df[df['Location Description'].isin(list(df['Location Description'].value_counts().iloc[:10].index))],
                x = 'Location Description', color='Primary Type',
                title="Number of 10 most common locations of crimes during 2012-2017 period with most common crimes",
                category_orders = {"Primary Type":list(df['Primary Type'].value_counts().to_frame().reset_index()['index'])}).update_xaxes(categoryorder="total descending")

fig.update_layout(
    autosize=False,
    width=1100,
    height=600)

fig.show() # too many 'colors'


# ### 10 most common crimes in 5 top locations 

# In[43]:


B = df[df['Location Description']=='STREET']['Primary Type'].value_counts().to_frame().reset_index()
# creating new data frame to have real percentages and less relevant crimes as 'REST'
B.loc[32] = ['REST', sum(B['Primary Type'][24:])] # appends new row to the end of the data frame
C = pd.concat([B.loc[:24, :], B.loc[32:33, :]], axis=0).rename(columns = {'index':'Primary Type', 'Primary Type':'Number of Crimes'})
fig = px.pie(C, values='Number of Crimes', names='Primary Type', title='Crimes commited on the streets of Chicago')
fig.show()


# In[44]:


B = df[df['Location Description']=='RESIDENCE']['Primary Type'].value_counts().to_frame().reset_index()
B.loc[32] = ['REST', sum(B['Primary Type'][24:])]
C = pd.concat([B.loc[:24, :], B.loc[32:33, :]], axis=0).rename(columns = {'index':'Primary Type', 'Primary Type':'Number of Crimes'})
fig = px.pie(C, values='Number of Crimes', names='Primary Type', title='Crimes commited in residences of citizens of Chicago')
fig.show()


# In[45]:


B = df[df['Location Description']=='APARTMENT']['Primary Type'].value_counts().to_frame().reset_index()
B.loc[32] = ['REST', sum(B['Primary Type'][19:])]
C = pd.concat([B.loc[:19, :], B.loc[32:33, :]], axis=0).rename(columns = {'index':'Primary Type', 'Primary Type':'Number of Crimes'})
fig = px.pie(C, values='Number of Crimes', names='Primary Type', title='Crimes commited in apartments of citizens of Chicago')
fig.update_traces(textposition='inside', textinfo='percent+label') #other representation of the piechart; it looks better in that case
fig.show()


# In[46]:


B = df[df['Location Description']=='SIDEWALK']['Primary Type'].value_counts().to_frame().reset_index()
B.loc[32] = ['REST', sum(B['Primary Type'][24:])]
C = pd.concat([B.loc[:24, :], B.loc[32:33, :]], axis=0).rename(columns = {'index':'Primary Type', 'Primary Type':'Number of Crimes'})
fig = px.pie(C, values='Number of Crimes', names='Primary Type', title='Crimes commited on sidewalks of Chicago')
fig.update_traces(textposition='inside', textinfo='percent+label') #other representation of the piechart; it looks better in that case
fig.show()


# In[47]:


B = df[df['Location Description']=='OTHER']['Primary Type'].value_counts().to_frame().reset_index()
B.loc[32] = ['REST', sum(B['Primary Type'][24:])]
C = pd.concat([B.loc[:24, :], B.loc[32:33, :]], axis=0).rename(columns = {'index':'Primary Type', 'Primary Type':'Number of Crimes'})
fig = px.pie(C, values='Number of Crimes', names='Primary Type', 
            title='Crimes commited in other locations of Chicago')
fig.update_traces(textposition='inside', textinfo='percent+label', pull=[0.1]) # 'pull' attribute pulls traces from the chart
fig.show()


# In[48]:


df.head()


# ### Crimes vs time of the day

# In[49]:


df[df.isnull()['Date']==True]


# In[50]:


def f(i):
    if (i[-2:]=='AM'):
        if (int(int(i[-11:-9])) == 12):
            return 'Midnight'
        elif (int(int(i[-11:-9])) in [1, 2, 3]):
            return 'Night'
        elif (int(int(i[-11:-9])) in [4, 5]):
            return 'Early Morning'
        elif (int(int(i[-11:-9])) in [6, 7, 8]):
            return 'Morning'
        elif (int(int(i[-11:-9])) in [9, 10]):
            return 'Late Morning'
        elif (int(int(i[-11:-9])) == 11):
            return 'Noon'
    if (i[-2:]=='PM'):
        if (int(int(i[-11:-9])) == 12):
            return 'Noon'
        elif (int(int(i[-11:-9])) in [1, 2, 3, 4, 5]):
            return 'Afternoon'
        elif (int(int(i[-11:-9])) in [6, 7, 8]):
            return 'Evening'
        elif (int(int(i[-11:-9])) in [9, 10, 11]):
            return 'Late Evening'


# In[51]:


df['Time of Day'] = [f(i) for i in df['Date']]


# In[52]:


B = df['Time of Day'].value_counts().to_frame().reset_index().rename(columns = {'index':'Time of Day', 'Time of Day':'Number of Crimes'})
fig = px.pie(B, values='Number of Crimes', names='Time of Day', 
            title='Number of crimes commited within the day')
#fig.update_traces(textposition='inside', textinfo='percent+label', pull=[0.1]) # 'pull' attribute pulls traces from the chart
fig.show()


# In[53]:


B = df[df['Primary Type']=='THEFT']['Time of Day'].value_counts().to_frame().reset_index().rename(columns = {'index':'Time of Day', 'Time of Day':'Number of Crimes'})
fig = px.pie(B, values='Number of Crimes', names='Time of Day', 
            title='Number of thefts commited in each part of the day')
fig.update_traces(textposition='inside', textinfo='percent+label', pull=[0.1]) # 'pull' attribute pulls traces from the chart
fig.show()


# In[54]:


B = df[df['Primary Type']=='CRIM SEXUAL ASSAULT']['Time of Day'].value_counts().to_frame().reset_index().rename(columns = {'index':'Time of Day', 'Time of Day':'Number of Crimes'})
fig = px.pie(B, values='Number of Crimes', names='Time of Day', 
            title='Number of sexual assaults commited in each part of the day')
fig.update_traces(textposition='inside', textinfo='percent+label', pull=[0.1]) # 'pull' attribute pulls traces from the chart
fig.show()


# In[55]:


B = df[df['Time of Day']=='Night']['Primary Type'].value_counts().to_frame().reset_index().rename(columns = {'index':'Primary Type', 'Primary Type':'Number of Crimes'})
fig = px.pie(B, values='Number of Crimes', names='Primary Type', 
            title='Crimes commited at night')
fig.update_traces(textposition='inside', textinfo='percent+label', pull=[0.1]) # 'pull' attribute pulls traces from the chart
fig.show()


# ### Top crime locations for top crimes

# In[60]:


B = df[df['Primary Type']=='BATTERY']['Location Description'].value_counts().to_frame().reset_index()
B.loc[32] = ['REST', sum(B['Location Description'][10:])]
pd.concat([B.loc[:10, :], B.loc[32:33, :]], axis=0).iplot(kind='pie', values='Location Description', labels = 'index', title='Battery in Chicago')


# In[61]:


B = df[df['Primary Type']=='CRIMINAL DAMAGE']['Location Description'].value_counts().to_frame().reset_index()
B.loc[32] = ['REST', sum(B['Location Description'][10:])]
pd.concat([B.loc[:10, :], B.loc[32:33, :]], axis=0).iplot(kind='pie', values='Location Description', labels = 'index', title='Criminal damage in Chicago')


# In[62]:


B = df[df['Primary Type']=='NARCOTICS']['Location Description'].value_counts().to_frame().reset_index()
B.loc[32] = ['REST', sum(B['Location Description'][10:])]
pd.concat([B.loc[:10, :], B.loc[32:33, :]], axis=0).iplot(kind='pie', values='Location Description', labels = 'index', title='Narcotics in Chicago')


# In[63]:


B = df[df['Primary Type']=='ASSAULT']['Location Description'].value_counts().to_frame().reset_index()
B.loc[32] = ['REST', sum(B['Location Description'][10:])]
pd.concat([B.loc[:10, :], B.loc[32:33, :]], axis=0).iplot(kind='pie', values='Location Description', labels = 'index', title='Assaults in Chicago')


# ### Arrests within crimes

# In[64]:


B = df[df['Primary Type']=='THEFT']['Arrest'].value_counts().to_frame().reset_index().iplot(kind='bar', x='index', y = 'Arrest', title='Theft arrests')


# In[65]:


B = df[df['Primary Type']=='BATTERY']['Arrest'].value_counts().to_frame().reset_index().iplot(kind='bar', x='index', y = 'Arrest', title='Battery arrests')


# In[66]:


B = df[df['Primary Type']=='NARCOTICS']['Arrest'].value_counts().to_frame().reset_index().iplot(kind='bar', x='index', y = 'Arrest', title='Narcotics arrests')


# In[67]:


B = df[['Primary Type','Arrest']]
B_1 = B.groupby(['Primary Type'])
B_2 = (B_1.count().unstack()-B_1.sum().unstack()).to_frame() # data frame with count of not arrested (count of all - count of arrested)
B_2 = B_2.droplevel(0) #index column is 'Arrest' so we need to drop it
merged = pd.concat([B_1.sum(), B_2.rename(columns={0: "Not arrested"})], axis=1, sort=False)
merged


# In[68]:


s = merged['Arrest']+merged['Not arrested']

fig = go.Figure(data=[
    go.Bar(name='Arrested', x=merged.index, y=merged['Arrest']/s),
    go.Bar(name='Not rrested', x=merged.index, y=merged['Not arrested']/s)
])
fig.update_layout(barmode='stack')
fig.show()


# # FOLIUM

# In[69]:


df.head()


# fig = px.scatter_mapbox(df[df['Year'] == 2016][df['Primary Type'] == 'THEFT'], lat="Latitude", lon="Longitude", hover_name="Location Description", hover_data=["Description"],
#                         color_discrete_sequence=["fuchsia"], zoom=3, height=500)
# fig.update_layout(
#     mapbox_style="white-bg",
#     mapbox_layers=[
#         {
#             "below": 'traces',
#             "sourcetype": "raster",
#             "sourceattribution": "United States Geological Survey",
#             "source": [
#                 "https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/MapServer/tile/{z}/{y}/{x}"
#             ]
#         }
#       ])
# fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
# fig.show()

# In[70]:


import geopandas
from urllib.request import urlopen
with urlopen('https://raw.githubusercontent.com/radosuaw/chicago/master/Boundaries_Police_Districts_1.json') as data:
    g = json.load(data)


# In[71]:


g


# g = geopandas.read_file("Boundaries_Police_Districts.geojson")

# In[73]:


# district number as float
# reset_index to get a column with districts, otherwise the district numbers will be indexes
temp = df[df['Primary Type']=='THEFT']['District'].value_counts().to_frame().reset_index()
# district number as string required in further code
temp['index'] = temp['index'].map(lambda i: str(int(i))) 
# changing names of the data frame columns
temp.rename(columns = {"index": 'District', "District": 'Number of thefts'}, inplace=True)


# In[74]:


fig = px.choropleth_mapbox(temp, geojson=g, color="Number of thefts", color_continuous_scale = "YlGnBu",
                           locations="District", featureidkey = "properties.dist_num",
                           center={"lat": 41.8781, "lon": -87.6298},
                           mapbox_style="carto-positron", zoom=9.5, opacity=0.7, height=800)
#fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()


# Above we can see that thefts in districts that cover central area near the lake are the most common.

# In[75]:


E = df[df['Primary Type']=='THEFT'].groupby(['Year','District']).count()['ID'].to_frame().reset_index().rename(columns = {'ID':'Number of thefts'})
E['District'] = E['District'].map(lambda i: str(int(i))) 
E.head()


# In[76]:


fig = px.choropleth_mapbox(E, geojson=g, color="Number of thefts", color_continuous_scale = "YlGnBu",
                           locations="District", featureidkey = "properties.dist_num", animation_frame='Year',
                           center={"lat": 41.8781, "lon": -87.6298}, mapbox_style="carto-positron", zoom=9.5, opacity=0.7, height=800)
fig.show()


# In each year thefts seem to be dominating in 1s, 18th and 19th district.

# In[77]:


temp = df[df['Primary Type']=='NARCOTICS']['District'].value_counts().to_frame().reset_index()
# district number as string required in further code
temp['index'] = temp['index'].map(lambda i: str(int(i))) 
# changing names of the data frame columns
temp.rename(columns = {"index": 'District', "District": 'Number of crimes related to narcotics'}, inplace=True)

fig = px.choropleth_mapbox(temp, geojson=g, color="Number of crimes related to narcotics", color_continuous_scale = "YlGnBu",
                           locations="District", featureidkey = "properties.dist_num",
                           center={"lat": 41.8781, "lon": -87.6298},
                           mapbox_style="carto-positron", zoom=9.5, opacity=0.7, height=800)
#fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()


# Overall, 11th district is heavily affected by crimes related to drugs, more than any other. 

# In[78]:


F = df[df['Primary Type']=='NARCOTICS'].groupby(['Year','District']).count()['ID'].to_frame().reset_index().rename(columns = {'ID':'Number of crimes related to narcotics'})
F['District'] = F['District'].map(lambda i: str(int(i)))
fig = px.choropleth_mapbox(F, geojson=g, color="Number of crimes related to narcotics", color_continuous_scale = "YlGnBu",
                           locations="District", featureidkey = "properties.dist_num", animation_frame='Year',
                           center={"lat": 41.8781, "lon": -87.6298}, mapbox_style="carto-positron", zoom=9.5, opacity=0.7, height=800)
fig.show()


# Throught the years we can observe that, as predicted, in every year mainly 11th district is affected by narcotics.

# In[79]:


temp = df[df['Primary Type']=='BATTERY']['District'].value_counts().to_frame().reset_index()
# district number as string required in further code
temp['index'] = temp['index'].map(lambda i: str(int(i))) 
# changing names of the data frame columns
temp.rename(columns = {"index": 'District', "District": 'Number of batteries'}, inplace=True)

fig = px.choropleth_mapbox(temp, geojson=g, color="Number of batteries", color_continuous_scale = "YlGnBu",
                           locations="District", featureidkey = "properties.dist_num",
                           center={"lat": 41.8781, "lon": -87.6298},
                           mapbox_style="carto-positron", zoom=9.5, opacity=0.7, height=800)
#fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()


# In[80]:


F = df[df['Primary Type']=='BATTERY'].groupby(['Year','District']).count()['ID'].to_frame().reset_index().rename(columns = {'ID':'Number of batteries'})
F['District'] = F['District'].map(lambda i: str(int(i)))
fig = px.choropleth_mapbox(F, geojson=g, color="Number of batteries", color_continuous_scale = "YlGnBu",
                           locations="District", featureidkey = "properties.dist_num", animation_frame='Year',
                           center={"lat": 41.8781, "lon": -87.6298}, mapbox_style="carto-positron", zoom=9.5, opacity=0.7, height=800)
fig.show()


# When it comes to battery, there's no main area that batteries are most common, because the entire south and central area of the city seems to have a lot of cases of battery. On the other hand, northern districts seem to be less susceptible to such a crime. 

# ## /Previous use of folium/

# In[81]:


m = folium.Map(location=[41.8781, -87.6298], zoom_start=10, width='70%', height='90%')


# In[82]:


import geopandas
DATA = geopandas.read_file("Boundaries_Police_Districts.geojson")

for i in range(len(DATA['dist_label'])):
    DATA['dist_label'] = DATA['dist_label'].replace(str(DATA.iloc[i][0]), str(DATA.iloc[i][0][:-2]))
# In[83]:


# district number as float
# reset_index to get a column with districts, otherwise the district numbers will be indexes
temp = df[df['Primary Type']=='THEFT']['District'].value_counts().to_frame().reset_index()
# district number as string required in further code
temp['index'] = temp['index'].map(lambda i: str(int(i))) 
# changing names of the data frame columns
temp.rename(columns = {"index": "dist_num", "District": "num_of_thefts"}, inplace=True)


# In[84]:


MERGED  = temp.merge(DATA, on='dist_num')


# In[85]:


folium.Choropleth(
    geo_data = 'Boundaries_Police_Districts.geojson',
    name = 'choropleth',
    data = MERGED,
    columns = ['dist_num', 'num_of_thefts'],
    key_on = 'feature.properties.dist_num',
    fill_color = 'YlGnBu',
    fill_opacity = 0.7,
    line_opacity = 0.2,
    legend_name = 'Overall number of thefts from 2012 to 2017',
).add_to(m)
folium.LayerControl().add_to(m)
m


# In[ ]:




