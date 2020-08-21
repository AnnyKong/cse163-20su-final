"""
Anny Kong
CSE 163 AC
This program implements the functions for the second research
question of the final project. It mainly makes multiple global
covid-19 plots and explores the relationship between different
countries/regions and the number of confirmed/deaths/recovered
cases.
Original file is located at
    https://colab.research.google.com/drive/1BXoGeS60R95IVPccp0SnrQYq6nESFs4F
"""

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()


def plot_geo_by_month(df, countries, m, t):
    """
      Plot a map of global Covid-19 cases by given type and month
    """
    df_type = df
    is_m = df_type['Last_Update'].dt.month == m
    df_type = df_type[is_m].groupby(['Latitude', 'Longitude'])[t].sum()
    df_type = df_type.reset_index(name=t)
    geo_data = df_type
    geo_data['coordinates'] = [
        Point(long, lat)
        for long, lat
        in zip(geo_data['Longitude'], geo_data['Latitude'])
    ]
    geo_data = geo_data.sort_values(t)

    fig, ax = plt.subplots(1, figsize=(15, 7))
    geo_data = gpd.GeoDataFrame(geo_data, geometry='coordinates')
    countries.plot(color='#EEEEEE', markersize=10, ax=ax)
    geo_data.plot(column=t, markersize=geo_data[t] / 50000, legend=True, ax=ax)
    ax.set_title('Global Covid-19 ' + t + ' Map - Month ' + str(m))
    fig.savefig('results/' + t + '_geo_vis_' + str(m) + '.png')


def plot_by_month(df, m, t):
    """
      Makes a bar plot of global Covid-19 cases by Country/Region with
      given type and month
    """
    df_type = df
    is_m = df_type['Last_Update'].dt.month == m
    df_type = df_type[is_m].groupby(['Country_Region'])[t].sum()
    df_type = df_type.reset_index(name=t)

    # fig, ax = plt.subplots(1, figsize=(15, 7))
    sns_plot = sns.catplot(x='Country_Region', y=t, kind='bar',
                           data=df_type, aspect=6)
    plt.xlabel('Country/Region')
    plt.ylabel("Number of " + t + " cases")
    plt.xticks(rotation=90)
    plt.title('Global Covid-19 ' + t + ' Cases by Country/Region - Month '
              + str(m))
    sns_plot.savefig('results/' + t + '_plot_' + str(m) + '.png')


def long_plot_by_month(df, m, t):
    """
      Makes a line plot of global Covid-19 cases by Longitude with
      given type and month
    """
    df_type = df
    is_m = df_type['Last_Update'].dt.month == m
    df_type = df_type[is_m].groupby(['Longitude', 'Latitude'])[t].sum()
    df_type = df_type.reset_index(name=t)

    sns_plot = sns.relplot(x='Longitude', y=t, kind='line',
                           data=df_type, aspect=6)
    plt.xlabel('Longitude')
    plt.ylabel("Number of " + t + " cases")
    plt.xticks(rotation=90)
    plt.title('Global Covid-19 ' + t + ' Cases by Longitude - Month ' + str(m))
    sns_plot.savefig('results/' + t + '_long_plot_' + str(m) + '.png')


def lat_plot_by_month(df, m, t):
    """
      Makes a line plot of global Covid cases by Latitude with
      given type and month
    """
    df_type = df
    is_m = df_type['Last_Update'].dt.month == m
    df_type = df_type[is_m].groupby(['Longitude', 'Latitude'])[t].sum()
    df_type = df_type.reset_index(name=t)

    sns_plot = sns.relplot(x='Latitude', y=t, kind='line',
                           data=df_type, aspect=6)
    plt.xlabel('Latitude')
    plt.ylabel("Number of " + t + " cases")
    plt.xticks(rotation=90)
    plt.title('Global Covid-19 ' + t + ' Cases by Latitude - Month ' + str(m))
    sns_plot.savefig('results/' + t + '_lat_plot_' + str(m) + '.png')


def main():
    df = pd.read_csv('combined.csv')

    # preprocess
    countries = gpd.read_file(
      'data/ne_110m_admin_0_countries.shp')
    df = df[['Last_Update', 'Country_Region', 'Longitude', 'Latitude',
             'Confirmed', 'Deaths', 'Recovered']].dropna()
    df['Last_Update'] = df['Last_Update'].apply(pd.to_datetime)
    months = df['Last_Update'].dt.month.unique()

    # geo map
    for i in sorted(months):
        plot_geo_by_month(df, countries, i, 'Confirmed')

    for i in sorted(months):
        plot_geo_by_month(df, countries, i, 'Deaths')

    for i in sorted(months):
        plot_geo_by_month(df, countries, i, 'Recovered')

    # bar plot
    for i in sorted(months):
        plot_by_month(df, i, 'Confirmed')

    for i in sorted(months):
        plot_by_month(df, i, 'Deaths')

    for i in sorted(months):
        plot_by_month(df, i, 'Recovered')

    # line plot
    for i in sorted(months):
        long_plot_by_month(df, i, 'Confirmed')
        lat_plot_by_month(df, i, 'Confirmed')

    for i in sorted(months):
        long_plot_by_month(df, i, 'Deaths')
        lat_plot_by_month(df, i, 'Deaths')

    for i in sorted(months):
        long_plot_by_month(df, i, 'Recovered')
        lat_plot_by_month(df, i, 'Recovered')


if __name__ == '__main__':
    main()
