"""
Anny Kong
CSE 163 AC
This program implements the test functions for the second research
question of the final project.
Colab Version of the test file located at
    https://colab.research.google.com/drive/1BXoGeS60R95IVPccp0SnrQYq6nESFs4F?usp=sharing
"""

import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()


def test_plot_geo_by_month(df, countries, m, t):
    """
      Tests the plot_geo_by_month function
    """
    df_type = df
    is_m = df_type['Last_Update'].dt.month == m
    df_type = df_type[is_m].groupby(['Latitude', 'Longitude'])[t].sum()
    df_type = df_type.reset_index(name=t)
    print(df_type)

    geo_data = df_type
    geo_data['coordinates'] = [
        Point(long, lat)
        for long, lat
        in zip(geo_data['Longitude'], geo_data['Latitude'])
    ]
    geo_data = geo_data.sort_values(t)
    print(geo_data)

    fig, ax = plt.subplots(1, figsize=(15, 7))
    geo_data = gpd.GeoDataFrame(geo_data, geometry='coordinates')
    countries.plot(color='#EEEEEE', markersize=10, ax=ax)
    geo_data.plot(column=t, markersize=geo_data[t] / 50000, legend=True, ax=ax)
    ax.set_title('Global Covid-19 ' + t + ' Map - Month ' + str(m))
    fig.savefig('test_results/' + t + '_geo_vis_' + str(m) + '.png')


def table_by_month(df, m, t):
    """
      Tests the given t cases in month m by printing out
      the top ten list table.
    """
    df_type = df
    is_m = df_type['Last_Update'].dt.month == m
    df_type = df_type[is_m].groupby(['Country_Region'])[t].sum()
    df_type = df_type.reset_index(name=t)
    print('Global Covid-19 ' + t + ' Cases by Country/Region - Month '
          + str(m))
    print(df_type.sort_values(t, ascending=False)[:10])
    print()
    test_results = df_type.sort_values(t, ascending=False)[:10]
    test_results.to_csv('test_results/' + t + '_case_by_region_month_'
                        + str(m) + '.csv')


def main():
    df = pd.read_csv('combined.csv')

    # preprocess
    countries = gpd.read_file(
      'data/ne_110m_admin_0_countries.shp')
    df = df[['Last_Update', 'Country_Region', 'Longitude', 'Latitude',
             'Confirmed', 'Deaths', 'Recovered']].dropna()
    df['Last_Update'] = df['Last_Update'].apply(pd.to_datetime)
    months = df['Last_Update'].dt.month.unique()

    for i in sorted(months):
        table_by_month(df, i, 'Confirmed')
    for i in sorted(months):
        table_by_month(df, i, 'Deaths')
    for i in sorted(months):
        table_by_month(df, i, 'Recovered')
    test_plot_geo_by_month(df, countries, 5, 'Confirmed')


if __name__ == '__main__':
    main()
