'''
Zealer Xiao
CSE 163 AB
This program implements the functions for the first research
question of the final project. It mainly uses machine learning algorithm
along with plots to explore the trend for each country.
'''
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge


def merge_data():
    '''
    read csv files from local folders and combine them into a single csv file.
    return the combined csv file
    '''
    dfs = []
    cols = {'Province/State': 'Province_State',
            'Country/Region': 'Country_Region',
            'Last Update': 'Last_Update', 'Lat': 'Latitude',
            'Long_': 'Longitude'}
    for csv in os.listdir():
        if csv.endswith('csv'):  # need to change the path
            df = pd.read_csv(csv)
            try:
                df.rename(columns=cols, inplace=True)
            except Exception:
                pass
            dfs.append(df)
    df = pd.concat(dfs)
    df.reset_index(inplace=True, drop=True)
    return df


def read_data():
    '''
    read combined file and return a dataframe object
    '''
    df = pd.read_csv('combined.csv')
    df = df.astype({'Province_State': str, 'Country_Region': str,
                    'Last_Update': 'datetime64', 'Admin2': str,
                    'Combined_Key': str})
    return df


def add_day_col(df):
    '''
    takes a dataframe object
    return df that contains days for each country from start to the last day.
    '''
    df['Last_Update'] = df['Last_Update'].apply(lambda t: t.date())
    df1 = df.groupby(['Country_Region', 'Last_Update']).sum().reset_index()
    df1['Day'] = df1.groupby('Country_Region')['Last_Update'].cumcount()
    return df1


def train_model(x, y, poly=False, ploy_degree=2):
    '''
    takes training data x and y
    return the predicted values and score of that prediction
    '''
    x = x.values.reshape(-1, 1)
    y = y.values.reshape(-1, 1)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=1)
    if poly:
        poly_model = make_pipeline(PolynomialFeatures(
            degree=ploy_degree),  LinearRegression())
        poly_model.fit(x_train, y_train)
        prediction = poly_model.predict(np.sort(x_test, axis=0))
        sorted_x_test = np.sort(x_test, axis=0)
        return sorted_x_test, prediction, poly_model.score(x_test, y_test)
    else:
        reg = LinearRegression()
        reg.fit(x_train, y_train)
        prediction = reg.predict(x_test)
        return x_test, prediction, reg.score(x_test, y_test)


def calc_linear_score(df1):
    '''
    takes a dataframe object
    return a Series of country:score using this linear regression model.
    '''
    linear_result = pd.DataFrame({'Country': [], 'Score': []})
    for country in df1['Country_Region'].unique():
        mask = df1[df1['Country_Region'] == country]
        if len(mask) > 5:
            x = mask['Day']
            y = mask['Confirmed']
            _, _, score = train_model(x, y)
            linear_result = linear_result.append(
                {'Country': country, 'Score': score}, ignore_index=True)
    return linear_result


def x_y(name, df1):
    '''
    takes a country name and plot the Day vs Cases for that country
    return day and cases
    '''
    country = df1[(df1['Country_Region'] == name)]
    plt.title(name)
    plt.plot('Day', 'Confirmed', data=country, label=name)
    x = df1[df1['Country_Region'] == name]['Day']
    y = df1[df1['Country_Region'] == name]['Confirmed']
    return x, y


def plot_India_US_linear(df1):
    '''
    takes a dataframe object
    plot graphs for US and India using prediction and actual trend
    '''
    x, y = x_y('India', df1)
    train, prediction, _ = train_model(x, y)
    plot_graph(train, prediction, 'India_linear')
    x, y = x_y('US', df1)
    train, prediction, _ = train_model(x, y)
    plot_graph(train, prediction, 'US_linear')


def ridge_regression(country, df1, alpha=0.1,):
    '''
    take a country name, a penelization factor
    return x_test, prediction, and score
    if country is not applicable , return None, None, NAN
    '''
    ridge = Ridge(alpha=alpha, normalize=True)
    mask = df1[df1['Country_Region'] == country]
    if len(mask) > 5:
        x = mask['Day'].values.reshape(-1, 1)
        y = mask['Confirmed'].values.reshape(-1, 1)
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=0.3, random_state=1)
        ridge.fit(x_train, y_train)
        ridge_predict = ridge.predict(x_test)
        return x_test, ridge_predict, ridge.score(x_test, y_test)
    return None, None, np.nan


def calc_ridge_score(df1):
    '''
    takes a dataframe object
    return a Series of country:score using this ridge regression model.
    '''
    ridge_result = pd.DataFrame({'Country': [], 'Score': []})
    for country in df1['Country_Region'].unique():
        _, _, score = ridge_regression(country, df1)
        ridge_result = ridge_result.append(
            {'Country': country, 'Score': score}, ignore_index=True)
    return ridge_result


def plot_India_US_ridge(df1):
    '''
    takes a dataframe object
    plot graphs for US and India using prediction and actual trend
    '''
    _, _ = x_y('India', df1)
    train, prediction, _ = ridge_regression('India', df1)
    plot_graph(train, prediction, 'India_ridge')
    _, _ = x_y('US', df1)
    train, prediction, _ = ridge_regression('US', df1)
    plot_graph(train, prediction, 'US_ridge')


def calc_poly_score(df1, degree=2):
    '''
    takes a dataframe, and a degree for the polynomial regression
    return the accuracy for each country in a new dataframe
    '''
    polynimial_result = pd.DataFrame({'Country': [], 'Score': []})
    for country in df1['Country_Region'].unique():
        mask = df1[df1['Country_Region'] == country]
        if len(mask) > 5:
            x = mask['Day']
            y = mask['Confirmed']
            _, _, score = train_model(x, y, True, ploy_degree=degree)
            polynimial_result = polynimial_result.append(
                {'Country': country, 'Score': score}, ignore_index=True)
    return polynimial_result


def plot_India_US_poly(df1):
    '''
    takes a dataframe object
    plot graphs for US and India using prediction and actual trend
    '''
    x, y = x_y('India', df1)
    train, prediction, _ = train_model(x, y, True, 3)
    plot_graph(train, prediction, 'India_poly')
    x, y = x_y('US', df1)
    train, prediction, _ = train_model(x, y, True,  3)
    plot_graph(train, prediction, 'US_poly')


def plot_graph(x, y, name):
    '''
    takes x ,y data plot Day vs Cases
    '''
    plt.xlabel('Day')
    plt.ylabel('Confirmed cases')
    plt.plot(x, y, label='prediction')
    plt.legend()
    plt.savefig('results/part1/'+name+'.png')
    plt.clf()


def tune_param_poly(df1):
    '''
    takes a dataframe object tune the polynomial regression model
    using grid search and plot the degree Vs accuracy graph
    '''
    results = []

    for i in range(1, 8):
        polynimial_result = calc_poly_score(df1, i)
        results.append(polynimial_result.mean())
    plt.xlabel('degree')
    plt.ylabel('accuracy')
    plt.plot(np.arange(1, 8), results, 'bo')
    plt.savefig('results/part1/'+'degree_vs_accuracy.png')


def extrapolate(df, country, start_day, end_day):
    '''
    takes a dataframe, country name, start_day(int) and end_day(int)
    plot Day Vs Confirmed Cases in this day range for the given country.
    '''
    mask = df[df['Country_Region'] == country]
    x_train = mask['Day'].values.reshape(-1, 1)
    y_train = mask['Confirmed'].values.reshape(-1, 1)
    poly_model = make_pipeline(
        PolynomialFeatures(degree=5),  LinearRegression())
    poly_model.fit(x_train, y_train)
    x_predict = np.arange(start_day, end_day+1).reshape(-1, 1)
    prediction = poly_model.predict(x_predict)
    plt.plot(x_predict, prediction)
    plt.xlabel('Day')
    plt.title(country+' prediction')
    plt.ylabel('Confirmed cases')
    plt.savefig('results/part1/'+country+'_prediction')
    plt.clf()


def main():
    df = read_data()
    df1 = add_day_col(df)
    linear_result = calc_linear_score(df1)
    print('Average Score using linear regression: ', linear_result.mean())
    plot_India_US_linear(df1)
    ridge_result = calc_ridge_score(df1)
    print('Average Score using ridge regression: ', ridge_result.mean())
    plot_India_US_ridge(df1)
    tune_param_poly(df1)
    polynimial_result = calc_poly_score(df1, 5)
    print('Average Score using polynomial regression: ',
          polynimial_result['Score'].mean())
    plot_India_US_poly(df1)
    extrapolate(df1, 'India', 200, 350)
    extrapolate(df1, 'US', 200, 350)


if __name__ == "__main__":
    main()
