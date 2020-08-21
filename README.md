# COVID-19 Trend Prediction and Analysis

CSE163 20SU Final Project at UW

As the health crisis triggered by the pandemic COVID-19 keeps the world in its grasp, most people are forced to work from home as well as to practice social distancing. Our project could potentially foretell how future pandemics are going. It will focus on how different government policies affect the spread of this virus. We will visualize the virus spread with respect to different regions/countries and analyze how the virus spread is correlated to different geolocations, and see how the projection of infected cases will be like in the near future.

## Authors

- Anny Kong
- Forrest Jiang
- Zealer Xiao

## How to install the libraries required

Make sure you have Python with version 3.7 or higher. Do `pip install` for the following libs:

- geopandas
- pandas
- matplotlib
- seaborn
- numpy
- sklearn

## How to reproduce our results

1. Clone the repo with HTTPS or SSH

```
git clone https://github.com/AnnyKong/cse163-20su-final.git
cd cse163-20su-final
```

2. Unzip preprocessed data csv file

```
unzip combined.csv.zip
```

### Research Question 1 ([Results](https://github.com/AnnyKong/cse163-20su-final/tree/master/results/part1))

What’s the future trend of COVID-19 in each country?

**Our result shows that if every country does not enforce more health guidelines,
the number will continue to grow according to the polynomial model, with 83% accuracy.**

1. Run the following Python script and see results in `results/part1`

```
python3 cse163_final_part1.py
```

### Research Question 2 ([Results](https://github.com/AnnyKong/cse163-20su-final/tree/master/results))

By data visualization, what can we say about how different regions (countries) may affect the number of confirmed, deaths, and recovered cases?

```

```

1. Run the following Python script and see results in `results/`

```
python3 cse163_final_part2.py
```

(OPTIONAL) 2. You may want to checkout pre-run output at [cse163_final_part2.ipynb](cse163_final_part2.ipynb) or [Colab script for development purposes](https://colab.research.google.com/drive/1BXoGeS60R95IVPccp0SnrQYq6nESFs4F?usp=sharing)

### Research Question 3

How do infection rate and fatality rate of COVID-19 relate to regions’ income level?

1.

```

```

2.

```

```
