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

3.1. Run the following Python script and see results in `results/part1`

```
python3 cse163_final_part1.py
```

### Research Question 2 ([Results Webpage](https://annykong.github.io/cse163-20su-final/)/[Results](https://github.com/AnnyKong/cse163-20su-final/tree/master/results))

By data visualization, what can we say about how different regions (countries) may affect the number of confirmed, deaths, and recovered cases?

**The month is a large factor in the relationship with the current Covid-19 condition of different regions. From February to August, Covid-19 has spread all over the world. Though for those neighboring affected countries/regions, they are more likely to be affected, there isn’t evidence for a causal relationship between longitude/latitude and the number of cases.**


3.2. Run the following Python script and see results in `results/` (We also included a results gallery for RQ2 in `docs/`)

```
python3 research_question_2.py
```
4.2. Run the following Python test script and see test results in `test_results/`

```
python3 test_research_question_2.py
```
(OPTIONAL) 5.2. You may want to checkout pre-run output at [research_question_2.ipynb](research_question_2.ipynb) or [Colab script](https://colab.research.google.com/drive/1BXoGeS60R95IVPccp0SnrQYq6nESFs4F?usp=sharing) which was used for development and testing purposes.
