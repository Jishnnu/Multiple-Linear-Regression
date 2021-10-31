# LINEAR REGRESSION PROJECT 01 : MULTIPLE LINEAR REGRESSION


# IMPORT ALL THE NECESSARY LIBRARIES
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


data = pd.read_csv(r"C:\Multiple Linear Regression\Car_Models.csv")
car_info = pd.get_dummies(data.car)


# CLEANING THE DATA
cleaned_data = pd.concat([data, car_info], axis='columns')
cleaned_data = cleaned_data.drop(['car', 'Audi A5'], axis='columns')


# DETERMINING X & Y
regression = LinearRegression()
X = cleaned_data.drop(['sell_price'], axis='columns')
Y = cleaned_data.sell_price

# FITTING THE MODEL
regression.fit(X, Y)

# SCORE
score = regression.score(X, Y) * 100
print("SCORE : " + str(score))
