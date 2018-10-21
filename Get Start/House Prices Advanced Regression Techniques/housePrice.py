import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

#Read the data
train = pd.read_csv('F:\kaggle\Get Start\House Prices Advanced Regression Techniques\data/train.csv')

# pull data into target (y) and predictors (X)
train_y = train.SalePrice
predictor_cols = ['LotArea', 'OverallQual', 'YearBuilt', 'TotRmsAbvGrd']

# Create training predictors data
train_x = train[predictor_cols]

my_model = RandomForestRegressor()
my_model.fit(train_x, train_y)

# Read the test data
test = pd.read_csv('F:\kaggle\Get Start\House Prices Advanced Regression Techniques\data/test.csv')
# Treat the test data in the same way as training data. In this case, pull same columns.
test_x = test[predictor_cols]

# Use the model to make predictions
predicted_prices = my_model.predict(test_x)
# We will look at the predicted prices to ensure we have something sensible.
print(predicted_prices)

my_submission = pd.DataFrame({'Id':test.Id, 'SalePrice':predicted_prices})
# you could use any filename. We choose submission here
my_submission.to_csv('F:\kaggle\Get Start\House Prices Advanced Regression Techniques\data/submission.csv', index=False)