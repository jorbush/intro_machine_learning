from datetime import datetime
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

def basic_data_exploration():
    # save filepath to variable for easier access
    melbourne_file_path = './input/melb_data.csv'
    # read the data and store data in DataFrame titled melbourne_data
    melbourne_data = pd.read_csv(melbourne_file_path)
    # print a summary of the data in Melbourne data
    print(melbourne_data.describe())

def exercise_explore_your_data():
    # Path of the file to read
    iowa_file_path = './input/train.csv'
    # Fill in the line below to read the file into a variable home_data
    home_data = pd.read_csv(iowa_file_path)
    # Print summary statistics in next line
    statistics = home_data.describe()
    print(statistics)
    # What is the average lot size (rounded to nearest integer)?
    avg_lot_size = round(statistics.loc['mean','LotArea'])
    print(avg_lot_size)
    # As of today, how old is the newest home (current year - the date in which it was built)
    newest_home_age = round(datetime.now().year - statistics.loc['max','YearBuilt'])
    print(newest_home_age)

def first_machine_learning_model():
    melbourne_file_path = './input/melb_data.csv'
    melbourne_data = pd.read_csv(melbourne_file_path)
    print(melbourne_data.columns)
    # The Melbourne data has some missing values (some houses for which some variables weren't recorded.)
    # We'll learn to handle missing values in a later tutorial.
    # Your Iowa data doesn't have missing values in the columns you use.
    # So we will take the simplest option for now, and drop houses from our data.
    # Don't worry about this much for now, though the code is:
    # dropna drops missing values (think of na as "not available")
    melbourne_data = melbourne_data.dropna(axis=0)
    # We'll use the dot notation to select the column we want to predict, which is called the prediction target.
    # By convention, the prediction target is called y.
    y = melbourne_data.Price
    print(y)
    # The columns that are inputted into our model (and later used to make predictions) are called "features."
    melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']
    # By convention, this data is called X.
    X = melbourne_data[melbourne_features]
    print(X.describe())
    print(X.head())
    '''
    The steps to building and using a model are:
    - Define: What type of model will it be? A decision tree? Some other type of model? 
                Some other parameters of the model type are specified too.
    - Fit: Capture patterns from provided data. This is the heart of modeling.
    - Predict: Just what it sounds like
    - Evaluate: Determine how accurate the model's predictions are.
    '''
    # Define model. Specify a number for random_state to ensure same results each run
    melbourne_model = DecisionTreeRegressor(random_state=1)
    # Fit model
    melbourne_model.fit(X, y)
    '''
    Many machine learning models allow some randomness in model training. Specifying a number for `random_state` 
    ensures you get the same results in each run. This is considered a good practice. You use any number, 
    and model quality won't depend meaningfully on exactly what value you choose.
    '''
    print("Making predictions for the following 5 houses:")
    print(X.head())
    print("The predictions are")
    print(melbourne_model.predict(X.head()))

if __name__ == '__main__':
    # basic_data_exploration()
    # exercise_explore_your_data()
    first_machine_learning_model()