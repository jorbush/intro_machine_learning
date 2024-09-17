from datetime import datetime
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
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

def exercise_first_machine_learning_model():
    melbourne_file_path = './input/train.csv'
    home_data = pd.read_csv(melbourne_file_path)
    # print the list of columns in the dataset to find the name of the prediction target
    print(home_data.columns)
    y = home_data.SalePrice
    print(y)
    # Create the list of features below
    feature_names = [
        "LotArea",
        "YearBuilt",
        "1stFlrSF",
        "2ndFlrSF",
        "FullBath",
        "BedroomAbvGr",
        "TotRmsAbvGrd"
    ]
    # Select data corresponding to features in feature_names
    X = home_data[feature_names]
    # Review data
    # print description or statistics from X
    print(X.describe())
    # print the top few lines
    print(X.head())
    # For model reproducibility, set a numeric value for random_state when specifying the model
    iowa_model = DecisionTreeRegressor(random_state=1)
    # Fit the model
    iowa_model.fit(X, y)
    predictions = iowa_model.predict(X)
    print(predictions)

def model_validation():
    # Load data
    melbourne_file_path = './input/melb_data.csv'
    melbourne_data = pd.read_csv(melbourne_file_path)
    # Filter rows with missing price values
    filtered_melbourne_data = melbourne_data.dropna(axis=0)
    # Choose target and features
    y = filtered_melbourne_data.Price
    melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea',
                          'YearBuilt', 'Lattitude', 'Longtitude']
    X = filtered_melbourne_data[melbourne_features]
    # Define model
    melbourne_model = DecisionTreeRegressor()
    # Fit model
    melbourne_model.fit(X, y)
    '''
    You'll want to evaluate almost every model you ever build. In most (though not all) applications, 
    the relevant measure of model quality is predictive accuracy. In other words, will the model's predictions 
    be close to what actually happens.
    There are many metrics for summarizing model quality, but we'll start with one called 
    Mean Absolute Error (also called MAE).
    '''
    predicted_home_prices = melbourne_model.predict(X)
    mean_absolute_error(y, predicted_home_prices)

    # split data into training and validation data, for both features and target
    # The split is based on a random number generator. Supplying a numeric value to
    # the random_state argument guarantees we get the same split every time we
    # run this script.
    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)
    # Define model
    melbourne_model = DecisionTreeRegressor()
    # Fit model
    melbourne_model.fit(train_X, train_y)
    # get predicted prices on validation data
    val_predictions = melbourne_model.predict(val_X)
    print(mean_absolute_error(val_y, val_predictions))

if __name__ == '__main__':
    # basic_data_exploration()
    # exercise_explore_your_data()
    # first_machine_learning_model()
    # exercise_first_machine_learning_model()
    model_validation()