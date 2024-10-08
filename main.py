from datetime import datetime
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
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

def exercise_model_validation():
    # Path of the file to read
    iowa_file_path = './input/train.csv'
    home_data = pd.read_csv(iowa_file_path)
    y = home_data.SalePrice
    feature_columns = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
    X = home_data[feature_columns]
    # Specify Model
    iowa_model = DecisionTreeRegressor()
    # Fit Model
    iowa_model.fit(X, y)
    print("First in-sample predictions:", iowa_model.predict(X.head()))
    print("Actual target values for those homes:", y.head().tolist())
    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
    iowa_model = DecisionTreeRegressor(random_state=1)
    iowa_model.fit(train_X, train_y)
    # Predict with all validation observations
    val_predictions = iowa_model.predict(val_X)
    # print the top few validation predictions
    print(val_predictions)
    # print the top few actual prices from validation data
    print(val_y.head())
    val_mae = mean_absolute_error(val_y, val_predictions)
    print(val_mae)

def underfitting_and_overfitting():
    # This is a phenomenon called overfitting, where a model matches the training data almost perfectly,
    # but does poorly in validation and other new data.
    # At an extreme, if a tree divides houses into only 2 or 4, each group still has a wide variety of houses.
    # Resulting predictions may be far off for most houses, even in the training data (and it will be bad in
    # validation too for the same reason). When a model fails to capture important distinctions and patterns
    # in the data, so it performs poorly even in training data, that is called underfitting.

    def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
        model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
        model.fit(train_X, train_y)
        preds_val = model.predict(val_X)
        mae = mean_absolute_error(val_y, preds_val)
        return (mae)

    melbourne_file_path = './input/melb_data.csv'
    melbourne_data = pd.read_csv(melbourne_file_path)
    filtered_melbourne_data = melbourne_data.dropna(axis=0)
    y = filtered_melbourne_data.Price
    melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea',
                          'YearBuilt', 'Lattitude', 'Longtitude']
    X = filtered_melbourne_data[melbourne_features]
    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)
    # compare MAE with differing values of max_leaf_nodes
    for max_leaf_nodes in [5, 50, 500, 5000]:
        my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
        print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" % (max_leaf_nodes, my_mae))

def exercise_underfitting_and_overfitting():
    # Path of the file to read
    iowa_file_path = './input/train.csv'
    home_data = pd.read_csv(iowa_file_path)
    # Create target object and call it y
    y = home_data.SalePrice
    # Create X
    features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
    X = home_data[features]
    # Split into validation and training data
    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
    # Specify Model
    iowa_model = DecisionTreeRegressor(random_state=1)
    # Fit Model
    iowa_model.fit(train_X, train_y)
    # Make validation predictions and calculate mean absolute error
    val_predictions = iowa_model.predict(val_X)
    val_mae = mean_absolute_error(val_predictions, val_y)
    print("Validation MAE: {:,.0f}".format(val_mae))

    def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
        model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
        model.fit(train_X, train_y)
        preds_val = model.predict(val_X)
        mae = mean_absolute_error(val_y, preds_val)
        return (mae)

    candidate_max_leaf_nodes = [5, 25, 50, 100, 250, 500]
    current_mae = float('inf')
    best_leaf_nodes = None
    # Write loop to find the ideal tree size from candidate_max_leaf_nodes
    for max_leaf_nodes in candidate_max_leaf_nodes:
        my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
        print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" % (max_leaf_nodes, my_mae))
        if my_mae < current_mae:
            current_mae = my_mae
            best_leaf_nodes = max_leaf_nodes
    best_tree_size = best_leaf_nodes
    print(best_tree_size)

    # Fill in argument to make optimal size and uncomment
    final_model = DecisionTreeRegressor(max_leaf_nodes=best_tree_size, random_state=0)
    # fit the final model and uncomment the next two lines
    final_model.fit(X, y)

def random_forests():
    # The random forest uses many trees, and it makes a prediction by averaging the predictions of each component tree.
    # It generally has much better predictive accuracy than a single decision tree and it works well with default
    # parameters. If you keep modeling, you can learn more models with even better performance, but many of those are
    # sensitive to getting the right parameters
    melbourne_file_path = './input/melb_data.csv'
    melbourne_data = pd.read_csv(melbourne_file_path)
    # Filter rows with missing values
    melbourne_data = melbourne_data.dropna(axis=0)
    # Choose target and features
    y = melbourne_data.Price
    melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea',
                          'YearBuilt', 'Lattitude', 'Longtitude']
    X = melbourne_data[melbourne_features]
    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)
    # We build a random forest model similarly to how we built a decision tree in scikit-learn - this time using
    # the RandomForestRegressor class instead of DecisionTreeRegressor.
    forest_model = RandomForestRegressor(random_state=1)
    forest_model.fit(train_X, train_y)
    melb_preds = forest_model.predict(val_X)
    print(mean_absolute_error(val_y, melb_preds))

def exercise_random_forests():
    # Path of the file to read
    iowa_file_path = './input/train.csv'
    home_data = pd.read_csv(iowa_file_path)
    # Create target object and call it y
    y = home_data.SalePrice
    # Create X
    features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
    X = home_data[features]
    # Split into validation and training data
    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
    # Specify Model
    iowa_model = DecisionTreeRegressor(random_state=1)
    # Fit Model
    iowa_model.fit(train_X, train_y)
    # Make validation predictions and calculate mean absolute error
    val_predictions = iowa_model.predict(val_X)
    val_mae = mean_absolute_error(val_predictions, val_y)
    print("Validation MAE when not specifying max_leaf_nodes: {:,.0f}".format(val_mae))
    # Using best value for max_leaf_nodes
    iowa_model = DecisionTreeRegressor(max_leaf_nodes=100, random_state=1)
    iowa_model.fit(train_X, train_y)
    val_predictions = iowa_model.predict(val_X)
    val_mae = mean_absolute_error(val_predictions, val_y)
    print("Validation MAE for best value of max_leaf_nodes: {:,.0f}".format(val_mae))

    # Define the model. Set random_state to 1
    rf_model = RandomForestRegressor(random_state=1)
    # fit your model
    rf_model.fit(train_X, train_y)
    # Calculate the mean absolute error of your Random Forest model on the validation data
    rf_val_mae = mean_absolute_error(val_y, rf_model.predict(val_X))
    print("Validation MAE for Random Forest Model: {}".format(rf_val_mae))

def machine_learning_competitions():
    # Load the data, and separate the target
    iowa_file_path = './input/train.csv'
    home_data = pd.read_csv(iowa_file_path)
    y = home_data.SalePrice
    # Create X (After completing the exercise, you can return to modify this line!)
    features = ['LotArea', 'YearBuilt', '1stFlrSF', '2ndFlrSF', 'FullBath', 'BedroomAbvGr', 'TotRmsAbvGrd']
    # Select columns corresponding to features, and preview the data
    X = home_data[features]
    X.head()
    # Split into validation and training data
    train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=1)
    # Define a random forest model
    rf_model = RandomForestRegressor(random_state=1)
    rf_model.fit(train_X, train_y)
    rf_val_predictions = rf_model.predict(val_X)
    rf_val_mae = mean_absolute_error(rf_val_predictions, val_y)
    print("Validation MAE for Random Forest Model: {:,.0f}".format(rf_val_mae))

    # To improve accuracy, create a new Random Forest model which you will train on all training data
    rf_model_on_full_data = RandomForestRegressor(random_state=1)
    # fit rf_model_on_full_data on all data from the training data
    rf_model_on_full_data.fit(X, y)
    # path to file you will use for predictions
    test_data_path = './input/test.csv'
    # read test data file using pandas
    test_data = pd.read_csv(test_data_path)
    # create test_X which comes from test_data but includes only the columns you used for prediction.
    # The list of columns is stored in a variable called features
    test_X = test_data[features]
    # make predictions which we will submit.
    test_preds = rf_model_on_full_data.predict(test_X)

if __name__ == '__main__':
    # basic_data_exploration()
    # exercise_explore_your_data()
    # first_machine_learning_model()
    # exercise_first_machine_learning_model()
    # model_validation()
    # exercise_model_validation()
    # underfitting_and_overfitting()
    # exercise_underfitting_and_overfitting()
    # random_forests()
    # exercise_random_forests()
    machine_learning_competitions()