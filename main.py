from datetime import datetime
import pandas as pd

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

if __name__ == '__main__':
    # basic_data_exploration()
    exercise_explore_your_data()