import pandas as pd

def basic_data_exploration():
    # save filepath to variable for easier access
    melbourne_file_path = './input/melb_data.csv'
    # read the data and store data in DataFrame titled melbourne_data
    melbourne_data = pd.read_csv(melbourne_file_path)
    # print a summary of the data in Melbourne data
    print(melbourne_data.describe())

def exercise_explore_your_data():
    return NotImplementedError

if __name__ == '__main__':
    basic_data_exploration()