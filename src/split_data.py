from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR/'data'/'processed'/'user_item_features.csv'
TRAIN_FILE = BASE_DIR/'data'/'processed'/'train.csv'
TEST_FILE = BASE_DIR/'data'/'processed'/'test.csv'

def time_based_split(data: pd.DataFrame, test_size: float = 0.2):
    """
    Split the data into training and testing sets based on time.

    Args:
        data (pd.DataFrame): The input DataFrame containing user-item features.
        test_size (float): The proportion of the dataset to include in the test split.

    Returns:
        train_data (pd.DataFrame): The training set.
        test_data (pd.DataFrame): The testing set.
    """
    # Sort the data by last interaction timestamp
    data_sorted = data.sort_values(by='last_interaction')

    # Calculate the index for splitting
    split_index = int(len(data_sorted) * (1 - test_size))

    # Split the data into training and testing sets
    train_data = data_sorted.iloc[:split_index]
    test_data = data_sorted.iloc[split_index:]

    return train_data, test_data

def main():

    print("Loading user-item features data...")
    data=pd.read_csv(INPUT_FILE)

    print("\nData shape:", data.shape)

    print('Total interactions:', data['interaction_count'].sum())
    print("Time based splitting the data into train and test sets...")
    train, test =time_based_split(data)

    print("Saving the train and test sets...")
    train.to_csv(TRAIN_FILE, index=False)
    test.to_csv(TEST_FILE, index=False)

    print("\nTrain and test sets saved successfully.")

    print("Train Interactions:", train['interaction_count'].sum())
    print("Test Interactions:", test['interaction_count'].sum())

    print("Train Period")
    print(train['last_interaction'].min(), "to", train['last_interaction'].max())
    print("Test Period")
    print(test['last_interaction'].min(), "to", test['last_interaction'].max())

if __name__ == "__main__":
    main()