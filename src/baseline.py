from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

TRAIN_DATA= BASE_DIR/'data'/'processed'/'train.csv'
TEST_DATA= BASE_DIR/'data'/'processed'/'test.csv'

# load the train and test data

train=pd.read_csv(TRAIN_DATA)
test=pd.read_csv(TEST_DATA)

#######################
# BASELINE - 1
#######################

def most_viewed_baseline(train ,k=10):
    """
    Most Viewed Baseline Model

    Args:
        train (pd.DataFrame): The training set containing user-item features.
        test (pd.DataFrame): The testing set containing user-item features.

    Returns:
        recommendations (dict): A dictionary where keys are user IDs and values are lists of recommended item IDs.
    """
    # Get the most viewed items from the training set
    most_viewed_items = train.groupby('item_id')['most_viewed'].sum().sort_values(ascending=False).head(k).index.tolist()

    # Create a dictionary to store recommendations for each user
    # recommendations = {most_viewed_items}

    # # For each user in the test set, recommend the most viewed items
    # for user_id in test['user_id'].unique():
    #     recommendations[user_id] = most_viewed_items[:10]  # Recommend top 10 most viewed items

    return most_viewed_items


def most_engaged_baseline(train, k=10):
    """
    Most Engaged Baseline Model

    Args:
        train (pd.DataFrame): The training set containing user-item features.
        test (pd.DataFrame): The testing set containing user-item features.

    Returns:
        recommendations (dict): A dictionary where keys are user IDs and values are lists of recommended item IDs.
    """
    # Get the most engaged items from the training set
    most_engaged_items = train.groupby('item_id')['interaction_score'].sum().sort_values(ascending=False).head(k).index.tolist()

    # Create a dictionary to store recommendations for each user
    # recommendations = {most_engaged_items}

    # # For each user in the test set, recommend the most engaged items
    # for user_id in test['user_id'].unique():
    #     recommendations[user_id] = most_engaged_items[:10]  # Recommend top 10 most engaged items

    return most_engaged_items

def most_purchased_baseline(train, k=10):
    """
    Most Purchased Baseline Model

    Args:
        train (pd.DataFrame): The training set containing user-item features.
        test (pd.DataFrame): The testing set containing user-item features.

    Returns:
        recommendations (dict): A dictionary where keys are user IDs and values are lists of recommended item IDs.
    """
    # Get the most purchased items from the training set
    most_purchased_items = train.groupby('item_id')['purchased'].sum().sort_values(ascending=False).head(k).index.tolist()

    # Create a dictionary to store recommendations for each user
    # recommendations = {most_purchased_items}

    # # For each user in the test set, recommend the most purchased items
    # for user_id in test['user_id'].unique():
    #     recommendations[user_id] = most_purchased_items[:10]  # Recommend top 10 most purchased items

    return most_purchased_items

def most_added_to_cart_baseline(train, k=10):

    most_add_to_cart =(train
        .groupby("item_id")["added_to_cart"]
        .sum()
        .sort_values(ascending=False)
        .head(k)
        .index
        .tolist())
    return most_add_to_cart

if __name__ == '__main__':
    print("Most Viewed :\n")
    print(most_engaged_baseline(train))


    print("\n Most Engaged: \n")
    print(most_engaged_baseline(train))


    print("\n Most Purchsed: \n")
    print(most_purchased_baseline(train))

    print("\n Most Added to cart: \n")
    print(most_added_to_cart_baseline(train))

