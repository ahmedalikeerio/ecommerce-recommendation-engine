from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = BASE_DIR / "data" / "raw" / "events.csv"
OUTPUT_DATA_DIR = BASE_DIR / "data" / "processed" / "user_item_features.csv"

def load_data(path: Path)-> pd.DataFrame:
    """
    Load the raw data from the specified path.

    Args:
        path (Path): The path to the raw data file."""

    events = pd.read_csv(path)
    return events

def preprocess_events(events: pd.DataFrame) -> pd.DataFrame:
    '''
    Preprocess the events data, clean events data and create user-interaction dataframe with user-item features.
    '''

    # keep only required features
    interactions= events[['visitorid', 'itemid', 'event', 'timestamp']].copy()

    # convert timestamp to datetime
    interactions['timestamp']=pd.to_datetime(interactions['timestamp'], unit='ms')

    # Rename columns
    interactions.rename(columns={'visitorid': 'user_id', 'itemid': 'item_id'}, inplace=True)

    # initial weight implicitly assigned to each event type
    event_weights ={'view':1, 'addtocart':3, 'transaction':5 }
    interactions['event_weight'] = interactions['event'].map(event_weights)

    # remove unexpected event type
    interactions.dropna(subset=['event_weight'], inplace=True)


    # create user-item features
    user_item_features = (interactions.groupby(['user_id', 'item_id']).agg(
        interaction_score=('event_weight','sum'),
        interaction_count=('event', 'count'),
        viewed = ('event', lambda x: (x=='view').sum()),
        added_to_cart = ('event', lambda x: (x=='addtocart').sum()),
        purchased= ('event', lambda x: (x=='transaction').sum()),
        last_interaction=('timestamp', 'max')
    ).reset_index())
    return user_item_features

    # save dataframe to csv
def save_to_csv(df: pd.DataFrame, path: Path)-> None:
    """
    Save the DataFrame to a CSV file at the specified path.

    Args:
        df (pd.DataFrame): The DataFrame to save.
        path (Path): The path to save the CSV file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)  # Create directories if they don't exist
    df.to_csv(path, index=False)


def main():

    print("Loading raw data...")
    events = load_data(RAW_DATA_DIR)
    print("Preprocessing events data...")
    user_item_features = preprocess_events(events)

    print("Unique users:", user_item_features['user_id'].nunique())
    print("Saving user-item features to CSV...")
    save_to_csv(user_item_features, OUTPUT_DATA_DIR)
    print("Preprocessing completed successfully.")
if __name__ == "__main__":
    main()
