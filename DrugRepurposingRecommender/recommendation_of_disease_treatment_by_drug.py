import numpy as np
import pandas as pd
from pandas import DataFrame, Series
import re
from recommendation_system import RecommendationSystem
from analyze_profiles import calculate_drug_dist, calculate_disease_dist
from analyze_profiles import load_medical_drugs_profiles

status_to_rating_map = {
    'Approved': 2,
    'Withdrawn (Phase 1)': -1/2, 'Withdrawn (Phase 2)': -2/6, 'Withdrawn (Phase 3)': -1/6,
    'Suspended (Phase 1)': 1/2, 'Suspended (Phase 2)': 4/6, 'Suspended (Phase 3)': 5/6,
    'Terminated (Phase 1)': -1, 'Terminated (Phase 3)': -5/6, 'Terminated (Phase 2)': -4/6,
}

rating_to_status_map = {rating: status for status, rating in status_to_rating_map.items()}


def load_data(file_name):
    data = pd.read_csv(file_name)
    # Set first column to be index
    data = data.set_index(data.columns[0])
    return data

def preprocess(data: DataFrame):
    """
    Preprocess data to feature and labels according to label name

    :param data: Data to preprocess
    :param label_names: labels to split columns of data by
    :return: 4-tuple (train_set_features, train_set_labels, test_set_features, test_set_l) with
        Features and labels for trains set and features and labels for test set.
    """
    preprocessed_entries_data = _preprocess_entries(data)
    utility_matrix = _compute_utility_matrix(preprocessed_entries_data)
    training_set, test_set = split(utility_matrix)
    return training_set, test_set


def _preprocess_entries(data: DataFrame):
    preprocessed_entries = data.copy()
    made_phases_pattern = r"Phase (\d+)/Phase (\d+)"

    for item, item_row in data.iterrows():
        for user in data.columns:
            status = data.loc[item, user]
            # if pd.isna(status): # convert np.nans to Unrated values
            #     # אם הייתי רוצה תהליך הפוך בצורה מושלמת אז כל ערך שהוא nana עבור nan כל ערך שהוא nan לא הייתי נוגע בו...
            #     preprocessed_entries.at[item, user] = 'Unrated'
            # else: המר כל ערך שהוא לא nan באופן הבא:
            if pd.notna(status):
                match = re.search(made_phases_pattern, status)
                if match: # Remove / and later phase from any status containing (Phase x/Phase y)
                    x = int(match.group(1))
                    y = int(match.group(2))
                    status = re.sub(f"/", "", status)
                    status = re.sub(f"Phase {max(x, y)}", "", status)
                status = status.replace("Early ", "")

                if status in status_to_rating_map: #
                    preprocessed_entries.at[item, user] = status
                else: # Unrate any other invalid format key...
                    # כל ערך שהוא לא תקין הייתי הופך ל-nan...
                    #preprocessed_entries.at[item, user] = 'Unrated'
                    preprocessed_entries.at[item, user] = np.nan
    return preprocessed_entries

def _compute_utility_matrix(data: DataFrame):
    """
    Assume there are only valid keys...
    :param data:
    :return:
    """

    return data.applymap(lambda status: status_to_rating_map[status] if pd.notna(status) else status)

def split(utility_matrix: DataFrame , train_proportion=0.8):
    """

    :param data_features:
    :param data_labels:
    :param train_proportion:
    :return: 4-tuple
    """

    utility_matrix_flattened = utility_matrix.stack().reset_index()
    utility_matrix_flattened.columns = ['User', 'Item', 'Rating'] # אני רוצה שuser יהיה מחלה ו-item זה תרופה

    #utility_matrix_flattened = utility_matrix_flattened[utility_matrix_flattened['Rating'] != status_to_rating_map['Unrated']]
    utility_matrix_flattened = utility_matrix_flattened[pd.notna(utility_matrix_flattened['Rating'])]
    train_set = utility_matrix_flattened.sample(frac=train_proportion, random_state=42)
    test_set = utility_matrix_flattened.drop(train_set.index)
    return train_set, test_set

def _compute_drug_to_disease_matrix(utility_matrix: DataFrame):
    return utility_matrix.applymap(lambda rating: rating_to_status_map[rating] if pd.notna(rating) else rating)
    #return data.applymap(lambda status: status_to_rating_map[status])

def postprocess(predictions: DataFrame):
    predictions['Rating'] = predictions['Rating'].apply(lambda rating: _adjust_to_rating_of_status(rating)) #v

def _adjust_to_rating_of_status(rating):
    if rating >= 11/12:
        return status_to_rating_map['Approved']
    rating_of_statuses = np.array(list(status_to_rating_map.values()))
    abs_distances = np.abs(rating_of_statuses - rating)
    closest_index = abs_distances.argmin()
    return rating_of_statuses[closest_index]


def load_profiles_of_diseases():
    profiles_of_diseases = pd.read_csv('nih_disease_info.csv')
    return profiles_of_diseases

def limit_to_items_with_profile(train_set, test_set, item_encoding, item_profiles):
    limited_train_set = train_set[train_set['Item'].isin(item_profiles['DrugBank Accession Number'])]
    limited_test_set = test_set[test_set['Item'].isin(item_profiles['DrugBank Accession Number'])]
    return limited_train_set, limited_test_set

