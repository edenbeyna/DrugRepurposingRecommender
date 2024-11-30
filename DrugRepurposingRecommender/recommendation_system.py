import numpy as np
import pandas as pd
from pandas import DataFrame, Series

class RecommendationSystem:


    def _rate(self, user, item, similarities_to_user: DataFrame, train_matrix: DataFrame, N):
        """
        Rate item for user...
        (using user-user...)
        :param test_set: DataFrame used to set rating in cell of item and user.
            Assume has format of 'User', 'Item', 'Rating
        :param user: User for which item is rated
        :param item: Item which is rated for user
        :param N: set of closest items to item
        :return: float - rating of item
        """
        if item in train_matrix.columns:
            similarities_of_N = similarities_to_user.loc[user, N].to_numpy()
            ratings_of_N = train_matrix.loc[N, item].to_numpy()
            sum_of_similarites = np.sum(similarities_of_N)

            if sum_of_similarites == 0:
                rating = np.nan
            else:
                dot_product = np.dot(similarities_of_N, ratings_of_N)
                rating = np.divide(dot_product, sum_of_similarites)
            return rating
        else:
            return 0



    def load_users(self):
        users = pd.read_csv('nih_disease_info.csv')
        return users

    def _compute_k_closest_indices(self, series: Series, k=10):
        bottom_k = series.nsmallest(min(k, len(series)))
        closest_users = bottom_k.index
        return closest_users



    def _sample_k_most_similar_users(self, training_matrix: DataFrame, user, item, similarities_of_users, k=10):
        if item in training_matrix.columns:
            users_by_item = training_matrix[item]
            users_rating_item = users_by_item[users_by_item.notna()].index
            similarities_to_user = similarities_of_users.loc[user]
            similarities_to_user_by_item = similarities_to_user[users_rating_item]
            closest_items = self._compute_k_closest_indices(similarities_to_user_by_item, k)
            return closest_items
        else:
            return pd.Index([])

    def predict(self, train_set: DataFrame, test_set: DataFrame, user_profiles: DataFrame, distance_func, k=10):
        predictions = test_set.copy() # TODO: אולי עדיף לקרוא לזה predictions ולא recommendations

        similarities_between_users = distance_func(user_profiles)
        similarities_between_users = similarities_between_users[similarities_between_users.index.isin(test_set['User'])] # All rows of distances will be diseases of test set... # check all rows of similarities between users then check rows of users to see similarities of test set # עבור דמיון עבור מחלה עבור שורה  עבור train_set עבור עמודה. אז עובר דמיון יש שם מחלה שמתאים לה עבור מוצר עבור tarin_set עבור מוצר אז עבור train_set עבור משתמש אז עבור train_set עבור עמודה עבור משתמש אז עבור train set יש עמודה משלו יש משתמש משלו בעמודה משלו ב-train_set אז עבור שם מוצר יש שם עמודה שמתאים לה
        similarities_between_users = similarities_between_users[train_set['User'].unique()]
        train_matrix = train_set.pivot(index='User', columns='Item', values='Rating')

        for row_index, row in test_set.iterrows():
            user = row['User']
            item = row['Item']
            N = self._sample_k_most_similar_users(train_matrix, user, item, similarities_between_users, k)
            rating = self._rate(user, item, similarities_between_users, train_matrix, N)
            predictions.at[row_index, 'Rating'] = rating

        return predictions



    def predict_item_item(self, train_set: DataFrame, test_set: DataFrame, item_profiles: DataFrame, distance_func,
                          mapping_to_item, k=10):
        predictions = test_set.copy()  # TODO: אולי עדיף לקרוא לזה predictions ולא recommendations
        similarities_between_items = distance_func(item_profiles)
        similarities_between_items = similarities_between_items.rename(index=mapping_to_item)
        similarities_between_items = similarities_between_items[similarities_between_items.index.isin(test_set[
                                                                                                                         'Item'])]  # All rows of distances will be diseases of test set... # check all rows of similarities between users then check rows of users to see similarities of test set # עבור דמיון עבור מחלה עבור שורה  עבור train_set עבור עמודה. אז עובר דמיון יש שם מחלה שמתאים לה עבור מוצר עבור tarin_set עבור מוצר אז עבור train_set עבור משתמש אז עבור train_set עבור עמודה עבור משתמש אז עבור train set יש עמודה משלו יש משתמש משלו בעמודה משלו ב-train_set אז עבור שם מוצר יש שם עמודה שמתאים לה
        similarities_between_items = similarities_between_items.rename(columns=mapping_to_item)
        similarities_between_items = similarities_between_items[train_set['Item'].unique()]
        train_matrix = train_set.pivot(index='User', columns='Item', values='Rating')

        for row_index, row in test_set.iterrows():
            user = row['User']
            item = row['Item']
            N = self._sample_k_most_similar_users(train_matrix.T, item, user, similarities_between_items, k)
            rating = self._rate(item, user, similarities_between_items, train_matrix.T, N)
            predictions.at[row_index, 'Rating'] = rating

        return predictions

