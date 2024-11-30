import numpy as np
import pandas as pd
import sklearn

from recommendation_of_disease_treatment_by_drug import _preprocess_entries, _compute_utility_matrix, _adjust_to_rating_of_status
from recommendation_system import RecommendationSystem
from recommendation_of_disease_treatment_by_drug import *
from analyze_profiles import *
import matplotlib.pyplot as plt
import seaborn as sns


def misclassification_error(y_pred: np.ndarray, y_true: np.ndarray, normalize: bool = True) -> float:
    """
    Calculate misclassification loss

    Parameters
    ----------
    y_true: ndarray of shape (n_samples, )
        True response values
    y_pred: ndarray of shape (n_samples, )
        Predicted response values
    normalize: bool, default = True
        Normalize by number of samples or not

    Returns
    -------
    Misclassification of given predictions
    """
    loss = np.sum(np.abs(y_true)[np.sign(y_true) != np.sign(y_pred)], axis=0)
    if normalize:
        return loss / y_true.shape[0]
    return loss



def profiles_form_ratings(data):
    """
    uses the data about drug status for disease to create rating matrix
    Args:
        data: matrix of diseases over drugs and the drugs status for the disease

    Returns: rating matrix

    """
    preprocessed_entries_data = _preprocess_entries(data)
    rating_matrix = _compute_utility_matrix(preprocessed_entries_data)
    return rating_matrix

def calculate_ratings_dist(df):
    """
    calculate the distance between two user/ items based on the rating matrix, using cosine similarity
    Args:
        df: rating matrix

    Returns: matrix that holds distances between each two user or items

    """
    df = df.fillna(0)
    df = df.sub(df.mean(axis=1), axis=0)
    dist_mat = pd.DataFrame(pairwise_distances(df, metric='cosine'), index=df.index, columns=df.index)
    return dist_mat




def evaluate_recommendation():
    data = load_data('repodb_for_visualization_medical.csv')
    train_set, test_set = preprocess(data)
    recommendation_system = RecommendationSystem()
    user_medical_profiles = load_medical_diseases_profiles()
    item_medical_profiles = load_medical_drugs_profiles()
    item_wiki_profiles, user_wiki_profiles  = load_wiki_profiles("wiki_data")
    user_rating_profiles = profiles_form_ratings(data)
    item_rating_profiles = profiles_form_ratings(data).T
    profiles_and_dist_func_dict = {'user, rating matrix': (user_rating_profiles, calculate_ratings_dist),
                                   'user, wiki profiles': (user_wiki_profiles,calculate_wiki_dist),
                                   'user, medical profiles': (user_medical_profiles, calculate_disease_dist),
                                   'item, rating matrix': (item_rating_profiles, calculate_ratings_dist),
                                   'item, medical profiles': (item_medical_profiles, calculate_drug_dist)}
    precisiosns = []
    recalls = []
    misclassification_errors = []
    models_order = []
    drug_name_to_dbank_number_map = None
    for name,profiles_and_dist_func in profiles_and_dist_func_dict.items():
        for k in [5, 10]:
            if name == 'user, medical profiles':
                train_set = train_set[train_set['User'].isin(profiles_and_dist_func[0]['ind_name'])]
                test_set = test_set[test_set['User'].isin(profiles_and_dist_func[0]['ind_name'])]
            elif name == 'user, wiki profiles':
                train_set = train_set[train_set['User'].isin(profiles_and_dist_func[0].index)]
                test_set = test_set[test_set['User'].isin(profiles_and_dist_func[0].index)]
            elif name == 'item, medical profiles':
                train_set = train_set[train_set['Item'].isin(profiles_and_dist_func[0]['DrugBank Accession Number'])]
                test_set = test_set[test_set['Item'].isin(profiles_and_dist_func[0]['DrugBank Accession Number'])]
                drug_name_to_dbank_number_map = profiles_and_dist_func[0].set_index('Generic Name')['DrugBank Accession Number'].to_dict()
            elif name == 'item, rating matrix':
                drug_name_to_dbank_number_map = {drugbank_id:  drugbank_id for drugbank_id in profiles_and_dist_func[0].index.tolist()}
            if 'user' in name:
                predictions = recommendation_system.predict(train_set, test_set, profiles_and_dist_func[0], profiles_and_dist_func[1], k)
            if 'item' in name:
                predictions = recommendation_system.predict_item_item(train_set, test_set, profiles_and_dist_func[0], profiles_and_dist_func[1], drug_name_to_dbank_number_map, k)
            predictions['Rating'] = predictions['Rating'].apply(lambda rating: _adjust_to_rating_of_status(rating))
            preds_for_recall = np.where(predictions['Rating'] == status_to_rating_map['Approved'], 1, 0)
            test_for_recall = np.where(test_set['Rating'] == status_to_rating_map['Approved'], 1, 0)
            precisiosns.append(sklearn.metrics.precision_score(test_for_recall, preds_for_recall, pos_label=1))
            recalls.append(sklearn.metrics.recall_score(test_for_recall, preds_for_recall, pos_label=1))
            misclassification_errors.append(misclassification_error(test_set['Rating'], predictions['Rating'], normalize=True))
            models_order.append(name+" k="+str(k))
    df_recall = pd.DataFrame({'Model': models_order, 'Metric': ['Recall']*len(recalls), 'Value': recalls})
    df_precision = pd.DataFrame({'Model': models_order, 'Metric': ['Precision']*len(precisiosns), 'Value': precisiosns})
    df_misclass = pd.DataFrame({'Model': models_order, 'Metric': ['Misclassification Error']*len(misclassification_errors),'Value': misclassification_errors})
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    sns.barplot(data=df_recall, x='Metric', y='Value', hue='Model', ax=axes[0])
    axes[0].set_title('Recall')
    axes[0].set_ylabel('Value')
    axes[0].set_xlabel('Model')
    sns.barplot(data=df_precision, x='Metric', y='Value', hue='Model', ax=axes[1])
    axes[1].set_title('Precision')
    axes[1].set_ylabel('Value')
    axes[1].set_xlabel('Model')
    sns.barplot(data=df_misclass, x='Model', y='Value', hue='Model', ax=axes[2])
    axes[2].set_title('Misclassification Error')
    axes[2].set_ylabel('Value')
    axes[2].set_xlabel('Model')
    for ax in axes[:2]:
        ax.get_legend().set_visible(False)
    plt.legend(title='Model', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
    plt.savefig("rec_sys_evaluation_fig.png")

if __name__ == '__main__':
    evaluate_recommendation()




