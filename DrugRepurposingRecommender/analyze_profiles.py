import json
import os
import pandas as pd
import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import pairwise_distances
from seaborn import clustermap, heatmap
from collections import defaultdict
import re


##### The following functions reads the profiles created
def load_wiki_profiles(directory_path):
    """
        loads the profiles that were built based on wikipedia
        :return: pandas df of the drugs profiles, pandas df of the diseases profiles
        """
    drug_profiles = {}
    disease_profiles = {}
    for filename in os.listdir(directory_path):
        if filename.endswith('.json'):
            file_path = os.path.join(directory_path, filename)
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                entity_name = data['name']
                content = data.get('content', '')
                category = data.get('category')

                if category == 'drug':
                    drug_profiles[entity_name] = content
                elif category == 'disease':
                    disease_profiles[entity_name] = content

    drugs_df = pd.DataFrame({'profile': drug_profiles.values()}, index=drug_profiles.keys())
    diseases_df = pd.DataFrame({'profile': disease_profiles.values()}, index=disease_profiles.keys())
    return drugs_df, diseases_df


def load_medical_diseases_profiles():
    """
    loads the diseases profiles that were built based on medical info
    :return: pandas df of the profiles
    """
    disease_profiles = pd.read_csv("nih_disease_info.csv")
    disease_profiles = disease_profiles[
        disease_profiles.Definition.notna() | disease_profiles.Pubmed_links.notna() | disease_profiles.Type.notna()].reset_index()
    return disease_profiles


def load_medical_drugs_profiles():
    """
        loads the drugs profiles that were built based on medical info
        :return: pandas df of the profiles
        """
    drugs_profiles = pd.read_csv("drugbank_info_for_df.csv").drop_duplicates()
    drugs_profiles.replace('Not Available', np.nan, inplace=True)
    drugs_profiles = drugs_profiles[
        drugs_profiles['Summary'].notna() | drugs_profiles['Background'].notna()].reset_index()
    return drugs_profiles


### The following functions calculate distance between profiles ###
def run_tfidf(df):
    """
        Computes the TF-IDF matrix for the 'profile' column in a DataFrame.

        Parameters:
            df (pd.DataFrame): DataFrame containing a 'profile' column with text data.

        Returns:
            pd.DataFrame: A DataFrame representing the TF-IDF matrix.
        """
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['profile'])
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=vectorizer.get_feature_names_out())
    tfidf_df.index = df.index
    return tfidf_df


def calculate_wiki_dist(df):
    """
    calculates the distance between each two wikipedia based profiles
    :param df: profiles df to calculate distance on
    :return: matrix with distance between wikipedia based profiles
    """
    tfidf_mat = run_tfidf(df)
    tfidf_dist_mat = pd.DataFrame(euclidean_distances(tfidf_mat), index=tfidf_mat.index, columns=tfidf_mat.index)
    return tfidf_dist_mat


def calculate_disease_dist(df, w_tfidf=0.75, w_type=0.25):
    """
        Calculates a combined distance metric for diseases based on TF-IDF and type dummy variables.

        Parameters:
            df (pd.DataFrame): DataFrame containing disease profiles and types.
            w_tfidf (float): Weight for the TF-IDF distance component.

            w_type (float): Weight for the type distance component.

        Returns:
            pd.DataFrame: A DataFrame representing the combined distance matrix.
        """
    concatenated_df = pd.DataFrame({'index': df['ind_name'],
                                    'profile': df['Definition'].fillna('').str.cat(
                                        df['Pubmed_links'].fillna(''), sep=' ')}).set_index('index')
    tfidf_mat = run_tfidf(concatenated_df)
    tfidf_dist_mat = pd.DataFrame(euclidean_distances(tfidf_mat), index=tfidf_mat.index, columns=tfidf_mat.index)
    type_df = df['Type'].str.get_dummies(sep=';')
    type_dist = pd.DataFrame(euclidean_distances(type_df), index=tfidf_mat.index,
                             columns=tfidf_mat.index)
    return w_tfidf * tfidf_dist_mat + w_type * type_dist


def extract_numeric(s):
    """
        Extracts numeric values from a specific formatted string.

        Parameters:
            s (str): A string containing numeric values in a predefined format.

        Returns:
            tuple: Extracted average and monoisotopic weight as strings, or (0, 0) on failure.
        """
    try:
        avg_match = re.search(r'(\d+\.\d+)', s.split(':')[1])
        mono_match = re.search(r'(\d+\.\d+)', s.split(':')[2])
        return avg_match.group(1) if avg_match else None, \
            mono_match.group(1) if mono_match else None
    except Exception:
        return 0, 0


def calculate_drug_dist(df, w_tfidf=0.72, w_weight=0.28):
    """
        Calculates a combined distance metric for drugs based on TF-IDF and weight attributes.

        Parameters:
            df (pd.DataFrame): DataFrame containing drug profiles.
            w_tfidf (float): Weight for the TF-IDF distance component.
            w_weight (float): Weight for the weight distance component.

        Returns:
            pd.DataFrame: A DataFrame representing the combined distance matrix.
        """
    concatenated_df = pd.DataFrame({'index': df['Generic Name'],
                                    'profile': df['Background'].fillna('').str.cat(
                                        df['Summary'].fillna(''), sep=' ')}).set_index('index')
    df[['Average Weight', 'Monoisotopic Weight']] = df['Weight'].apply(extract_numeric).apply(
        pd.Series)
    df['Average Weight'] = df['Average Weight'].astype(float)
    df['Monoisotopic Weight'] = df['Monoisotopic Weight'].astype(float)
    df['Average Weight'] /= 1000
    df['Monoisotopic Weight'] /= 1000
    tfidf_mat = run_tfidf(concatenated_df)
    tfidf_dist_mat = pd.DataFrame(euclidean_distances(tfidf_mat), index=tfidf_mat.index, columns=tfidf_mat.index)
    weight_dist = pd.DataFrame(pairwise_distances((df['Average Weight']).values.reshape(-1, 1), metric='cityblock'),
                               index=tfidf_mat.index, columns=tfidf_mat.index)
    return w_tfidf * tfidf_dist_mat + w_weight * weight_dist


### create clustered heatmaps based on the distances calculated ###

def create_heatmap(df, name, dist_callable):
    """
        Creates a clustered heatmap from a distance matrix derived from the input DataFrame.

        Parameters:
            df (pd.DataFrame): DataFrame containing the data to be clustered.
            name (str): Column name used for labeling in the heatmap.
            dist_callable (function): A callable function that computes distances.

        Returns:
            tuple: The clustered heatmap object and a list of new order of indices.
        """
    dist_mat = dist_callable(df)
    clustered_heatmap = clustermap(dist_mat, figsize=(10, 10), yticklabels=False, xticklabels=False)
    if name in df.columns:
        new_order = [df[name][i] for i in clustered_heatmap.dendrogram_row.reordered_ind]
    else:
        new_order = [df.index[i] for i in clustered_heatmap.dendrogram_row.reordered_ind]
    print(new_order)
    return clustered_heatmap, new_order


def create_diseases_heatmap(name, profile_type, disease_profiles, dist_callable):
    """
        Creates a heatmap for diseases by loading disease profiles from a CSV file,
        filtering them, and computing distances.

        Parameters:
            name (str): Column name used for labeling in the heatmap.
            profile_type (str): Type of profile to be indicated in the output filename.

        Returns:
            tuple: The heatmap object and a new order of indices.
        """
    dis_heatmap, new_order = create_heatmap(disease_profiles, name, dist_callable)
    # dis_heatmap.figure.suptitle(f"Diseases over Diseases, {profile_type} profiles", y=0.9, fontsize=16)
    # dis_heatmap.ax_heatmap.set_ylabel("Diseases")
    # dis_heatmap.ax_heatmap.set_xlabel("Disease")
    dis_heatmap.figure.savefig(f"disease_heatmap_{profile_type}.png")
    return dis_heatmap, new_order


def create_drugs_heatmap(name, profile_type, drugs_profiles, dist_callable):
    """
        Creates a heatmap for drugs by loading drug profiles from a CSV file,
        processing weights, and computing distances.

        Parameters:
            name (str): Column name used for labeling in the heatmap.
            profile_type (str): Type of profile to be indicated in the output filename.

        Returns:
            tuple: The heatmap object and a new order of indices.
        """
    drugs_heatmap, drugs_order = create_heatmap(drugs_profiles, name, dist_callable)
    drugs_heatmap.figure.savefig(f"drugs_heatmap_{profile_type}.png")
    return drugs_heatmap, drugs_order


### Functions that reorders the repodb.csv to new format for visualization ###

def create_pivoted_dataframe(csv_file, col_names):
    """
        Creates a pivoted DataFrame from a CSV file containing drug data.

        Parameters:
            csv_file (str): Path to the CSV file with drug information.

        Returns:
            pd.DataFrame: A pivoted DataFrame indexed by drugbank_id with drug status and phase.
        """
    df = pd.read_csv(csv_file)
    drug_data = defaultdict(lambda: defaultdict(str))
    for _, row in df.iterrows():
        drug = row[col_names]
        ind_name = row['ind_name']
        status = row['status']
        phase = row['phase']
        if pd.notna(status) and pd.notna(phase):
            combined_status = f"{status} ({phase})"
        elif pd.notna(status):
            combined_status = status
        else:
            combined_status = phase
        drug_data[drug][ind_name] = combined_status
    pivoted_df = pd.DataFrame(drug_data)
    return pivoted_df


def reorder_df(df, disease_order, drug_order):
    """
        Reorders a DataFrame based on specified orders for diseases and drugs.

        Parameters:
            df (pd.DataFrame): The DataFrame to be reordered.
            disease_order (list): The new order of disease indices.
            drug_order (list): The new order of drug columns.

        Returns:
            pd.DataFrame: The reordered DataFrame.
        """
    df_reordered = df.reindex(index=disease_order)
    df_reordered = df_reordered[drug_order]
    return df_reordered

### Analyzes the profiles and creates visualizations ###

def analyze_profiles(diseases_profiles, drugs_profiles, repodb_df, profile_type, diseases_dist_callable,
                     drugs_dist_callable):
    """
        Main function to analyze medical profiles by creating disease and drug heatmaps,
        pivoting data, and saving the result for visualization.

        It orchestrates the overall analysis flow and saves a combined CSV file for final visualization.
        """

    diseases_heatmap, disease_order = create_diseases_heatmap('ind_name', profile_type, diseases_profiles,
                                                              diseases_dist_callable)
    drugs_heatmap, drugs_order = create_drugs_heatmap('DrugBank Accession Number', profile_type, drugs_profiles,
                                                      drugs_dist_callable)
    disease_order = disease_order + [i for i in repodb_df.index.values if i not in disease_order]
    drugs_order = drugs_order + [i for i in repodb_df.columns.values if i not in drugs_order]
    # saves the df to csv, later the final visualization is done using R
    reorder_df(repodb_df, disease_order, drugs_order).to_csv(f"repodb_for_visualization_{profile_type}.csv")
    return diseases_heatmap, drugs_heatmap


def analyze_medical_profiles():
    """
    analyzes the medical based profiles
    :return: clustered heatmaps of diseases and drugs. based on medical profiles
    """
    diseases_profiles = load_medical_diseases_profiles()
    drugs_profiles = load_medical_drugs_profiles()
    repodb_df = create_pivoted_dataframe("repodb.csv", "drugbank_id")
    return analyze_profiles(diseases_profiles, drugs_profiles, repodb_df, 'medical', calculate_disease_dist,
                            calculate_drug_dist)


def analyze_wiki_profiles():
    """
        analyzes the wikipedia based profiles
        :return: clustered heatmaps of diseases and drugs. based on wikipedia profiles
        """
    drugs_profiles, diseases_profiles = load_wiki_profiles('wiki_data')
    repodb_df = create_pivoted_dataframe("repodb.csv", "drug_name")
    return analyze_profiles(diseases_profiles, drugs_profiles, repodb_df, 'wikipedia', calculate_wiki_dist,
                            calculate_wiki_dist)




if __name__ == '__main__':
    analyze_wiki_profiles()
    analyze_medical_profiles()
