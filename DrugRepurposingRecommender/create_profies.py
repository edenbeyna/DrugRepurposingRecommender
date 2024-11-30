import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
import json
import csv
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager




drugbank_info_cols = ['drug_name','Generic Name', 'Brand Names', 'DrugBank Accession Number', 'Summary', 'Background', 'Weight',
                                    'Average Weight', 'Monoisotopic Weight', 'Chemical Formula', 'Groups', 'Indication', 'Associated Therapies',
                                    'Pharmacodynamics', 'Mechanism of action', 'Absorption', 'Volume of distribution', 'Protein binding', 'Metabolism',
                                    'Route of elimination', 'Half-life', 'Clearance', 'Toxicity', 'Pathways', 'Pharmacogenomic Effects/ADRs']


def load_datasets():
    repo_df = pd.read_csv("repodb.csv")
    drugs_info_df = pd.read_csv("drugs_side_effects_drugs_com.csv")
    drugs_info_df['drug_name'] = drugs_info_df['drug_name'].str.lower()
    return repo_df, drugs_info_df





def scrape_drugbank_page(name, url, session, out_csv='drugbank_info_full.csv'):
    print(f"Processing: {url}")
    response = session.get(url)
    if response.status_code != 200:
        print("Error, couldn't fetch medication page")
        return None
    soup = BeautifulSoup(response.text, 'html.parser')
    dt_elements = [e for e in soup.find_all('dt') if
                   e.text.strip().lower() in [col.lower() for col in drugbank_info_cols]]
    csv_row = pd.DataFrame(columns=drugbank_info_cols, index=[0])
    csv_row['drug_name'] = name
    for dt in dt_elements:
        dd = dt.find_next_sibling('dd')
        if dd:
            csv_row[dt.text] = dd.get_text(strip=True)
    if not csv_row['Summary'].isna()[0]:
        csv_row['Summary'] =  csv_row['Summary'][0].split(" ",1)[0].removesuffix('is')+ " is "+ csv_row['Summary'][0].split(" ",1)[1]
    if not csv_row['Weight'].isna()[0]:
        csv_row['Average Weight'] = float(csv_row['Weight'][0].split(':')[1].split('M')[0])
        csv_row['Monoisotopic Weight'] = float(csv_row['Weight'][0].split(':')[-1])
    csv_row.replace('Not Available', None)
    csv_row.to_csv(out_csv, mode ='a', index = False, header = False)



def scrape_for_drug_info(row, session):
    name, drugbank_link, side_effects_link = row['drug_name'], row['drugbank_link'], row['side_effects_link']
    scrape_drugbank_page(name, drugbank_link, session)


def scrape_nih(name, id, url, session, out_csv='nih_disease_info_nan.csv'):
    print(f"Processing: {url}")
    try:
        response = session.get(url)
        if response.status_code != 200:
            print(f"Error, couldn't fetch medication page {url}, status code: {response.status_code}")
            return None
    except:
        print(f"Error, couldn't fetch medication page: {url}")
        return None
    csv_row = pd.DataFrame(columns=['ind_name', 'ind_id', 'Type', 'Definition', 'Pubmed_links'], index=[0])
    csv_row['ind_name'] = name
    csv_row['ind_id'] = id
    soup = BeautifulSoup(response.text, 'html.parser')
    type = soup.find(class_="rprtid")
    if type:
        type = type.text.split("•")[-1]
        csv_row['Type'] = type
    possible_definition = soup.find(class_="portlet mgSection")
    if possible_definition:
        if possible_definition.text.strip().split("\n")[0] != 'Definition':
            possible_definition = None
        else:
            possible_definition = possible_definition.text.strip().split("\n")[1].split("[")[0]
    csv_row['Definition'] = possible_definition
    not_pubmed_links = ['Definition', 'Term Hierarchy', 'Conditions with this feature', 'Professional guidelines', 'Clinical features',
                        'Recent clinical studies', 'Recent systematic reviews', 'Table of contents', 'Practice guidelines',
                        'Clinical resources', 'Consumer resources', 'Reviews', 'Related information', 'Recent activity', 'Genetic Testing Registry']
    pubmed_links = [c.text for c in soup.find_all(class_='nl') if c.text not in not_pubmed_links]
    csv_row['Pubmed_links'] = " ".join(pubmed_links)
    csv_row.to_csv(out_csv, mode='a', index=False, header=False)

def scrape_for_disease_info(row, session):
    name, id,  nih_link = row['ind_name'], row['ind_id'] ,row['disease_link']
    scrape_nih(name, id, nih_link, session)



if __name__ == '__main__':
    repo_df, drugs_info_df = load_datasets()
    repo_df['drugbank_link'] = "https://go.drugbank.com/drugs/"+repo_df['drugbank_id'].values
    repo_df['side_effects_link'] = "https://www.drugs.com/sfx/" + repo_df['drug_name']+"-side-effects.html"
    repo_df['disease_link'] = "https://www.ncbi.nlm.nih.gov/medgen/?term=" + repo_df['ind_id']
    # scrape web for drug information
    session = requests.Session()
    repo_df.drop_duplicates(subset=['drug_name', 'drugbank_id']).apply(scrape_for_drug_info, session=session,axis=1)
    # scrape web for indication information
    session = requests.Session()
    repo_df.drop_duplicates(subset=['ind_name', 'ind_id']).apply(scrape_for_disease_info, session=session, axis=1)
