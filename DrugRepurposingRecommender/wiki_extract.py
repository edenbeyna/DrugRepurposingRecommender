import re
import os
import json
import pandas as pd
import wikipedia

def get_wikipedia_text(entity_name):

    try:
        page_content = wikipedia.page(entity_name).content
        cleaned_content = page_content.replace('\n', ' ').strip()
        return cleaned_content
    except wikipedia.exceptions.PageError:
        print(f"Page not found for {entity_name}")
        return None
    except wikipedia.exceptions.DisambiguationError as e:
        print(f"Disambiguation error for {entity_name}: {e.options}")
        return None

def clean_filename(entity_name):
    cleaned_name = re.sub(r'[\\/*?:"<>|]', '_', entity_name)
    return cleaned_name[:100]

def save_to_json(entity_name, category, content):
    profile_data = {
        "name": entity_name,
        "category": category,
        "content": content
    }
    cleaned_name = clean_filename(entity_name)
    file_path = os.path.join(directory_path, f"{cleaned_name}.json")
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(profile_data, f)


if __name__ == '__main__':
    directory_path = 'wiki_data'
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    data = pd.read_csv('repodb.csv', na_values=['NA'])
    drug_names = data['drug_name'].dropna().unique()
    disease_names = data['ind_name'].dropna().unique()
    for drug in drug_names:
        content = get_wikipedia_text(drug)
        if content:
            save_to_json(drug, 'drug', content)
    for disease in disease_names:
        content = get_wikipedia_text(disease)
        if content:
            save_to_json(disease, 'disease', content)


