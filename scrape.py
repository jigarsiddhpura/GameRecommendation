from serpapi import GoogleSearch
import json
from dotenv import load_dotenv
import os, csv, re
load_dotenv()

def preprocess_text(input_string):
    # Replace '\n' with spaces
    input_string = str(input_string)
    replaced_newlines = input_string.replace('\n', ' ')

    return replaced_newlines

def add_data_from_list_to_csv(file_path, data_list):
    """
    Extracts data from a list and adds it to a CSV file.

    Parameters:
    - file_path: Path to the CSV file.
    - data_list: List of dictionaries where each dictionary represents a row of data.
    """
    headers = ["title", "link", "rating", "category", "downloads", "thumbnail", "description"]

    with open(file_path, mode='a', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers)

        # Write the headers if the file is empty
        if csvfile.tell() == 0:
            writer.writeheader()

        for item in data_list:
            filtered_item = {key: preprocess_text(item[key]) for key in headers if key in item}  # Only keep keys that are in the headers
            if filtered_item:  
                writer.writerow(filtered_item)


params = {
    'api_key': os.environ.get('SERPAPI_KEY'),	    
    'engine': 'google_play_games',   
}

search = GoogleSearch(params)   # where data extraction happens on the SerpApi backend
result_dict = search.get_dict() # JSON -> Python dict

google_play_games = result_dict['organic_results']

# data = json.dumps(google_play_games, indent=2, ensure_ascii=False)

add_data_from_list_to_csv('./data/games2.csv',google_play_games[0]['items'])
add_data_from_list_to_csv('./data/games2.csv',google_play_games[1]['items'])