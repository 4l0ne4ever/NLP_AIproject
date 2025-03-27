import spacy
import pandas as pd
import nltk
from nltk.tokenize import sent_tokenize
from ast import literal_eval
import os
import sys
import json
import pathlib

folder_path = pathlib.Path().parent.resolve()
sys.path.append(str(os.path.join(folder_path, '../')))
from utils import load_subtitles_dataset

class NamedEntityRecognizer:
    def __init__(self, character_list_path='/Users/duongcongthuyet/Downloads/workspace/AI /project/data/list_characters.jsonl'):
        self.nlp_model = self.load_model()
        self.character_set = self.load_character_names(character_list_path)
        # Removed unnecessary 'pass' statement

    def load_model(self):
        nlp = spacy.load("en_core_web_trf")
        return nlp
    
    def load_character_names(self, filepath):
        character_names = set()
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line.strip())
                    name = data.get('character_name', '').strip()
                    if name:
                        # Add both full name and first name to the set
                        character_names.add(name)
                        first_name = name.split()[0]
                        character_names.add(first_name)
        except FileNotFoundError:
            print(f"Error: Character list file '{filepath}' not found.")
            return set()
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON in '{filepath}': {e}")
            return set()
        return character_names
    
    def get_ners_inference(self, script):
        script_sentences = sent_tokenize(script)
        ner_output = []
        for sentence in script_sentences:
            doc = self.nlp_model(sentence)
            ners = set()
            for entity in doc.ents:
                if entity.label_ == "PERSON":
                    full_name = entity.text.strip()
                    first_name = full_name.split()[0].strip()
                    if full_name in self.character_set or first_name in self.character_set:
                        ners.add(first_name)
            ner_output.append(ners)
        return ner_output
    
    def get_ners(self, dataset_path, save_path=None):
        if save_path is not None and os.path.exists(save_path):
            df = pd.read_csv(save_path)
            df["ners"] = df["ners"].apply(lambda x: literal_eval(x) if isinstance(x, str) else x)
            return df
            
        # Load the dataset
        df = load_subtitles_dataset(dataset_path)
        # Run inference 
        df["ners"] = df["script"].apply(self.get_ners_inference)
        if save_path is not None:
            df.to_csv(save_path, index=False)
        return df