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
    # For google colab
    CHARACTER_FILE_PATH = "/content/list_characters.jsonl" 
    # For local
    # CHARACTER_FILE_PATH = "/Users/duongcongthuyet/Downloads/workspace/AI /project/data/list_characters.jsonl"

    def __init__(self):
        self.nlp_model = self.load_model()
        self.character_set = self.load_character_names(self.CHARACTER_FILE_PATH)

    def load_model(self):
        try:
            nlp = spacy.load("en_core_web_trf")
            return nlp
        except Exception as e:
            raise Exception(f"Error loading spaCy model: {str(e)}")
    
    def load_character_names(self, filepath):
        character_names = set()
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    data = json.loads(line.strip())
                    name = data.get('character_name', '').strip()
                    if name:
                        character_names.add(name)
                        first_name = name.split()[0]
                        character_names.add(first_name)
            print(f"Loaded {len(character_names)} character names from {filepath}")
        except FileNotFoundError:
            print(f"Error: Character list file '{filepath}' not found.")
            return set()
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON in '{filepath}': {e}")
            return set()
        return character_names
    
    def get_ners_inference(self, script):
        try:
            script_sentences = sent_tokenize(script)
            ner_output = []
            for sentence in script_sentences:
                doc = self.nlp_model(sentence)
                ners = set()
                for entity in doc.ents:
                    if entity.label_ == "PERSON":
                        full_name = entity.text
                        first_name = full_name.split(" ")[0].strip()
                        if full_name in self.character_set or first_name in self.character_set:
                            ners.add(first_name)
                ner_output.append(ners)
            return ner_output
        except Exception as e:
            raise Exception(f"Error in NER inference: {str(e)}")
    
    def get_ners(self, dataset_path, save_path=None):
        try:
            if save_path is not None and os.path.exists(save_path):
                df = pd.read_csv(save_path)
                df["ners"] = df["ners"].apply(lambda x: literal_eval(x) if isinstance(x, str) else x)
                return df
            
            df = load_subtitles_dataset(dataset_path)
            df["ners"] = df["script"].apply(self.get_ners_inference)
            if save_path is not None:
                df.to_csv(save_path, index=False)
            return df
        except Exception as e:
            raise Exception(f"Error processing dataset: {str(e)}")