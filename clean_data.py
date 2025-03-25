import json
import re
import os
import pandas as pd

# Path files
input_file = '/Users/duongcongthuyet/Downloads/workspace/AI /project/data/locations.jsonl'
output_file = '/Users/duongcongthuyet/Downloads/workspace/AI /project/data/locations_cleaned.jsonl'

# Check if input file exists
if not os.path.exists(input_file):
    print(f"Error '{input_file}'")
    print(f"Path file: {os.getcwd()}")
    exit(1)

# Read and clean data
cleaned_data = []
line_number = 0
with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        line_number += 1
        stripped_line = line.strip()
        
        # Skip empty lines
        if not stripped_line:
            print(f"Warn: {line_number} empty")
            continue
        
        # JSON decode
        try:
            item = json.loads(stripped_line)
        except json.JSONDecodeError as e:
            continue
        
        name = item.get('location_name', '').strip()
        desc = item.get('location_description', '').strip()
        
        # Skip if name or description is empty
        if not name or not desc:
            continue
        
        # Clean description
        desc = re.sub(r'This article is a stub.*?information!|Spoiler Warning!.*?(discretion|here)|"We acquire more knowledge.*?added\.|Behind the scenes.*?References.*$', 
                     '', desc, flags=re.DOTALL)
        desc = re.sub(r'\[.*?\]|\n|\s{2,}', ' ', desc).strip()
        
        cleaned_data.append({
            'location_name': name,
            'location_description': desc
        })

# Convert to DataFrame
df = pd.DataFrame(cleaned_data)

# Labeling
action_keywords = ['battle', 'fight', 'experiment']  
safe_keywords = ['safe', 'home', 'school', 'plan'] 
danger_keywords = ['monster', 'creatures', 'beast', 'predator', 'hunt', 'attack', 'kill', 
                   'threat', 'dark', 'shadow', 'fog', 'dimension', 'curse', 'vines', 'Upside Down']  # Fixed typo

# Labeling function
def classify_location(description):
    if not isinstance(description, str):
        return 'Minor Location'
    
    # Preprocess description: lowercase and strip whitespace
    desc = description.lower().strip()
    
    # Count whole-word matches for each category using regex
    action_count = sum(len(re.findall(r'\b' + re.escape(keyword) + r'\b', desc)) 
                       for keyword in action_keywords)
    safe_count = sum(len(re.findall(r'\b' + re.escape(keyword) + r'\b', desc)) 
                     for keyword in safe_keywords)
    danger_count = sum(len(re.findall(r'\b' + re.escape(keyword) + r'\b', desc)) 
                       for keyword in danger_keywords)
    
    # If no keywords are found, return 'Minor Location'
    if action_count == 0 and safe_count == 0 and danger_count == 0:
        return 'Minor Location'
    
    # Determine the maximum count
    max_count = max(action_count, safe_count, danger_count)
    
    # Resolve ties with priority: Danger Zone > Action Hub > Safe Haven
    if action_count > 0:
        return 'Action Hub' 
    elif danger_count == max_count:
        return 'Danger Zone'
    elif safe_count == max_count:
        return 'Safe Haven'
    
    # Fallback (should not occur due to max_count check)
    return 'Minor Location'

# Apply the classification
df['location_type'] = df['location_description'].apply(classify_location)

# Save to file
with open(output_file, 'w', encoding='utf-8') as f:
    for _, row in df.iterrows():
        f.write(json.dumps(row.to_dict()) + '\n')

# Print result
print(f"Clean {len(df)} and save to '{output_file}'.")
print(df)