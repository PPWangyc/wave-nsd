# Subject filtering code for stimuli_info DataFrame

# Option 1: Filter columns that contain 'subject' in their name
stimuli_info = stimuli_info[stimuli_info.columns[stimuli_info.columns.str.contains('subject', case=False)]]

# Option 2: If you want to filter rows where the ID column contains 'subject'
# stimuli_info = stimuli_info[stimuli_info['id'].str.contains('subject', case=False, na=False)]

# Option 3: If you want to filter for specific subject IDs
# SUBJECT = ['subject01', 'subject02', 'subject03']  # Replace with your subject IDs
# stimuli_info = stimuli_info[stimuli_info['id'].isin(SUBJECT)]

print(f"Filtered stimuli_info shape: {stimuli_info.shape}")
print(f"Columns: {list(stimuli_info.columns)}") 