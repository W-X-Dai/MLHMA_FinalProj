import shutil
import pandas as pd
from pathlib import Path

src_root = Path('./DataSet') 
dst_root = Path('./data')
mapping_list = []

subjects_found = []
for mat_file in src_root.rglob('result.mat'):
    patient_folder = mat_file.parent.parent.parent
    date_folder = patient_folder.parent
    
    subject_key = (date_folder.name, patient_folder.name)
    if subject_key not in subjects_found:
        subjects_found.append(subject_key)

subjects_found.sort()

subject_map = {key: f"Patient{i+1:02d}" for i, key in enumerate(subjects_found)}

dst_root.mkdir(parents=True, exist_ok=True)

for mat_file in src_root.rglob('result.mat'):
    trial_folder_name = mat_file.parent.name
    old_patient_name = mat_file.parent.parent.parent.name
    old_date_name = mat_file.parent.parent.parent.parent.name
    
    new_patient_id = subject_map[(old_date_name, old_patient_name)]
    
    target_subject_dir = dst_root / new_patient_id
    target_subject_dir.mkdir(exist_ok=True)
    
    existing_trials = len(list(target_subject_dir.glob('*.mat')))
    new_trial_name = f"trial{existing_trials + 1:02d}.mat"
    
    target_path = target_subject_dir / new_trial_name
    
    mapping_list.append({
        'New_ID': new_patient_id,
        'New_Trial': new_trial_name,
        'Original_Date': old_date_name,
        'Original_Patient_Alias': old_patient_name,
        'Original_Trial': trial_folder_name
    })
    
    shutil.copy2(mat_file, target_path)

df_mapping = pd.DataFrame(mapping_list).sort_values(['New_ID', 'New_Trial'])
df_mapping.to_csv(dst_root / 'mapping.csv', index=False, encoding='utf-8-sig')