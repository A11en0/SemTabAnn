import pandas as pd
import re
import multiprocessing

# Read the ground truth files for SOTABv2
cta_train_gt = pd.read_csv('/data2/yzy/sotabv2/sotab_v2_cta_training_set.csv')
cta_val_gt = pd.read_csv('/data2/yzy/sotabv2/sotab_v2_cta_validation_set.csv')
cta_test_gt = pd.read_csv('/data2/yzy/sotabv2/sotab_v2_cta_test_set.csv')

gt = {'train':{}, 'val':{}, 'test':{}}
for index, row in cta_train_gt.iterrows():
    if row['table_name'] not in gt['train']:
        gt['train'][row['table_name']] = {}
        
    gt['train'][row['table_name']][row['column_index']] = row['label']
val = {}
for index, row in cta_val_gt.iterrows():
    if row['table_name'] not in gt['val']:
        gt['val'][row['table_name']] = {} 
    gt['val'][row['table_name']][row['column_index']] = row['label']
test = {}
for index, row in cta_test_gt.iterrows():
    if row['table_name'] not in gt['test']:
        gt['test'][row['table_name']] = {}
    gt['test'][row['table_name']][row['column_index']] = row['label']

cta_train_cols = (cta_train_gt['table_name'] + '|' + cta_train_gt['column_index'].map(str) + '|' + cta_train_gt['label']).tolist()
cta_val_cols = (cta_val_gt['table_name'] + '|' + cta_val_gt['column_index'].map(str) + '|' + cta_val_gt['label']).tolist()
cta_test_cols = (cta_test_gt['table_name'] + '|' + cta_test_gt['column_index'].map(str) + '|' + cta_test_gt['label']).tolist()

type_labels = list(cta_val_gt['label'].unique())
print(len(type_labels))

import os

data_path = '/data2/yzy/code/cta/effective-fiesta/data/sotab_v2'
os.makedirs(data_path, exist_ok=True)  # ✅ 自动创建目录（如果已存在则跳过）

file_path = os.path.join(data_path, 'type_ontology.txt')
with open(file_path, 'w') as f:
    for label in type_labels:
        f.write(label + '\n')

print(f"文件已保存到: {file_path}")

#Simple Preprocessing

def clean_text(text):        
    if(isinstance(text, dict)):
        text = ' '.join([ clean_text(v) for k, v in text.items()] )
    elif(isinstance(text, list)):
        text = map(clean_text, text)
        text = ' '.join(text)
        
    if pd.isnull(text):
        return ''
        
    #Remove excess whitespaces
    text = re.sub(' +', ' ', str(text)).strip()
    
    return text


# Prepare format of input datasets for Doduo models: table_id, [labels], data, label_ids
def get_table_column(column):
    file_name, column_index, label = column.split('|')

    if file_name in cta_train_gt['table_name'].tolist():
        path = '/data2/yzy/sotabv2/Train/'+file_name # Path for train tables
    elif file_name in cta_val_gt['table_name'].tolist():
        path = '/data2/yzy/sotabv2/Validation/'+file_name # Path for validation tables
    else:
        path = '/data2/yzy/sotabv2/Test/'+file_name # Path for test tables

    df = pd.read_json(path, compression='gzip', lines=True)

    y = [0] * len(type_labels)
    y[type_labels.index(label)] = 1

    return [
        file_name, #table_id
        [label], #[labels]
        clean_text(df.iloc[:, int(column_index)].tolist()), #data
        y, #label_ids
        column_index
    ]


pool = multiprocessing.Pool(processes=20)
train_result = pool.map(get_table_column, cta_train_cols)
val_result = pool.map(get_table_column, cta_val_cols)
test_result = pool.map(get_table_column, cta_test_cols)
pool.close()
pool.join()

cta = {}
cta['train'] = pd.DataFrame(train_result, columns=['table_id', 'labels', 'data', 'label_ids','column_index'])
cta['dev'] = pd.DataFrame(val_result, columns=['table_id', 'labels', 'data', 'label_ids','column_index'])
cta['test'] = pd.DataFrame(test_result, columns=['table_id', 'labels', 'data', 'label_ids','column_index'])


import pickle
file_name='table_col_type_serialized.pkl'
f = open(data_path+file_name,'wb')
pickle.dump(cta,f)
f.close()

