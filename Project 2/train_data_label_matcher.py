import os
import pandas as pd

pictures = '/home/ubuntu/Desktop-Sync-Folder/Exam2/train/pics'
labels = '/home/ubuntu/Desktop-Sync-Folder/Exam2/train/labels'

# ---------------------------------------------- Create CSV file for processing ----------------------------------------
# Creating an empty Dataframe with column names only
data_list = pd.DataFrame(columns=['Picture_Path', 'Labels', 'red_blood_cell', 'difficult', 'gametocyte', 'trophozoite',
                     'ring', 'schizont', 'leukocyte'])

# Process Cell Pictures
picture_list = os.listdir(pictures)
picture_list.sort()

for pic in picture_list:

    path = os.path.join(pictures, pic)
    path = os.path.abspath(path)

    data_list = data_list.append({'Picture_Path': path}, ignore_index=True)

# Process Labels
labels_list = os.listdir(labels)
labels_list.sort()

label_index = 0
for text_file in labels_list:
    path = os.path.join(labels, text_file)

    cell_text = open(path)
    cell_label_string = cell_text.readlines()

    label_sub_1 = 0
    label_sub_2 = 0
    label_sub_3 = 0
    label_sub_4 = 0
    label_sub_5 = 0
    label_sub_6 = 0
    label_sub_7 = 0

    # Assign label value per content
    if 'red blood cell' in cell_label_string:
        label_sub_1 = 1
    if 'red blood cell\n' in cell_label_string:
        label_sub_1 = 1

    if 'difficult' in cell_label_string:
        label_sub_2 = 1
    if 'difficult\n' in cell_label_string:
        label_sub_2 = 1

    if 'gametocyte' in cell_label_string:
        label_sub_3 = 1
    if 'gametocyte\n' in cell_label_string:
        label_sub_3 = 1

    if 'trophozoite' in cell_label_string:
        label_sub_4 = 1
    if 'trophozoite\n' in cell_label_string:
        label_sub_4 = 1

    if 'ring' in cell_label_string:
        label_sub_5 = 1
    if 'ring\n' in cell_label_string:
        label_sub_5 = 1

    if 'schizont' in cell_label_string:
        label_sub_6 = 1
    if 'schizont\n' in cell_label_string:
        label_sub_6 = 1

    if 'leukocyte' in cell_label_string:
        label_sub_7 = 1
    if 'leukocyte\n' in cell_label_string:
        label_sub_7 = 1

    data_list.at[label_index, 'Labels'] = cell_label_string
    data_list.at[label_index, 'red_blood_cell'] = label_sub_1
    data_list.at[label_index, 'difficult'] = label_sub_2
    data_list.at[label_index, 'gametocyte'] = label_sub_3
    data_list.at[label_index, 'trophozoite'] = label_sub_4
    data_list.at[label_index, 'ring'] = label_sub_5
    data_list.at[label_index, 'schizont'] = label_sub_6
    data_list.at[label_index, 'leukocyte'] = label_sub_7

    label_index += 1

data_list.to_csv('/home/ubuntu/Desktop-Sync-Folder/Exam2/train_data_with_labels.csv', index=False)
