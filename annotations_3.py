# Import libraries necessary for this project
import numpy as np
import pandas as pd
import json

# Import visualisation libraries
import seaborn as sns
import matplotlib.pyplot as plt
import skimage.io as io
from pycocotools.coco import COCO

#  creating data frame from json annotations

def create_df_from_json(fname_json):
    """
    creating data frame from json annotations

    """
    with open(fname_json,'r') as f:
        data = json.loads(f.read())

    # Flatten table to a data frame respect to the categories, annotations, and images
    df_categories = pd.json_normalize(data, record_path =['categories'])
    df_annotations = pd.json_normalize(data, record_path =['annotations'])
    df_images = pd.json_normalize(data, record_path =['images'])
    # joining all the data in one data set

    # In data frame df_images: change the column name 'id' to 'image_id'.
    df_images.rename(columns={'id':'image_id'}, inplace=True)

    # Left join the df_annotations and df_images
    df = df_annotations.merge(df_images,on='image_id', how='left')


    # In data frame df_categories: change the column name 'id' to 'category_id'.
    df_categories.rename(columns={'id':'category_id'}, inplace=True)

    # Left join the df and df_categories
    df = df.merge(df_categories,on='category_id', how='left')
    return df
    
# crating new json file
def creating_json_from_df (df, file_name):

    coco_dict = {'images': [], 'annotations': [], 'categories': []}

    # Add categories
    categories = df[['category_id', 'name', 'supercategory','name_readable']].drop_duplicates()
    for _, row in categories.iterrows():
        category = {'id': row['category_id'], 'name': row['name'], 'name_readable': row['name_readable'], 'supercategory': row['supercategory']}
        coco_dict['categories'].append(category)

    # Add images and annotations
    images= df[['image_id','file_name','height','width']].drop_duplicates()
    for _, row in images.iterrows():
    # Add image
        image = {'id': row['image_id'], 'file_name': row['file_name'], 'height': row['height'], 'width': row['width']}
        coco_dict['images'].append(image)
    
    for _, row in df.iterrows():
    # Add annotation
        annotation = {'id': row['id'], 'image_id': row['image_id'], 'category_id': row['category_id'],
                    'segmentation': row['segmentation'], 'area': row['area'], 'bbox': row['bbox'], 'iscrowd': row['iscrowd']}
        coco_dict['annotations'].append(annotation)

    with open(file_name, 'w') as f:
        json.dump(coco_dict, f)    
    return True        

import settings_preprocessing 

if __name__ == "__main__":

    PATH_TRAIN_ANNOTATIONS=settings_preprocessing.PATH_TO_ANNOTATIONS
    OUTPUT_ANNOTATIONS=settings_preprocessing.PATH_OUTPUT_ANNOTATIONS

    df_train=create_df_from_json(PATH_TRAIN_ANNOTATIONS)

    df_files=df_train.groupby(['file_name','category_id']).count().reset_index().groupby(['file_name']).count().reset_index()

    file_list3more=df_files[df_files.category_id>=3].file_name.unique().tolist()
    
    df_3more=df_train[df_train.file_name.isin(file_list3more)]

    creating_json_from_df(df_3more, OUTPUT_ANNOTATIONS)
