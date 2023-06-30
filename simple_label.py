import numpy as np
import pandas as pd
import json
import cv2
# Import visualisation libraries
import seaborn as sns
import matplotlib.pyplot as plt
import skimage.io as io
from pycocotools.coco import COCO
import re

from Subset_train_val import get_annotations_subset

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

def get_category_files(df_from_json,cat):
    return df_from_json[df_from_json.category_id==cat].file_name.unique().tolist()

def random_n_files(file_list,n):
    
    if len(file_list)<=n:
        return file_list
    else:
        return np.random.choice(file_list,n, replace=False)

import settings_preprocessing 

if __name__ == "__main__":

    PATH_TO_ANNOTATIONS=settings_preprocessing.PATH_TO_ANNOTATIONS
    OUTPUT_PATH=settings_preprocessing.PATH_OUTPUT_ANNOTATIONS
   

    df_train=create_df_from_json(PATH_TO_ANNOTATIONS)

    for i, row in df_train.iterrows():
        names=df_train.name.iloc[i]
        if bool(re.search('cheese',names)) & (not bool(re.search('(risotto|quiche|sandwich)',names))): df_train.name.iloc[i]='cheese'
        if bool(re.search('parmesan',names)): df_train.name.iloc[i]='cheese'
        if bool(re.search('gruyere',names)): df_train.name.iloc[i]='cheese'

        if re.search('water(?![a-z])',names): df_train.name.iloc[i]='water'
        if bool(re.search('bread',names)) & (not bool(re.search('(nut)',names))): df_train.name.iloc[i]='bread'
        if re.search('coffee(?![a-z])',names): df_train.name.iloc[i]='coffee'
        if re.search('espresso(?![a-z])',names): df_train.name.iloc[i]='coffee'
        if re.search('tea(?![a-z])',names): df_train.name.iloc[i]='tea'
        if re.search('pizza(?![a-z])',names): df_train.name.iloc[i]='pizza'
        if re.search('ham(?![a-z])',names): df_train.name.iloc[i]='ham'
        if re.search('chicken(?![a-z])',names): df_train.name.iloc[i]='chicken'
        if re.search('chocolate(?![a-z])',names): df_train.name.iloc[i]='chocolate'
    
        if bool(re.search('vegetable',names)) & (not bool(re.search('(lasagne|soup)',names))): df_train.name.iloc[i]='vegetables'
        if bool(re.search('pasta',names)): df_train.name.iloc[i]='pasta'
        if bool(re.search('rice',names)) & (not bool(re.search('(noodles|waffels)',names))): df_train.name.iloc[i]='rice'

        if bool(re.search('salad',names)) & (not bool(re.search('(fruit)',names))): df_train.name.iloc[i]='salad'

        if bool(re.search('tomato',names)) : df_train.name.iloc[i]='tomato'
        if bool(re.search('carrot',names)) & (not bool(re.search('(cake)',names))): df_train.name.iloc[i]='carrot'
        if bool(re.search('potato',names)) : df_train.name.iloc[i]='potato'

        if bool(re.search('sausage',names)) : df_train.name.iloc[i]='sausage'

        if bool(re.search('pork',names)) : df_train.name.iloc[i]='pork'
        if bool(re.search('beef',names)) : df_train.name.iloc[i]='beef'

        if bool(re.search('meat',names)) : df_train.name.iloc[i]='meat'

        if bool(re.search('braided-white-loaf',names)) : df_train.name.iloc[i]='bread'

        if bool(re.search('nuts',names)) : df_train.name.iloc[i]='nuts'
        if bool(re.search('nut',names)) : df_train.name.iloc[i]='nuts'
        if bool(re.search('mushroom',names)) : df_train.name.iloc[i]='mushrooms'

        if bool(re.search('sauce',names)) : df_train.name.iloc[i]='sauce'
        if bool(re.search('bacon',names)) : df_train.name.iloc[i]='bacon'

        if bool(re.search('asparagus',names)) : df_train.name.iloc[i]='asparagus'
        if bool(re.search('muesli',names)) : df_train.name.iloc[i]='muesli'
        if bool(re.search('curry',names)) : df_train.name.iloc[i]='curry'

        if bool(re.search('shrimp',names)) : df_train.name.iloc[i]='shrimp'
        if bool(re.search('prawn',names)) : df_train.name.iloc[i]='shrimp'

        if bool(re.search('yaourt',names)) : df_train.name.iloc[i]='yoghourt'

    top_names_list=df_train.groupby(['file_name','name']).count().reset_index().value_counts('name').index.tolist()[:50]


    train=df_train[df_train.name.isin(top_names_list)]

    for i, row in train.iterrows():
        train.category_id[i]=top_names_list.index(train.name[i])

    top_cat_list=train.category_id.unique().tolist()   

    categories, df_train1= get_annotations_subset(train,category_list=top_cat_list)
    
    n=200

    total_file_list=[]
    for cat in categories:
        file_list_cat=get_category_files(df_train1,cat)
        total_file_list.append(random_n_files(file_list_cat,n))

    # flatten the file list
    flt_file_list=[file for cat_list in total_file_list for file in cat_list]

    #subsetting annotationsfiles:
    df_new=df_train1[df_train1.file_name.isin(flt_file_list)]

    creating_json_from_df(df_new,OUTPUT_PATH)

