import numpy as np
import pandas as pd
import json

# Import visualisation libraries
import seaborn as sns
import matplotlib.pyplot as plt
import skimage.io as io
from pycocotools.coco import COCO

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
    
# creating new json file
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

def get_annotations_subset(df_from_json, type='first_cats', n=1, files=False, category_list=[]):
    '''
    picks up first n categories or n random_files from the annotation dataset
    returns dataset with annotations from files=False only categories
    files=True subset of files containing the n first categories and other categories from the files

    type='first_cats' takes n most frequent categories
    
    type='random_files' takes n random files (for this all annotations so files=True always)

    if category_list is not empty- looks for the categories from the list

    returns final dataset of annotations and chosen category list
    '''
    if len(category_list)==0:
        if type=='first_cats': 
            top_categories_list=df_from_json.groupby(['file_name','category_id']).count().reset_index().value_counts('category_id').index.tolist()[:n]
            if not files:
                return top_categories_list, df_from_json[df_from_json.category_id.isin(top_categories_list)]
            else:
                top_cat_file_list=df_from_json[df_from_json.category_id.isin(top_categories_list)].file_name.unique().tolist()
                return top_categories_list, df_from_json[df_from_json.file_name.isin(top_cat_file_list)]
        if type=='random_files':
            fnames=df_from_json.file_name.unique().tolist()
            flist=np.random.choice(fnames,n,replace=False).tolist()
            cat_list= df_from_json[df_from_json.file_name.isin(flist)].category_id.unique().tolist()
            return cat_list, df_from_json[df_from_json.file_name.isin(flist)]
    else:
        #if not files:
         #   return category_list, df_from_json[df_from_json.category_id.isin(category_list)]
        return category_list, df_from_json[df_from_json.category_id.isin(category_list)]
        #else:
         #   top_cat_file_list=df_from_json[df_from_json.category_id.isin(category_list)].file_name.unique().tolist()
          #  return category_list, df_from_json[df_from_json.file_name.isin(top_cat_file_list)]

import settings_preprocessing

if __name__ == "__main__":
    
    PATH_TRAIN_ANNOTAITONS=settings_preprocessing.PATH_TO_ANNOTATIONS
    PATH_VAL_ANNOTATIONS=settings_preprocessing.PATH_TO_VAL_ANNOTATIONS
    OUTPUT_TRAIN_ANNOTATIONS=settings_preprocessing.PATH_OUTPUT_ANNOTATIONS
    OUTPUT_VAL_ANNOTATIONS=settings_preprocessing.PATH_OUTPUT_VAL_ANNOTATIONS
   

    df_train=create_df_from_json(PATH_TRAIN_ANNOTATIONS)
    cats, df_sub_train= get_annotations_subset(df_train, type='first_cats',n=50)

    df_val=create_df_from_json(PATH_VAL_ANNOTATIONS)
    cats2, df_sub_val= get_annotations_subset(df_val, category_list=cats)

    creating_json_from_df(df_sub_train,OUTPUT_TRAIN_ANNOTATIONS)
    creating_json_from_df(df_sub_val,OUTPUT_VAL_ANNOTATIONS)
