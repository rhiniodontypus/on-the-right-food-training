# Import libraries necessary for this project
import numpy as np
import pandas as pd
import json
import cv2
# Import visualisation libraries
import seaborn as sns
import matplotlib.pyplot as plt
#import skimage.io as io
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
    
def get_coords(single_row_serie_from_df):
    '''
    returns separate x_coord and y_coord from the first segmentation in annotation
    
    '''
    num_points_segm=len(single_row_serie_from_df.segmentation[0])
    points=single_row_serie_from_df.segmentation[0]
    x_coord=[]
    y_coord=[]
    for idxp in range(num_points_segm):
        if idxp % 2 ==0 : x_coord.append(points[idxp])
        else: y_coord.append(points[idxp])

    assert len(x_coord)==int(num_points_segm/2)
    assert len(y_coord)==int(num_points_segm/2)
    return x_coord,y_coord

def get_coords_full(single_row_serie_from_df):
    '''
    returns separate x_coord and y_coord from all segmentations in annotation
    
    '''
    x_coord=[]
    y_coord=[]
    for segm in single_row_serie_from_df.segmentation:
        num_points_segm=len(segm)
        points=segm 
   
        for idxp in range(num_points_segm):
            if idxp % 2 ==0 : x_coord.append(points[idxp])
            else: y_coord.append(points[idxp])

        #assert len(x_coord)==int(num_points_segm/2)
        #assert len(y_coord)==int(num_points_segm/2)
    return x_coord,y_coord

def get_bbox(x_coord_list,y_coord_list, type='xyxy'):
    """
    returns a bbox from x and y coordinates in a format:
    type=
    'xyxy' xmin,ymin,xmax,ymax
    'xywh' xmin,ymin,width,height
    'cxywh' x_center,y_center, width, height

    """
    x_min=min(x_coord_list)
    x_max=max(x_coord_list)
    y_min=min(y_coord_list)
    y_max=max(y_coord_list)

    if type=='xyxy': 
        return([x_min,y_min,x_max,y_max])
    if type=='xywh': 
        return([x_min,y_min,x_max-x_min+1,y_max-y_min+1])
    if type=='cxywh': 
        return([int((x_min+x_max)//2),int((y_min+y_max)//2),x_max-x_min+1,y_max-y_min+1])
    


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

# getting imsize:
def img_dim(fname_list, path):
    fnal=dict()
    height_list=[]
    width_list=[]
    for fname in fname_list:
        if fname in fnal:
            height_list.append(fnal[fname][0])
            width_list.append(fnal[fname][1])
        else:
            img=cv2.imread(path+fname)
            height_list.append(img.shape[0])
            width_list.append(img.shape[1])
            fnal[fname]=[img.shape[0],img.shape[1]]
        
    
    
    return height_list, width_list 

import settings_preprocessing 

if __name__ == "__main__":
# write correct path to json and images

    PATH_TRAIN_SUBSET_ANNOTATIONS=settings_preprocessing.PATH_TO_ANNOTATIONS
    PATH_TRAIN_SUBSET_IMAGES=settings_preprocessing.PATH_TO_IMAGES
    PATH_TRAIN_SUBSET_NEW_ANNOTATIONS=settings_preprocessing.PATH_OUTPUT_ANNOTATIONS


    #reading annotations and creating new bboxes
    df_from_json=create_df_from_json(PATH_TRAIN_SUBSET_ANNOTATIONS)
    new_boxes=[]
    for i in range(len(df_from_json)):
        xc,yc=get_coords_full(df_from_json.iloc[i,])
        new_boxes.append(get_bbox(xc,yc,'xywh'))

    #exchange bbox column
    df_from_json.bbox=new_boxes

    dim_check=True

    if dim_check:
        new_height, new_width=img_dim(df_from_json.file_name.values.tolist(), path=PATH_TRAIN_SUBSET_IMAGES)    
        k=0
        for i in range(len(df_from_json)):
            if df_from_json.width[i]!= new_width[i]: 
                df_from_json.width[i]= new_width[i]
                k=k+1
            if df_from_json.height[i]!= new_height[i]: 
                df_from_json.height[i]= new_height[i]
                k=k+1
        print(k,' dimension were corrected')



    creating_json_from_df(df_from_json, PATH_TRAIN_SUBSET_NEW_ANNOTATIONS)
