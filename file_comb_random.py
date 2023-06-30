import numpy as np
import pandas as pd
import json

# Import visualisation libraries
import seaborn as sns
import matplotlib.pyplot as plt
import skimage.io as io
from pycocotools.coco import COCO

from matplotlib.pyplot import imshow
import cv2
import pickle



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

def annot_file_position(single_row_serie_from_df, pos=1, dim=[320,320]):
    
    FX=single_row_serie_from_df.width/dim[0]
    
    FY=single_row_serie_from_df.height/dim[1]

    new_box=single_row_serie_from_df.bbox
    
    if pos==1: add=[0,0]
    if pos==2: add=[dim[0], 0]
    if pos==3: add=[0, dim[1]]
    if pos==4: add=[dim[0],dim[1]]
    
    new_segm=[]
    for segm in single_row_serie_from_df.segmentation:
        num_points_segm=len(segm)
        points=segm 

        for idxp in range(num_points_segm):
            if idxp % 2 ==0 : points[idxp]=int(points[idxp]/FX+add[0])
            else: points[idxp]=int(points[idxp]/FY+add[1])

        new_segm.append(points)    

    new_box[0]=new_box[0]/FX+add[0]
    new_box[1]=new_box[1]/FY+add[1]
    new_box[2]=new_box[2]/FX
    new_box[3]=new_box[3]/FY

    new_area=int(single_row_serie_from_df.area/FX/FY)



    return new_segm, new_box, new_area    


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
    
    PATH_TO_IMAGES=settings_preprocessing.PATH_TO_IMAGES
    PATH_TO_ANNOTATIONS=settings_preprocessing.PATH_TO_ANNOTATIONS

    PATH_OUT_IMAGES=settings_preprocessing.PATH_OUTPUT_IMAGES
    PATH_OUT_ANNOTATIONS=settings_preprocessing.PATH_OUTPUT_ANNOTATIONS

    df_train_origin=create_df_from_json(PATH_TO_ANNOTATIONS)

    file_list=df_train_origin.file_name.unique().tolist()


    annotations=pd.DataFrame(columns=df_train_origin.columns)

    N_files=10000

    for times in range(N_files):
        random4files=np.random.choice(file_list,4, replace=False).tolist()
        if times%500 == 0: print(times," images are done!")	
        im1=cv2.resize(cv2.imread(PATH_TO_IMAGES+random4files[0]),[320, 320], interpolation = cv2.INTER_AREA)
        #im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
        im2=cv2.resize(cv2.imread(PATH_TO_IMAGES+random4files[1]),[320, 320], interpolation = cv2.INTER_AREA)
        #im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)
        im3=cv2.resize(cv2.imread(PATH_TO_IMAGES+random4files[2]),[320, 320], interpolation = cv2.INTER_AREA)
        #im3 = cv2.cvtColor(im3, cv2.COLOR_BGR2RGB)
        im4=cv2.resize(cv2.imread(PATH_TO_IMAGES+random4files[3]),[320, 320], interpolation = cv2.INTER_AREA)
        #im4 = cv2.cvtColor(im4, cv2.COLOR_BGR2RGB)

        im_con1=(np.concatenate((im1,im2), axis=1))
        im_con2=(np.concatenate((im3,im4), axis=1))
        im_con4=(np.concatenate((im_con1,im_con2), axis=0))

        df_copy=df_train_origin[df_train_origin.file_name.isin(random4files)].reset_index().drop('index',axis=1)

        subset_anno=pickle.loads(pickle.dumps(df_copy))

        new_file_name='myfile'+str(times)+'.jpg'
        new_file_ID=times # very important parameter
        
        new_boxes=[]
        new_segms=[]
        new_widths=[]
        new_heights=[]
        new_areas=[]

        for i, row in subset_anno.iterrows():
            pos=random4files.index(row.file_name)+1
            new_segm , new_box, new_area= annot_file_position(row, pos=pos, dim=[320,320])

            
            new_boxes.append(new_box)
            new_segms.append(new_segm)
            new_widths.append(640)
            new_heights.append(640)
            new_areas.append(new_area)



        subset_anno['bbox']=new_boxes
        subset_anno['segmentation']=new_segms
        subset_anno['width']=new_widths
        subset_anno['height']=new_heights
        subset_anno['file_name']=[new_file_name]*len(subset_anno)
        subset_anno['image_id']=[new_file_ID]*len(subset_anno)
        subset_anno['area']=new_areas  


        cv2.imwrite(PATH_OUT_IMAGES+new_file_name, im_con4)
        annotations=pd.concat([annotations,subset_anno])

    annotations=annotations.reset_index().drop('index', axis=1)
    annotations['id']=range(len(annotations))

    creating_json_from_df(annotations,PATH_OUT_ANNOTATIONS)
