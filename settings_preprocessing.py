
# set paths to the annotations file and the folder with images
# set output with a appropriate file name for .json file
# if you need to run several preprocessing steps keep in mind to rename the starting file and the ending file for every step of preprocessing
# NOTICE that often you need only make changes within .json file where all the data about images and annotations is stored, i.e. without creating a new subset folder of images

PATH_TO_ANNOTATIONS='./data/train/annotations.json'
PATH_TO_IMAGES='./data/train/images/'

PATH_OUTPUT_ANNOTATIONS='./data/train/annotations_new.json'
PATH_OUTPUT_IMAGES='./data/train/images_preprocessed/'


#set path to the validation dataset files and folders together with training set settings above to make a proper validation subset
PATH_TO_VAL_ANNOTATIONS='./data/val/annotations.json'
PATH_TO_VAL_IMAGES='./data/val/images/'
PATH_OUTPUT_VAL_ANNOTATIONS='./data/val/annotations_new.json'
PATH_OUTPUT_VAL_IMAGES='./data/val_new/images/'
