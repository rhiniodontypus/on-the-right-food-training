# On The Right Food - Training

The goal of this repository is to create a prediction model using a subset of preprocessed images described by the `annotations.json` file.

Here we describe both the preprocessing and training methods that we conducted on a virtual machine (VM) on the [Google Cloud Platform.](https://cloud.google.com/compute/)

To set up your VM you can clone and install this repository directly to your VM.

## 1. Installation
We recommend to set up a virtual environment. 

1. Set the local python version 3.9.8.

    `pyenv local 3.9.8`

2. Create and activate the virtual ennvironment

    `python -m venv .venv`
    
    `source .venv/bin/activate`

Make sure you use a pip version <= 23.0.1. Otherwise the installation of detectron2 will fail!

`python -m pip install --upgrade pip==23.0.1`

3. Install the required python packages:

    `python -m pip install -r requirements.txt`

4. Install detectron2:

    `python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'`

## 2. Preprocessing

To modify the images upload the original set of images (we used the dataset from the [AICrowd Food Recognition Competition](https://www.aicrowd.com/challenges/food-recognition-benchmark-2022)) to `/home/user/on-the-right-food-training/data/images/` and the `annotations.json` file to the `/home/user/on-the-right-food-training/data/annotations.json` folder on your VM. You can change the paths in `settings_preprocessing.py`.

1. `bboxcorrection.py` checks and corrects whether bounding boxes around annotated images are fitting the segmentation boundaries.

2. `file_comb_random.py` combines four images into one and creates in total 10000 random files (the number of files can be changed in the script) from the original dataset.

3. `annotations_3.py` selects files with 3 or more different food categories.

4. As a very particular task for the food recognition training dataset `simple_label.py` was created to reduce different labels for obviously similar classes of food.

5. `subset_train_val.py` adjusts the labelling of the training and validation datasets so that they have the same categories present after training dataset subsetting. Each of this scripts returns a new .json files with annotations that should be used in `trainer_gcp.py`.


## 3. Training

1. Run `trainer_gcp.py`.
    
    When the training is completed you can find the prediction model file `model_final.pth` in the `/home/user/on-the-right-food-training/output/` folder.

    If you want to add a timestamp to your model name, uncomment the `TIMESTAMP` block in `trainer_gcp-py`.

    If you want to dump the config values as yaml, uncomment the `YAML` block in `trainer_gcp-py`.

    If you want to archive your training output, uncomment the `ARCHIVE` block in `trainer_gcp-py`.


## 4. Transfer to your local machine

1. Now you can follow the instructions described under [on-the-right-food#2-web-app-installation.](https://github.com/rhiniodontypus/on-the-right-food#2-web-app-installation)


