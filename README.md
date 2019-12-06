# bbMLg

When training we save the model after every epoch.

The file size of each epoch exceeds GitHub's limit. Download the model from here 

https://drive.google.com/file/d/1WXOVx6kl8ex6GT5GVcsUusKEFx66FEx5/view

and place in the model_weights folder

## Preprocessing
Located in preprocessing.py

Note for ease of training, we store the preprocessed images in file located in /processed_info

## CNN Model
cnn_model.py contains the models used

cnn_tuning.py tries different hyperparameters to get the best accuracy

cnn_dice.py contains the endpoint for calling the model as well as training logic

## RNN Model
Can be found on the branch `lstm`

## Solution/UI
run with python3 DiceChecker1-0.py

## Blender
The blender model, associate texture files, and python script used to generate the images are in the blender directory
