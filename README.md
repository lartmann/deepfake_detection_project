# Deepfake Detection

In this repository is the main code for our Deepfake Detection Project. 
The project is spilt into preprocessing and main.

## Preprocessing
The preprocessing notebook downloads the dataset from Kaggle and applies the preprocessing functions from the helper_functions.py. Afterwards, the output files are saved into a Google Drive when run in Google Colab. In that way the preprocessing only needs to be done once.

## Main
The main Notebook imports the files from Google Drive and uses them to train and evaluate the different Models.
