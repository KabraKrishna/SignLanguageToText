# Sign Language To Speach / Text
### A machine learning and computer vision based approach to convert Sign language to Speach / Text

## Demo link while testing
https://youtu.be/f-ra7_zMwLg

## Dataset for train and test link
https://mavsuta-my.sharepoint.com/:f:/r/personal/vxc0340_mavs_uta_edu/Documents/dataset?csf=1&web=1&e=4XjNXj

## Link for the model.h5 file 
download and save the model.h5 file from the following link in the same directory before running the code

## Description:
Our aim for developing this research is that it may be deployed as a mobile application in the future for persons who cannot speak, acting as a transulator.
since these are the early stages of implementation, yet these are helpful concepts So far, we've improved our accuracy while training. When we tested the program on a simple background, it worked well, but when there is a lot of noise, such as background objects, it takes time and some prediction, and we also have to consider that we have to distinguish 26 different signs, rather than simply 1 or 2 faces in a face recognition thats so challenging.

## Dependencies
* python v3.10.x
* gTTS v2.2.4
* keras v2.8.0
* matplotlib v3.5.3
* nltk v3.7
* numpy v1.22.4
* opencv_python_headless v4.6.0.66
* pyttsx3 v2.90
* scikit_learn v1.1.3
* tensorflow v2.8.2
* textblob v0.17.1

## Steps to run the model
* before running the code unzip all the compressed files. 
* Install python v3.10.x if not present already.
* Install dependencies from **requirements.txt** file using command below:
    
    >  ` pip install requirements.txt`

* Run the jupyter notebook `app.ipynb` and run it to test the model.
