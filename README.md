# Image Inpainting Challenge 2022
This was a simple image inpainting project where a simple convolutional Neural Network CNN is used to fill in the holes in an image. This was a challenge in the course 'Programming in Python 2' at the Johannes Kepler University Linz. I already did the course; however, we had a different project. I wanted to do the project to showcase the students in the $2$. semester and my mentoring group. I explained the code and how to start a basic machine learning project in Python.

To play aroud with the model which can be done here: https://abdulbb-imageinpaintingchallenge2022-streamlit-app-v8lsgu.streamlitapp.com/

To run this programm via command line:

```
python3 main.py working_config.json
```

```
ImageInpaintingChallenge2022
|- architectures.py
|    Classes and functions needed for network architectures
|- datasets.py
|    Dataset classes and dataset helper functions. 
|- main.py
|    Main file. In this case also includes training and evaluation functions.
|- README.md
|    A readme file containing info on project.
|- utils.py
|    Utility functions and classes. Here you will find the plotting function.
|- working_config.json
|    An example configuration file. So one can run via command line arguments to main.py.
|- streamlit_app.py
|    File for the Web App. 
```
