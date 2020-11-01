# Simulation

A deep learning model created with Tensorflow and Keras + OpenCV to traverse the autonomous track used by Udacity Self-Driving Car Simulator\
https://github.com/udacity/self-driving-car-sim

To test this model, clone this repo and use a virtual env(like venv or conda environment)\
Install necessary modules by running 'pip install -requirements.txt' or 'conda insatll --file requirements.txt' R\
Run the sim_connect.py (Basic Flask, Socketio WSGL)
Run the simulator and choose autonomous mode\
[nvidia_model](https://github.com/Vignesh-Desmond/simulation/blob/master/nvidia_model.py) is the src file for nvidia_model

## Note:
Sample IMG and driving_log files are also uploaded

Also added:
1. [traffic_signs_classifier](https://github.com/Vignesh-Desmond/simulation/blob/master/traffic_signs_classifier.py) for road sign classification
2. [lanes](https://github.com/Vignesh-Desmond/simulation/blob/master/lanes.py) for lane detection (Hough Transform based)

For traffic_signs_classifier, [german-traffic-signs](https://bitbucket.org/jadslim/german-traffic-signs) can be used for training/validation
