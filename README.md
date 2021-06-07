# Real-time-employee-monitoring-system
This is a backend detection, recognition, and tracking algorithm to keep tracking the employee throughout his 9 hours in the
office and help in evaluating the efficiency and the productivity of the employees.

The project is implemented in the following steps:

1. Run the Capture script, the code will enter the name of the person to be trained. 
   That person's name folder will be created in the computer in the specified folder and samples will be saved there automatically.

2. Trained the model using HOG features to recognize the face of a person, with SVM as a classifier and saved the model for final tracking algorithm.

3. Tracking Algorithm Keeps the track of the person who came in front of the camera with time stamping and creates the CSV file of the same.
