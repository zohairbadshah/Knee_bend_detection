# Knee Bend Detection

This repository contains code and resources for a knee bend detection project using Python's OpenCV and Mediapipe libraries. The goal of this project is to accurately detect knee bend movements in real-time using computer vision techniques.

## About

This project aims to create a system that can detect and quantify knee bend movements using a webcam or video feed. The project utilizes OpenCV for video capture and display, as well as Mediapipe for pose estimation. By analyzing the pose of a person in the video stream, the system can determine the angle of the knee joint and classify it as bent or not bent.

## Algorithm
The algorithm follows these steps:

1. Initialize the webcam and capture frames.
2. Use Mediapipe to detect the pose landmarks of the person in the frame.
3. Calculate the angle between the thigh and calf using the knee joint landmarks.
4. Determine a threshold angle to classify the knee as bent or not bent.
5. Display the angle and classification on the video feed.
## Demo
![image](https://github.com/zohairbadshah/Knee_bend_detection/assets/91787690/2de91cad-a10b-45cc-8af2-baa9458d3951)

