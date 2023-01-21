# necessary imports
import mediapipe as mp
import cv2
import math
import numpy as np

# initialize variables
good_frames = 0
repetition = 0
good_time = 0
color = (127, 255, 0)
repetition_started = False
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# create a video capture object and load the video
filePath="C:/Users/HP/Downloads/knee_orginal.mp4"
cap = cv2.VideoCapture(filePath)
size = (int(cap.get(3)), int(cap.get(4)))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
result = cv2.VideoWriter('knee_bend_output.mp4',
                         fourcc,
                         20, size)


# read the first frame of the video for later comparision
_, frame1 = cap.read()
# convert the first frame to grayscale
gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

# list of all frames to average out
frames = [gray1]

# Number of frames to average
num_frames = 10


# function to find the inner bend of the knee
def angle3pt(p, q, r):
    if p[0] != 0 and q[0] != 0 and r[0] != 0:
        ang = math.degrees(math.atan2(r[1] - q[1], r[0] - q[0]) - math.atan2(p[1] - q[1], p[0] - q[0]))
        return ang + 360 if ang < 0 else ang


# start reading the video frame by frame
with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            # break the loop if video ends
            break

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        h, w = image.shape[:2]
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # process image using mediapipe and see landmarks
        results = pose.process(image)
        lm = results.pose_landmarks
        lmPose = mp_pose.PoseLandmark

        if lm is None:  # if no landmarks spotted go to the next frame
            continue

        else:

            # read the frame and append it to the list
            _, frame2 = cap.read()
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            frames.append(gray2)

            if len(frames) > num_frames:
                # Remove the oldest frame
                frames.pop(0)

            # Perform frame differencing
            average_frame = np.mean(frames, axis=0)
            average_frame = average_frame.astype(np.uint8)
            diff = cv2.absdiff(average_frame, gray2)

            # Threshold the difference image
            thresh = cv2.adaptiveThreshold(diff, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            non_zeros = cv2.countNonZero(thresh)

            threshold = 460000
            # if non zeros are less than threshold, the frame is static
            if non_zeros < threshold:
                print("Skipping static frame")
                # skip this frame and move to next
                continue
            else:  # if non static frame

                # find the x coordinate of the left shoulder
                # to determine if the person is sitting on left side or right
                l_shoulder_x = int(lm.landmark[lmPose.LEFT_SHOULDER].x * w)

                if (l_shoulder_x < 350):  # right side: use right landmarks
                    knee_x = int(lm.landmark[lmPose.RIGHT_KNEE].x * w)
                    knee_y = int(lm.landmark[lmPose.RIGHT_KNEE].y * h)

                    ankle_x = int(lm.landmark[lmPose.RIGHT_ANKLE].x * w)
                    ankle_y = int(lm.landmark[lmPose.RIGHT_ANKLE].y * h)

                    hip_x = int(lm.landmark[lmPose.RIGHT_HIP].x * w)
                    hip_y = int(lm.landmark[lmPose.RIGHT_HIP].x * h)
                else:  # left side: use left landmarks
                    knee_x = int(lm.landmark[lmPose.LEFT_KNEE].x * w)
                    knee_y = int(lm.landmark[lmPose.LEFT_KNEE].y * h)

                    ankle_x = int(lm.landmark[lmPose.LEFT_ANKLE].x * w)
                    ankle_y = int(lm.landmark[lmPose.LEFT_ANKLE].y * h)

                    hip_x = int(lm.landmark[lmPose.LEFT_HIP].x * w)
                    hip_y = int(lm.landmark[lmPose.LEFT_HIP].x * h)

                # knee,ankle,hip positions
                knee = [knee_x, knee_y]
                ankle = [ankle_x, ankle_y]
                hip = [hip_x, hip_y]

                # calculate the angle
                angle = round(angle3pt(hip, knee, ankle), 0)
                cv2.putText(image, "knee angle :" +str(round(angle,0)), (350, h - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                # if angle<180 increase the counter or
                # if angle>300(if the ankle is too close to hip)
                if angle < 180 or angle > 300:
                    good_frames += 1

                    # if counter is equal to 8 seconds increase repetition
                    if not repetition_started and good_time >= 8:
                        repetition += 1
                        repetition_started = True
                        color = (255, 0, 0)


                else:  # if leg straight make the counter 0

                    if good_time>0:
                        cv2.putText(image, "Keep your knee bent", (10, h - 100), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                                    (255, 0, 0), 2)

                    repetition_started = False
                    good_frames = 0


                # calculate time in the bend position
                good_time = (2 / fps) * good_frames

                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(image, str(round(good_time, 0)), (10, h - 50), font, 0.9, color, 2)

                if (good_time == 0):
                    color = (127, 255, 0)

                repetition_counter = 'Repetition : ' + str(repetition)
                green = (127, 255, 0)
                cv2.putText(image, repetition_counter, (10, h - 20), font, 0.9, green, 2)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # update the frame for next comparison
            gray1 = gray2

        # Draw the pose annotation on the image.
        image.flags.writeable = True
        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        # cv2.imshow('MediaPipe Pose', image)
        result.write(image)

        if cv2.waitKey(5) & 0xFF == 27:
            break

result.release()
cap.release()
cv2.destroyAllWindows()
print(repetition)
