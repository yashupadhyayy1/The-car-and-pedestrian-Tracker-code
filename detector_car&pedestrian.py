import cv2
from random import randrange


video = cv2.VideoCapture('Close Calls.mp4')
classifier_file = 'car_detector.xml'
classifier_file2 = 'haarcascade_fullbody.xml'

car_tracker = cv2.CascadeClassifier(classifier_file)
ped_tracker = cv2.CascadeClassifier(classifier_file2)

while True:
    (read_sucessful, frame) = video.read()
    if read_sucessful:
        b_n_w = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break

    cars = car_tracker.detectMultiScale(b_n_w)
    ped = ped_tracker.detectMultiScale(b_n_w)

    for (x,y,w,h) in cars:
        cv2.rectangle(frame,(x, y), (x+w, y+h),(128,0,128), 4)
        cv2.rectangle(frame,(x, y), (x+w, y+h),(128,0,128), 4)
    for (x,y,w,h) in ped:
        cv2.rectangle(frame,(x, y), (x+w, y+h),(0,255,255), 4)
    cv2.imshow('car Detector tasveer', frame)
    key = cv2.waitKey(1)
    if key == 27:  # 27 stands for ascii code key of ESC button
        break
video.release()
print('done 👍aha')
