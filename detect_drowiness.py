import cv2
import os
from keras.models import load_model
import numpy as np
from pygame import mixer


mixer.init()
sound = mixer.Sound('alarm.wav')
##files used to detect the face and the eyes of the driver
face = cv2.CascadeClassifier('detection_files\detection_frontalface_alt.xml')
leye = cv2.CascadeClassifier('detection_files\detection_lefteye_2splits.xml')
reye = cv2.CascadeClassifier('detection_files\detection_righteye_2splits.xml')

#check if eyes closed or open
check_eye = ['Close', 'Open']
model = load_model('models/cnn_classification.h5')
path = os.getcwd()
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
count = 0
#get how long the driver closed his eyes
#if score >10 alarm will turn on
score = 0
alarm_play = 4#parameter of the alarm
reye_predict= [10] # predict if right eye colsed or open
leye_predict= [10] # predict if left eye colsed or open

while (True):
    ret, frame = cap.read()
    height, width = frame.shape[:2]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face.detectMultiScale(gray, minNeighbors=5, scaleFactor=1.1, minSize=(25, 25))
    left_eye = leye.detectMultiScale(gray)
    right_eye = reye.detectMultiScale(gray)

    cv2.rectangle(frame, (0, height - 50), (200, height), (0, 0, 0), thickness=cv2.FILLED)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 100, 100), 1)

    for (x, y, w, h) in right_eye:
        r_eye = frame[y:y + h, x:x + w]
        count = count + 1
        r_eye = cv2.cvtColor(r_eye, cv2.COLOR_BGR2GRAY)
        r_eye = cv2.resize(r_eye, (24, 24))
        r_eye = r_eye / 255
        r_eye = r_eye.reshape(24, 24, -1)
        r_eye = np.expand_dims(r_eye, axis=0)

        reye_predict= model.predict(r_eye)
        reye_predict= np.argmax(reye_predict, axis=1)
        if (reye_predict[0] == 1):
            check_eye = 'Open'
        if (reye_predict[0] == 0):
            check_eye = 'Closed'
        break

    for (x, y, w, h) in left_eye:
        l_eye = frame[y:y + h, x:x + w]
        count = count + 1
        l_eye = cv2.cvtColor(l_eye, cv2.COLOR_BGR2GRAY)
        l_eye = cv2.resize(l_eye, (24, 24))
        l_eye = l_eye / 255
        l_eye = l_eye.reshape(24, 24, -1)
        l_eye = np.expand_dims(l_eye, axis=0)

        leye_predict = model.predict(l_eye)
        leye_predict= np.argmax(leye_predict, axis=1)
        if (leye_predict[0] == 1):
            check_eye = 'Open'
        if (leye_predict[0] == 0):
            check_eye = 'Closed'
        break

    if (reye_predict[0] == 0 and leye_predict[0] == 0): #if both eyes closed
        score = score + 1
        cv2.putText(frame, "Closed", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    else:
        score = score - 1
        cv2.putText(frame, "Open", (10, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)

    if (score < 0):
        score = 0
    cv2.putText(frame, 'Score:' + str(score), (100, height - 20), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
    if (score > 10):
        # the driver colsed his eyes or hold his phone for long time (10 sec) the alarm will turn on
        cv2.imwrite(os.path.join(path, 'image.jpg'), frame)
        try:
            sound.play()

        except:  # isplaying = False
            pass
        if (alarm_play < 16):
            alarm_play = alarm_play+ 2
        else:
            alarm_play = alarm_play- 2
            if (alarm_play < 2):
                alarm_play = 2
        cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), alarm_play )
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
