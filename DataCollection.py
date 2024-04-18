import cv2
import os

picture_number = 0

directory_a = 'C:/Users/maxvr/PycharmProjects/pythonProject2/NewDataSet/angry'
directory_h = 'C:/Users/maxvr/PycharmProjects/pythonProject2/NewDataSet/happy'
directory_n = 'C:/Users/maxvr/PycharmProjects/pythonProject2/NewDataSet/neutral'
directory_s = 'C:/Users/maxvr/PycharmProjects/pythonProject2/NewDataSet/sad'

# Start capturing video
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if cv2.waitKey(1) & 0xFF == ord('a'):
        os.chdir(directory_a)
        filename = f'savedImage_21_{picture_number}.jpg'
        cv2.imwrite(filename, frame)
        picture_number += 1
        print("Angry")

    if cv2.waitKey(1) & 0xFF == ord('h'):
        os.chdir(directory_h)
        filename = f'savedImage_21_{picture_number}.jpg'
        cv2.imwrite(filename, frame)
        picture_number += 1
        print("Happy")

    if cv2.waitKey(1) & 0xFF == ord('n'):
        os.chdir(directory_n)
        filename = f'savedImage_21_{picture_number}.jpg'
        cv2.imwrite(filename, frame)
        picture_number += 1
        print("Neutral")

    if cv2.waitKey(1) & 0xFF == ord('s'):
        os.chdir(directory_s)
        filename = f'savedImage_21_{picture_number}.jpg'
        cv2.imwrite(filename, frame)
        picture_number += 1
        print("Sad")


    cv2.imshow('Camera feed', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

