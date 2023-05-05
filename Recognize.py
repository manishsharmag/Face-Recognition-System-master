import datetime
import os
import time

import cv2
import pandas as pd


def recognize_attendence():
    # Initialize and start realtime video capture
    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cam.set(3, 640)  # set video width
    cam.set(4, 480)  # set video height

    # Check if "Attendance" folder exists, create it if it doesn't
    if not os.path.exists("Attendance"):
        os.makedirs("Attendance")
        print("Created 'Attendance' folder")

    # Check if CSV file exists, create it if it doesn't
    csv_filename = "Attendance/Attendance_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".csv"
    if not os.path.exists(csv_filename):
        attendance_df = pd.DataFrame(columns=["Id", "Name", "Date", "Time"])
        attendance_df.to_csv(csv_filename, index=False)
        print(f"Created '{csv_filename}' file")

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("TrainingImageLabel/Trainner.yml")
    cascade_path = "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    students_df = pd.read_csv("StudentDetails/StudentDetails.csv")
    font = cv2.FONT_HERSHEY_SIMPLEX
    min_w = 0.1 * cam.get(3)
    min_h = 0.1 * cam.get(4)

    while True:
        try:
            ret, im = cam.read()
            gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.2, 5,
                                                  minSize=(int(min_w), int(min_h)),
                                                  flags=cv2.CASCADE_SCALE_IMAGE)
            for (x, y, w, h) in faces:
                cv2.rectangle(im, (x, y), (x+w, y+h), (10, 159, 255), 2)
                id_, conf = recognizer.predict(gray[y:y+h, x:x+w])

                if conf < 100:
                    name = students_df.loc[students_df['Id'] == id_]['Name'].values
                    conf_str = f" {round(100 - conf)}%"
                    tt = f"{id_}-{name}"
                else:
                    id_ = "Unknown"
                    tt = str(id_)
                    conf_str = f" {round(100 - conf)}%"

                if (100 - conf) > 51:
                    ts = time.time()
                    date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                    time_stamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                    name = str(name)[2:-2]
                    attendance_df = pd.read_csv(csv_filename)
                    if not attendance_df[attendance_df["Id"] == id_].empty:
                        continue  # Skip if attendance for this ID has already been marked
                    attendance_df.loc[len(attendance_df)] = [id_, name, date, time_stamp]
                    attendance_df.to_csv(csv_filename, index=False)

                tt = str(tt)[2:-2]
                if (100 - conf) > 51:
                    tt = f"{tt} [Pass]"
                    cv2.putText(im, str(tt), (x, y+h), font, 1, (255, 255, 255), 2)

                else:
                    tt = f"{tt} [Fail]"
                    cv2.putText(im, str(tt), (x, y+h), font, 1, (0, 0, 255), 2)

            cv2.imshow('Attendance', im)
            k = cv2.waitKey(30) & 0xff
            if k == 27:  # Press 'ESC' to quit
                break

        except Exception as e:
            print(f"Error: {str(e)}")
            cam.release()
            cv2.destroyAllWindows()
            break

    cam.release()
    cv2.destroyAllWindows()
