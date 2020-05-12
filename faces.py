import cv2
import pickle


face_cascade = cv2.CascadeClassifier('cascades/data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainner.yml")

labels = {}
with open("labels.pickle", 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}

cap = cv2.VideoCapture(0)

while(True):
    #capture frame by frame
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    for (x, y, w, h) in faces:
        print(x, y, w, h)
        roi_gray = gray[y:y+h, x:x+w]

        id_, conf = recognizer.predict(roi_gray)
        if conf>=45: #and conf<=85:
            print(id_)
            print(labels[id_])
            print(conf)
            font = cv2.FONT_HERSHEY_SIMPLEX
            name =  labels[id_]
            color = (255, 255, 255)
            stroke = 1
            cv2.putText(gray, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)


        img_item =  "my-image.png"
        cv2.imwrite(img_item, roi_gray)

        color = (255, 0, 0)
        stroke = 2
        width = x + w
        height = y + h
        cv2.rectangle(gray, (x, y), (width, height), color, stroke)
    #display frame
    cv2.imshow('frame', gray)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
