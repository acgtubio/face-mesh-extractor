import cv2
import mediapipe as mp


mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

cap = cv2.VideoCapture(0)


while True:

    ret, image = cap.read()

    if ret is not True:
        break

    height, width, _ = image.shape
    #print(height, width)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    result = face_mesh.process(rgb_image)


    if result.multi_face_landmarks:
        for facial_landmarks in result.multi_face_landmarks:
            landmarks = facial_landmarks.landmark
            for landmark in landmarks:
                x = landmark.x
                y = landmark.y
                z = landmark.z

                cv2.circle(image, (int(landmark.x * width), int(landmark.y * height)), 2, (100,100,0), -1)

            print(x, y, z)


    cv2.imshow("Image", image)
    cv2.waitKey(1)