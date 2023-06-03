import cv2
import mediapipe as mp
import numpy as np
import os
import csv

class FaceMesh():
    def __init__(self, db=None):
        self.coordinates = []
        self.fileIndex = 0
        self.isNpArray = False

    @staticmethod
    def Extract3DFacialLandmarks(face_mesh):
        lm_coord = [] 
        for facial_landmarks in face_mesh.multi_face_landmarks:
            landmarks = facial_landmarks.landmark
            for landmark in landmarks:
                x = landmark.x
                y = landmark.y
                z = landmark.z
                lm_coord.append([x,y,z])

        return lm_coord
    
    @staticmethod
    def Extract2DFacialLandmarks(face_mesh):
        lm_coord = [] 
        for facial_landmarks in face_mesh.multi_face_landmarks:
            landmarks = facial_landmarks.landmark
            for landmark in landmarks:
                x = landmark.x
                y = landmark.y
                z = landmark.z
                lm_coord.append([x,y])

        return lm_coord

    @staticmethod
    def ExtractEyeLandmarks(face_mesh, is3d):
        right_eye_landmarks = [33, 246, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]
        left_eye_landmarks = [362, 382, 381, 380, 374, 373, 390, 249, 466, 388, 387, 386, 385, 384, 398]
        lm_coord = [] 
        for facial_landmarks in face_mesh.multi_face_landmarks:
            landmarks = facial_landmarks.landmark
            for i, landmark in enumerate(landmarks):
                if i not in left_eye_landmarks and i not in right_eye_landmarks:
                    continue
                x = landmark.x
                y = landmark.y
                z = landmark.z
                if is3d:
                    lm_coord.append([x, y, z])
                else: 
                    lm_coord.append([x, y])

        return lm_coord


    @staticmethod
    def CalculateEAR(landmarks):
        pass


    def ExtractFaceMesh(self, file, extractor):
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh()

        cap = cv2.VideoCapture(file)

        if (cap.isOpened() == False):
            raise Exception("Error opening video stream or file")
        
        while(cap.isOpened()):
            ret, image = cap.read()

            if ret is not True:
                break
            
            height, width, _ = image.shape
            #print(height, width)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            result = face_mesh.process(rgb_image)

            if result.multi_face_landmarks:
               self.coordinates.append(extractor(result))
                

            cv2.imshow("Image", image)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                self.coordinates = []
                break

        cap.release()
        self.SaveArray()
    
        self.fileIndex += 1
        cv2.destroyAllWindows()

    @staticmethod
    def FaceMeshTest():
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh()
        TEXT_FACE = cv2.FONT_HERSHEY_DUPLEX
        TEXT_SCALE = 0.26
        TEXT_THICKNESS = 2

        cap = cv2.VideoCapture(0)

        while (True):
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
                    for i, landmark in enumerate(landmarks):

                        # if i not in [362, 382, 381, 380, 374, 373, 390, 249, 466, 388, 387, 386, 385, 384, 398] and i not in [33, 246, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]:
                        #     continue

                        # if i in [446, 359, 263, 463, 382, 381]:
                        #     continue

                        x = landmark.x
                        y = landmark.y
                        z = landmark.z

                        cv2.putText(image, f'{i}', ((int(landmark.x * width), int(landmark.y * height))), TEXT_FACE, TEXT_SCALE, (255,255,0))
                        cv2.circle(image, (int(landmark.x * width), int(landmark.y * height)), 2, (50,50,0), -1)

                    # print(x, y, z)
                    
            cv2.imshow("Image", image)
            cv2.waitKey(1)

    def SaveArray(self, file_name=''):
        f = f'./drowsy/{self.fileIndex}.npy'
        features = np.array(self.coordinates)
        np.save(f, features)
        self.coordinates = []

if __name__ == '__main__':
    annot = '10'
    fold_list = ['Fold1_part1', 'Fold1_part2', 'Fold2_part1', 'Fold2_part2', 'Fold3_part1', 'Fold3_part2', 'Fold4_part1', 'Fold4_part2', 'Fold5_part1', 'Fold5_part2']
    extractor = FaceMesh()

    # Camera Test
    extractor.FaceMeshTest()

    # for fold in fold_list:
    #     f = f"./UTA-RLDD/{fold}/"
    #     folders = os.listdir(f)

    #     for folder in folders:
    #         f2 = f'{f}{folder}'
    #         files = os.listdir(f2)

    #         for fil in files:
    #             file_name, file_ext = os.path.splitext(fil)

    #             if file_name == annot:
    #                 inner = f'{f2}/{annot}{file_ext}'
    #                 print(inner)
    #                 extractor.ExtractFaceMesh(f'{inner}')