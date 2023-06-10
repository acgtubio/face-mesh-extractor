import cv2
import mediapipe as mp
import numpy as np
import os
import csv
from ear_calculator import calculate_mar, calculate_ear


def test(l, s):
    num_frames = len(l)
    split_size = round(num_frames / s)
    split_list = []

    for i in range(0, num_frames, split_size):
        split_list.append(l[i:i+split_size])

    print(split_list)

class FaceMesh():
    def __init__(self, db=None):
        self.fileIndex = 0
        self.isNpArray = False

    @staticmethod
    def ExtractEarMar(face_mesh, is3d, dims):
        lm_coord = []
        ear_mar = [] 
        for facial_landmarks in face_mesh.multi_face_landmarks:
            landmarks = facial_landmarks.landmark
            for landmark in landmarks:
                x = landmark.x
                y = landmark.y
                z = landmark.z
                
                if is3d:
                    lm_coord.append([x, y, z])
                else: 
                    lm_coord.append([x * dims['width'], y * dims['height']])

        calculate_ear(lm_coord, ear_mar)
        calculate_mar(lm_coord, ear_mar)

        return ear_mar


    @staticmethod
    def Extract3DFacialLandmarks(face_mesh, is3d):
        lm_coord = [] 
        for facial_landmarks in face_mesh.multi_face_landmarks:
            landmarks = facial_landmarks.landmark
            for landmark in landmarks:
                x = landmark.x
                y = landmark.y
                z = landmark.z
                
                if is3d:
                    lm_coord.append([x, y, z])
                else: 
                    lm_coord.append([x, y])

        return lm_coord
    
    @staticmethod
    def Extract2DFacialLandmarks(face_mesh, is3d):
        lm_coord = [] 
        for facial_landmarks in face_mesh.multi_face_landmarks:
            landmarks = facial_landmarks.landmark
            for landmark in landmarks:
                x = landmark.x
                y = landmark.y
                z = landmark.z

                if is3d:
                    lm_coord.append([x, y, z])
                else: 
                    lm_coord.append([x, y])

    @staticmethod
    def ExtractLandmarksWithSplit():
        pass

    @staticmethod
    def ExtractEyeAndMouthLandmarks(face_mesh, is3d):
        right_eye_landmarks = [33, 246, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]
        left_eye_landmarks = [362, 382, 381, 380, 374, 373, 390, 249, 466, 388, 387, 386, 385, 384, 398]
        mouth_landmarks = [13, 14, 312, 317, 82, 87, 178, 402, 311, 81, 88, 95, 183, 42, 78, 318, 310, 324, 415, 308]
        lm_coord = [] 
        for facial_landmarks in face_mesh.multi_face_landmarks:
            landmarks = facial_landmarks.landmark
            for i, landmark in enumerate(landmarks):
                if i not in left_eye_landmarks and i not in right_eye_landmarks and i not in mouth_landmarks:
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

    def ExtractFaceMesh(self, file, extractor, num_splits, is3d, class_name, split_all_frames=False):
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh()
        coordinates = []

        cap = cv2.VideoCapture(file)

        if (cap.isOpened() == False):
            raise Exception("Error opening video stream or file")
        
        while(cap.isOpened()):
            ret, image = cap.read()

            if ret is not True:
                break
            
            height, width, _ = image.shape
            dims = {
                'height': height,
                'width': width
            }
            # print(dims)
            #print(height, width)
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            result = face_mesh.process(rgb_image)

            if result.multi_face_landmarks:
               coordinates.append(extractor(result, is3d, dims))
                

            # cv2.imshow("Image", image)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                coordinates = []
                break

        
        split_list = []
        num_frames = len(coordinates)

        if (not split_all_frames):
            split_size = round(num_frames / num_splits)
        else:
            split_size = 1

        for i in range(0, num_frames, split_size):
            split_list.append(coordinates[i:i+split_size])

        cap.release()

        for i, split in enumerate(split_list):
            self.SaveArray(split, class_name, f'split{i}')
    
        self.fileIndex += 1
        cv2.destroyAllWindows()

    @staticmethod
    def FaceMeshTest():
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh()
        TEXT_FACE = cv2.FONT_HERSHEY_DUPLEX
        TEXT_SCALE = 0.25
        TEXT_THICKNESS = 2

        mouth_landmarks = [13, 14, 312, 317, 82, 87, 178, 402, 311,           81,                                   88,         95,               183,     42,     78,     318, 310, 324, 415     , 308,               ]
        right_eye_landmarks = [33, 246, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7]
        left_eye_landmarks = [362, 382, 381, 380, 374, 373, 390, 249, 466, 388, 387, 386, 385, 384, 398]
        exclude =         [13, 14, 312, 317, 82, 87, 178, 402, 303, 271, 311, 81, 41, 74, 40, 185, 146, 91, 90, 77, 88, 89, 96, 95, 76, 184, 191, 183, 80, 42, 61, 78, 62, 318, 310, 324, 415, 407, 391, 308, 375, 290, 292, 291, 306]

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

                        if i not in left_eye_landmarks and i not in right_eye_landmarks and i not in mouth_landmarks:
                            continue

                        # if i != 263:
                        #     continue

                        # if i in exclude:
                        #     continue

                        x = landmark.x
                        y = landmark.y
                        z = landmark.z

                        cv2.putText(image, f'{i}', ((int(landmark.x * width), int(landmark.y * height))), TEXT_FACE, TEXT_SCALE, (255,255,0))
                        cv2.circle(image, (int(landmark.x * width), int(landmark.y * height)), 2, (50,50,0), -1)

                    # print(x, y, z)
                    
            cv2.imshow("Image", image)
            cv2.waitKey(1)

    def SaveArray(self, coordinates, class_name, file_name=''):
        f = f'./{class_name}/{self.fileIndex}{file_name}.npy'
        features = np.array(coordinates)
        np.save(f, features)

if __name__ == '__main__':
    annot = ['0', '10']
    fold_list = ['Fold1_part1', 'Fold1_part2', 'Fold2_part1', 'Fold2_part2', 'Fold3_part1', 'Fold3_part2', 'Fold4_part1', 'Fold4_part2', 'Fold5_part1', 'Fold5_part2']

    # Camera Test
    # extractor = FaceMesh()
    # extractor.FaceMeshTest()

    print('EXTRACTING EAR ====================================')
    for label in annot:
        class_name = 'ear2/alert' if label == '0' else 'ear2/drowsy'
        extractor = FaceMesh()

        for fold in fold_list:
            f = f"/home/eyd/Documents/Coding/_DATASETS/UTA-RLDD/{fold}/"
            folders = os.listdir(f)

            for folder in folders:
                f2 = f'{f}{folder}'
                files = os.listdir(f2)

                for fil in files:
                    file_name, file_ext = os.path.splitext(fil)

                    if file_name == label:
                        inner = f'{f2}/{label}{file_ext}'
                        print(inner)
                        try:
                            extractor.ExtractFaceMesh(f'{inner}', FaceMesh.ExtractEarMar, 3, False, class_name, split_all_frames=True)
                        except Exception as e:
                            print(f'failed on {inner}. error {e}')

    # print('EXTRACTING EYES AND MOUTH LANDMARKS ====================================')

    # for label in annot:
    #     class_name = 'em/alert' if label == '0' else 'em/drowsy'
    #     extractor = FaceMesh()

    #     for fold in fold_list:
    #         f = f"/home/eyd/Documents/Coding/_DATASETS/UTA-RLDD/{fold}/"
    #         folders = os.listdir(f)

    #         for folder in folders:
    #             f2 = f'{f}{folder}'
    #             files = os.listdir(f2)

    #             for fil in files:
    #                 file_name, file_ext = os.path.splitext(fil)

    #                 if file_name == label:
    #                     inner = f'{f2}/{label}{file_ext}'
    #                     print(inner)
    #                     try:
    #                         extractor.ExtractFaceMesh(f'{inner}', FaceMesh.ExtractEyeAndMouthLandmarks, 3, True, class_name)
    #                     except Exception as e:
    #                         print(f'failed on {inner}. error {e}')
