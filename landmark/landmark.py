import cv2
import dlib
import numpy as np

def get_landmarks(image_path):
    # dlib 얼굴 검출기와 랜드마크 예측기 로드
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

    # 이미지 로드 및 그레이스케일 변환
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 얼굴 검출
    faces = detector(gray)

    for face in faces:
        landmarks = predictor(gray, face)

        # 랜드마크 좌표 추출
        landmarks_points = []
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            landmarks_points.append((x, y))
            cv2.circle(image, (x, y), 2, (0,0,255),-1)
            
    cv2.imwrite('/home/sangwon/test.png', image)

    return landmarks_points

def calculate_distances(landmarks1, landmarks2):
    distances = []
    for i in range(len(landmarks1)):
        p1 = np.array(landmarks1[i])
        p2 = np.array(landmarks2[i])

        # 두 랜드마크 간의 유클리드 거리 계산
        distance = np.linalg.norm(p1 - p2)
        distances.append(distance)
    
    return distances

distances_cal = 0

for i in range(124):
    
    # 이미지에서 랜드마크 추출
    landmarks1 = get_landmarks(f'/home/sangwon/dataset_for_FID/comb/119.png')
    #landmarks2 = get_landmarks(f'/home/sangwon/dataset_for_FID/hr/119.png')

    # 랜드마크 간 거리 계산
    #distances = calculate_distances(landmarks1, landmarks2)
    #distances_cal += sum(distances)/len(distances)
    break
    
# 거리 출력
#print(distances_cal/(i+1))
