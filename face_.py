import cv2
import dlib
import numpy as np

# 加载人脸检测器
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# 加载Dlib的人脸检测器
detector = dlib.get_frontal_face_detector()

# 加载Dlib的人脸特征提取器
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# 提取人脸特征
def extract_features(face):
    rect = detector(face)[0]
    shape = predictor(face, rect)
    features = []
    for i in range(68):
        features.append((shape.part(i).x, shape.part(i).y))
    return features

# 计算欧式距离
def calculate_euclidean_distance(features1, features2):
    distance = 0
    for i in range(len(features1)):
        distance += (features1[i][0] - features2[i][0])**2 + (features1[i][1] - features2[i][1])**2
    return np.sqrt(distance)

# 计算余弦相似度
def calculate_cosine_similarity(features1, features2):
    dot_product = np.dot(features1, features2)
    norm1 = np.linalg.norm(features1)
    norm2 = np.linalg.norm(features2)
    return dot_product / (norm1 * norm2)

# 加载两张人脸图片
img1 = cv2.imread('face1.jpg')
img2 = cv2.imread('face2.jpg')

# 将图片转换为灰度图
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# 检测人脸
faces1 = face_cascade.detectMultiScale(gray1, scaleFactor=1.1, minNeighbors=5)
faces2 = face_cascade.detectMultiScale(gray2, scaleFactor=1.1, minNeighbors=5)

# 提取人脸区域
face1 = gray1[faces1[0][1]:faces1[0][1]+faces1[0][3], faces1[0][0]:faces1[0][0]+faces1[0][2]]
face2 = gray2[faces2[0][1]:faces2[0][1]+faces2[0][3], faces2[0][0]:faces2[0][0]+faces2[0][2]]

# 提取人脸特征
features1 = extract_features(face1)
features2 = extract_features(face2)

# 计算欧式距离
distance = calculate_euclidean_distance(features1, features2)

# 计算余弦相似度
similarity = calculate_cosine_similarity(features1, features2)

print("欧式距离：", distance)
print("余弦相似度：", similarity)








