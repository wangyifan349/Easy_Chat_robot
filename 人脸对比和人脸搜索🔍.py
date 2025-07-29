"""
本程序基于 Flask、OpenCV 和 dlib，提供两个 RESTful 接口：/compare 用于上传两张人脸图片并返回欧式距离（L2 距离）与余弦相似度；/search 用于上传一张人脸图片并在本地图库中查找最相似的若干张人脸，返回文件名、L2 距离和余弦相似度。
通过 Haarcascade 和 dlib 68 点模型完成人脸检测与特征提取，简单易用，适合集成到人脸比对或检索应用中。
This application uses Flask, OpenCV and dlib to expose two RESTful endpoints: `/compare` accepts two face images and returns their Euclidean distance (L2) and cosine similarity; `/search` accepts one face image and finds the top matching faces in a local gallery, returning filenames along with their L2 distances and cosine similarities. 
Faces are detected and described via Haarcascade and a dlib 68-point landmark model. It is lightweight and ready to integrate into face comparison or retrieval systems.
"""

"""
1. 人脸对比（/compare）
–– 使用 curl ––  
curl -X POST http://127.0.0.1:5000/compare \
  -F "face1=@/path/to/face1.jpg" \
  -F "face2=@/path/to/face2.jpg"
–– 使用 wget ––  
wget --method=POST \
     --header="Content-Type: multipart/form-data" \
     --body-file=<(printf '%s\n' \
       --form "face1=@/path/to/face1.jpg" \
       --form "face2=@/path/to/face2.jpg") \
     -O - \
     http://127.0.0.1:5000/compare
2. 人脸搜索（/search）
–– 使用 curl ––  
curl -X POST http://127.0.0.1:5000/search \
  -F "face=@/path/to/query.jpg" \
  -F "top_k=3"
–– 使用 wget ––  
wget --method=POST \
     --header="Content-Type: multipart/form-data" \
     --body-file=<(printf '%s\n' \
       --form "face=@/path/to/query.jpg" \
       --form "top_k=3") \
     -O - \
     http://127.0.0.1:5000/search
说明：
- 将 `/path/to/...` 替换为本地图片路径。  
- 对比接口返回 JSON 包含 `l2_distance` 和 `cosine_similarity`。  
- 搜索接口返回 JSON 包含最相似的文件列表及对应指标。
"""
from flask import Flask, request, jsonify
import os
import cv2
import dlib
import numpy as np
from werkzeug.utils import secure_filename

# 配置项
UPLOAD_FOLDER = 'uploads'
GALLERY_FOLDER = 'gallery'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
TOP_K_DEFAULT = 5

# 初始化 Flask 应用
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# 创建目录
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(GALLERY_FOLDER):
    os.makedirs(GALLERY_FOLDER)

# 加载人脸检测模型
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
dlib_detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# 工具：检查文件后缀
def allowed_file(filename):
    if '.' not in filename:
        return False
    ext = filename.rsplit('.', 1)[1].lower()
    if ext in ALLOWED_EXTENSIONS:
        return True
    return False

# 工具：检测人脸并提取 68 点特征，返回 numpy 数组 68×2
def detect_and_extract(gray):
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
    if len(faces) == 0:
        raise RuntimeError('Haar 检测未找到人脸')
    x, y, w, h = faces[0]
    roi = gray[y:y+h, x:x+w]
    dets = dlib_detector(roi, 1)
    if len(dets) == 0:
        raise RuntimeError('dlib 检测未找到人脸')
    shape = shape_predictor(roi, dets[0])
    coords = np.zeros((68, 2), dtype=np.float32)
    for i in range(68):
        coords[i][0] = shape.part(i).x
        coords[i][1] = shape.part(i).y
    return coords

# 工具：计算 L2 距离
def compute_l2_distance(a, b):
    diff = a - b
    norm = np.linalg.norm(diff)
    return float(norm)

# 工具：计算余弦相似度
def compute_cosine_similarity(a, b):
    v1 = a.flatten()
    v2 = b.flatten()
    dot = float(np.dot(v1, v2))
    norm1 = float(np.linalg.norm(v1))
    norm2 = float(np.linalg.norm(v2))
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0
    return dot / (norm1 * norm2)

# 工具：从文件路径提取特征
def extract_feature_from_file(path):
    image = cv2.imread(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feat = detect_and_extract(gray)
    return feat

# 接口：对比两张人脸
@app.route('/compare', methods=['POST'])
def compare_faces():
    if 'face1' not in request.files or 'face2' not in request.files:
        return jsonify({'error': '必须上传 face1 和 face2'}), 400

    file1 = request.files['face1']
    file2 = request.files['face2']

    if not allowed_file(file1.filename) or not allowed_file(file2.filename):
        return jsonify({'error': '仅支持 png, jpg, jpeg'}), 400

    name1 = secure_filename(file1.filename)
    name2 = secure_filename(file2.filename)
    path1 = os.path.join(UPLOAD_FOLDER, name1)
    path2 = os.path.join(UPLOAD_FOLDER, name2)
    file1.save(path1)
    file2.save(path2)

    try:
        img1 = cv2.imread(path1)
        img2 = cv2.imread(path2)
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        feat1 = detect_and_extract(gray1)
        feat2 = detect_and_extract(gray2)
        l2 = compute_l2_distance(feat1, feat2)
        cos = compute_cosine_similarity(feat1, feat2)
    except Exception as e:
        if os.path.exists(path1):
            os.remove(path1)
        if os.path.exists(path2):
            os.remove(path2)
        return jsonify({'error': str(e)}), 500

    if os.path.exists(path1):
        os.remove(path1)
    if os.path.exists(path2):
        os.remove(path2)

    return jsonify({'l2_distance': l2, 'cosine_similarity': cos})

# 接口：在本地目录搜索最相似人脸
@app.route('/search', methods=['POST'])
def search_gallery():
    if 'face' not in request.files:
        return jsonify({'error': '必须上传 face'}), 400

    file = request.files['face']
    if not allowed_file(file.filename):
        return jsonify({'error': '仅支持 png, jpg, jpeg'}), 400

    top_k = TOP_K_DEFAULT
    top_k_str = request.form.get('top_k')
    try:
        if top_k_str is not None:
            tk = int(top_k_str)
            if tk > 0:
                top_k = tk
    except:
        top_k = TOP_K_DEFAULT

    name = secure_filename(file.filename)
    query_path = os.path.join(UPLOAD_FOLDER, name)
    file.save(query_path)

    try:
        query_img = cv2.imread(query_path)
        query_gray = cv2.cvtColor(query_img, cv2.COLOR_BGR2GRAY)
        query_feat = detect_and_extract(query_gray)
    except Exception as e:
        if os.path.exists(query_path):
            os.remove(query_path)
        return jsonify({'error': str(e)}), 500

    if os.path.exists(query_path):
        os.remove(query_path)

    results = []

    gallery_files = os.listdir(GALLERY_FOLDER)
    for fname in gallery_files:
        if not allowed_file(fname):
            continue
        full_path = os.path.join(GALLERY_FOLDER, fname)
        try:
            feat = extract_feature_from_file(full_path)
            l2 = compute_l2_distance(query_feat, feat)
            cos = compute_cosine_similarity(query_feat, feat)
            record = {
                'file': fname,
                'l2_distance': l2,
                'cosine_similarity': cos
            }
            results.append(record)
        except:
            pass

    if len(results) == 0:
        return jsonify({'error': '图库中未找到有效人脸'}), 404

    # 排序：先按 L2 升序，再按余弦相似度降序
    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            a = results[i]
            b = results[j]
            if a['l2_distance'] > b['l2_distance'] or (a['l2_distance'] == b['l2_distance'] and a['cosine_similarity'] < b['cosine_similarity']):
                temp = results[i]
                results[i] = results[j]
                results[j] = temp

    if len(results) > top_k:
        trimmed = []
        count = 0
        for item in results:
            if count >= top_k:
                break
            trimmed.append(item)
            count += 1
        results = trimmed

    return jsonify({
        'query': name,
        'top_k': len(results),
        'results': results
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
