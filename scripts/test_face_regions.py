import os
import sys
import cv2
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from skin_analysis import detect_face_landmarks, extract_face_regions

img_path = os.path.join('SKIN_DATA','1','front.jpg')
img = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)
if img is None:
    raise RuntimeError('Failed to load image: ' + img_path)

lm = detect_face_landmarks(img)
if lm is None:
    raise RuntimeError('Failed to detect face landmarks')

regions = extract_face_regions(img, lm)
print('regions:', list(regions.keys()))
for k, v in regions.items():
    print(k, v['bbox'])

# visualize and save
vis = img.copy()
colors = {
    'forehead': (0, 255, 0),
    'left_cheek': (255, 0, 0),
    'right_cheek': (0, 0, 255),
    'nose': (0, 255, 255),
    'mouth_area': (255, 0, 255),
    'chin': (255, 255, 0),
    'left_under_eye': (128, 0, 255),
    'right_under_eye': (128, 0, 255)
}
for name, data in regions.items():
    x, y, w, h = data['bbox']
    color = colors.get(name, (255,255,255))
    cv2.rectangle(vis, (x, y), (x+w, y+h), color, 2)

out_path = os.path.join('docs','output','face_regions_test.jpg')
os.makedirs(os.path.dirname(out_path), exist_ok=True)
cv2.imwrite(out_path, vis)
print('Saved:', out_path)
