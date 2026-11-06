from skimage.feature import hog
import cv2
import numpy as np
import os
from sklearn.svm import LinearSVC

def extract_hog(img, size=(32, 32)):
    img = cv2.resize(img, size)
    features = hog(
        img,
        orientations=9,
        pixels_per_cell=(4, 4),
        cells_per_block=(2, 2),
        block_norm='L2-Hys'
    )
    return features


def detect_p_svm(input_path, output_path, clf):
    img = cv2.imread(input_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        if cv2.contourArea(cnt) < 100:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        roi = binary[y:y+h, x:x+w]

        hog_feat = extract_hog(roi)
        score = clf.decision_function([hog_feat])[0]
        pred = clf.predict([hog_feat])[0]

        if pred == 1 and score > 0.32:  # klasa P
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)

    cv2.imwrite(output_path, img)


if __name__ == '__main__':
    X, y = [], []

    for label, folder in enumerate(["NOT_P","P"]):
        path = os.path.join("dataset", folder)
        for fname in os.listdir(path):
            img = cv2.imread(os.path.join(path, fname), cv2.IMREAD_GRAYSCALE)
            _, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
            hog_feat = extract_hog(img)
            X.append(hog_feat)
            y.append(label)

    X = np.array(X)
    y = np.array(y)

    #clf = LinearSVC()
    clf = LinearSVC(C=0.1, class_weight='balanced', max_iter=10000)
    clf.fit(X, y)
    detect_p_svm('ulaz.png', 'izlaz.png',clf)