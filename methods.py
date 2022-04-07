import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_lfw_people
from scipy.fftpack import dct
import cv2
from collections import Counter

def distance(el1, el2):
	return np.linalg.norm(np.array(el1) - np.array(el2))

def get_histogram(image):
    hist, bins = np.histogram(image.ravel(), bins=11)
    return hist

def get_dct(image, mat_side = 5):
    c = dct(image, axis=1)
    c = dct(c, axis=0)
    c = c[0:mat_side, 0:mat_side]
    return c

def get_gradient(image, n = 2):
    shape = image.shape[0]
    i, l = 0, 0
    r = n
    result = []

    while r <= shape:
        window = image[l:r, :]
        result.append(np.sum(window))
        i += 1
        l = i * n
        r = (i + 1) * n
    result = np.array(result)
    return result

def get_scale(image, scale = 6):
	h = image.shape[0]
	w = image.shape[1]
	new_size = (int(h * scale), int(w * scale))
	return cv2.resize(image, new_size, interpolation = cv2.INTER_AREA)

def get_dft(image, p = 32):
    f = np.fft.fft2(image)
    f = np.abs(f[0:p, 0:p])
    return f

def preparing_data(n_for_test):# shuffle=True
    faces = fetch_lfw_people(min_faces_per_person=100, resize=0.4)
    X = faces.images
    y = faces.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=n_for_test, random_state=42
    )
    return X_train, X_test, y_train, y_test


def classif(result, data, true_target, targets, method):
    i=1
    min = 100000
    ind = -1
    while i < 500:
        i = i + 1
        result2 = method(data[i])
        if (min > distance(result, result2)):
            min = distance(result, result2)
            ind = i
    return targets[ind]

def classif_parallel(result, data, true_target, targets):
   i=1
   min = [1000000]*5
   ind = [-1]*5
   result2 = [0]*5
   predicted_targets = [-1]*5
   while i < 500:
       i = i + 1
       result2[0] = get_histogram(data[i])
       result2[1] = get_dft(data[i])
       result2[2] = get_dct(data[i])
       result2[3] = get_scale(data[i])
       result2[4] = get_gradient(data[i])
       for j in range(5):
        if (min[j] > distance(result[j], result2[j])):
            min[j] = distance(result[j], result2[j])
            ind[j] = i

   for j in range(5):
    predicted_targets[j] = targets[ind[j]]

   c = Counter(predicted_targets)
   final_predicted_target = c.most_common(1)[0][0]

   return final_predicted_target


