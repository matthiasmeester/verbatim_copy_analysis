import cv2
import numpy as np


# from: https://stackoverflow.com/questions/60272082/how-to-segment-similar-looking-areas-color-wise-inside-a-image-that-belong-to

# Kmeans color segmentation
def kmeans_color_quantization(image, clusters=8, rounds=1):
    h, w = image.shape[:2]
    samples = np.zeros([h * w, 3], dtype=np.float32)
    count = 0

    for x in range(h):
        for y in range(w):
            samples[count] = image[x][y]
            count += 1

    compactness, labels, centers = cv2.kmeans(samples,
                                              clusters,
                                              None,
                                              (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10000, 0.0001),
                                              rounds,
                                              cv2.KMEANS_RANDOM_CENTERS)

    centers = np.uint8(centers)
    res = centers[labels.flatten()]
    return res.reshape((image.shape))


similarity_map = np.loadtxt('similarity_map.txt') * 255
similarity_map = similarity_map.astype('uint8')
print(similarity_map)
# Load original image
original = cv2.cvtColor(similarity_map, cv2.COLOR_GRAY2BGR)
print('Image Dimensions :', original.shape)

# Perform kmeans color segmentation
kmeans = kmeans_color_quantization(original, clusters=5)
cv2.imshow('similarity_map', original)

# Color threshold on kmeans image
hsv = cv2.cvtColor(kmeans, cv2.COLOR_BGR2HSV)
lower_bound = 0.7 * 255
lower = np.array([lower_bound, lower_bound, lower_bound])
upper = np.array([255, 255, 255])
mask = cv2.inRange(hsv, 150, 255)

# Apply mask onto original image
result = cv2.bitwise_and(original, original, mask=mask)
result[mask == 0] = (255, 255, 255)

# Display
cv2.imshow('kmeans', kmeans)
cv2.imshow('result', result)
cv2.imshow('mask', mask)
cv2.waitKey()
