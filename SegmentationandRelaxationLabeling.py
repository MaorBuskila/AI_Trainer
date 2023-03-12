import cv2
import PoseModule as pm
import numpy as np
# Read the image in color (in RGB format)
blobs = cv2.imread('pic.png') #reads color image in BGR format
blobs = cv2.cvtColor(blobs, cv2.COLOR_BGR2RGB) # converts the image to RGB format

# Function that initializes the confidence arrays for iteration e
def set_initial_confidence(image, nLabels):
    initial_confidence = np.zeros((nLabels,) + image.shape).astype(float)
    image = image / 255
    for label in range(nLabels):
        if label == 0:  # "background pixel"
            initial_confidence[label][image >= 0.5] = 0.001
            initial_confidence[label][image < 0.5] = 0.999
        else:  # label is 1, "object pixel"
            initial_confidence[label][image >= 0.5] = 0.999
            initial_confidence[label][image < 0.5] = 0.001
    return initial_confidence


# The support function:
def support(comp_kernel, confidence):
    nLabels, nRows, nColumns, dc = confidence.shape
    supp = np.zeros_like(confidence)
    for label in range(nLabels):
        supp[label] = cv2.filter2D(src=confidence[label], ddepth=cv2.CV_64F, kernel=comp_kernel).astype(float)
    return supp


# The confidence update rule:
def update(curr_confidence, comp_kernel):
    nLabels, nRows, nColumns, dc = curr_confidence.shape
    res = np.zeros_like(curr_confidence)
    res_denominator = np.zeros((nRows, nColumns, 3))
    supp = support(comp_kernel, curr_confidence)
    for label in range(nLabels):
        res[label] = curr_confidence[label] * supp[label]
        res_denominator += res[label]
    res = res / res_denominator
    return res

    # The average local consistency:


def average_local_consistency(confidence, comp_kernel):
    nLabels, nRows, nColumns, dc = confidence.shape
    res_denominator = np.zeros((nRows, nColumns, 3))
    supp = support(comp_kernel, confidence)
    for label in range(nLabels):
        res_denominator += confidence[label] * supp[label]
        alc = np.sum(res_denominator)  # the average local consistency
    return alc


# Set the final assignment for each pixel as the label with the highest probability
def choose_label(confidence):
    nLabels, nRows, nColumns, dc = confidence.shape
    segmentation_array = np.zeros((nRows, nColumns))
    for i in range(nRows):
        for j in range(nColumns):
            segmentation_array[i, j] = np.argmax(confidence[:, i, j])
    return segmentation_array


def RelaxationLabeling(image, nLables, comp_kernel, epsilon):
    '''
    This function should use relaxation labeling to correctly segment the input RGB image
    into 'object' and 'background' segments.
    It should get as input an image and any additional parameters needed for the
    computation.
    It should return as output a 'segmentation_array' which is 2D (has the same number of
    rows and columns as the original image), and the value in each of its cells
    represents the label of the corresponding pixel. The value represnting each label
    should be encoded by brightness level.
    Note: you may add any additional input and\or output parameters to this algorithm,
    but you must explain each of them in the atached pdf.
    '''
    # ======= Replace with relevant code
    curr_conf = set_initial_confidence(image, nLables)
    k = 0
    while True:

        next_conf = update(curr_conf, comp_kernel)
        diff = abs(average_local_consistency(curr_conf, comp_kernel) -
                   average_local_consistency(next_conf, comp_kernel))
        curr_conf = next_conf
        k = k + 1
        if diff < epsilon:
            break

    return choose_label(next_conf)


# Use your algorithm to segment the image
size = 41, 41
kernel = np.zeros((size))
kernel[20, 20] = 1
comp_kernel = cv2.GaussianBlur(kernel, (size), 0)
segmented_image = RelaxationLabeling(blobs, nLables=2, comp_kernel=comp_kernel, epsilon=15)




# Function that initializes the confidence arrays for iteration e
def set_initial_confidence(image, nLabels):
    initial_confidence = np.zeros((nLabels,) + image.shape).astype(float)
    image = image / 255
    # The following part is relevant to this specific problem, and will need to be
    # set differently for problems where nLabels >2
    for label in range(nLabels):
        if label == 0:  # "background pixel"
            initial_confidence[label][image >= 0.5] = 0.001
            initial_confidence[label][image < 0.5] = 0.999
        else:  # label is 1, "object pixel"
            initial_confidence[label][image >= 0.5] = 0.999
            initial_confidence[label][image < 0.5] = 0.001
    return initial_confidence


# The support function:
def support(comp_kernel, confidence):
    nLabels, nRows, nColumns, dc = confidence.shape
    supp = np.zeros_like(confidence)
    for label in range(nLabels):
        supp[label] = cv2.filter2D(src=confidence[label], ddepth=cv2.CV_64F, kernel=comp_kernel).astype(np.float)
    return supp


# The confidence update rule:
def update(curr_confidence, comp_kernel):
    nLabels, nRows, nColumns, dc = curr_confidence.shape
    res = np.zeros_like(curr_confidence)
    res_denominator = np.zeros((nRows, nColumns, 3))
    supp = support(comp_kernel, curr_confidence)
    for label in range(nLabels):
        res[label] = curr_confidence[label] * supp[label]
        res_denominator += res[label]
    res = res / res_denominator
    return res

    # The average local consistency:


def average_local_consistency(confidence, comp_kernel):
    nLabels, nRows, nColumns, dc = confidence.shape
    res_denominator = np.zeros((nRows, nColumns, 3))
    supp = support(comp_kernel, confidence)
    for label in range(nLabels):
        res_denominator += confidence[label] * supp[label]
    alc = np.sum(res_denominator)  # the average local consistency
    return alc


# Set the final assignment for each pixel as the label with the highest probability
def choose_label(confidence):
    nLabels, nRows, nColumns, dc = confidence.shape
    segmentation_array = np.zeros((nRows, nColumns))
    for i in range(nRows):
        for j in range(nColumns):
            segmentation_array[i, j] = np.argmax(confidence[:, i, j])
    return segmentation_array


def RelaxationLabeling(image, nLabels, comp_kernel, epsilon):
    '''
    This function should use relaxation labeling to correctly segment the input RGB image
    into 'object' and 'background' segments.
    It should get as input an image and any additional parameters needed for the
    computation.
    It should return as output a 'segmentation_array' which is 2D (has the same number of
    rows and columns as the original image), and the value in each of its cells
    represents the label of the corresponding pixel. The value represnting each label
    should be encoded by brightness level.
    Note: you may add any additional input and\or output parameters to this algorithm,
    but you must explain each of them in the atached pdf.
    '''
    # ======= Replace with relevant code
    curr_conf = set_initial_confidence(image, nLabels)
    k = 0
    while True:
        next_conf = update(curr_conf, comp_kernel)
        diff = abs(average_local_consistency(curr_conf, comp_kernel) -
                   average_local_consistency(next_conf, comp_kernel))
        k += 1
        if diff < epsilon or k > 100:
            break
        curr_conf = next_conf
    segmentation_array = choose_label(curr_conf)
    return segmentation_array


# Use your algorithm to segment the image
size = 41, 41
kernel = np.zeros((size))
kernel[20, 20] = 1
comp_kernel = cv2.GaussianBlur(kernel, (size), 0)
segmented_image = RelaxationLabeling(blobs, nLabels=2, comp_kernel=comp_kernel, epsilon=15)

# Plots your result
cv2.imshow('Frame', segmented_image)

