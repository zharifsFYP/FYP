import cv2
import numpy as np

def normalizationStep(image):#need to remember if i want to normalizaed before loading or not maybe no need nice jic
    normedImage = image.astype(np.float32)/255
    return normedImage

def resizedImages(image,sizeX,sizeY):
    resizedImage = cv2.resize(image,(sizeX,sizeY), interpolation=cv2.INTER_CUBIC)
    return resizedImage

def denoisedImages(image):
    denoisedImage = cv2.bilateralFilter(image, 9, 75, 75)#fyi arguments are range of  pixel, influence , space apart need to mess around with this more
    return denoisedImage

def gammaTransformP1(image,gamma): #note for me gamma >1 darker , < 1 brighter
    invertGamma = 1/gamma
    table = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        table[i] = np.clip(((i / 255.0) ** invertGamma) * 255, 0, 255)
    return cv2.LUT(image, table)

def gammaTransformP2(image,alpha): #used sparring;y fpr lowlight need to implement method to selective use case
    mean_intensity = np.mean(image)
    gamma = alpha * mean_intensity
    return gammaTransformP1(image, gamma)
    

def preprocessingPipeline(image, width, height, gamma=None):
    processed = resizedImages(image, width, height)
    processed = denoisedImages(processed)
    if gamma is not None:
        processed = gammaTransformP1(processed, gamma)
    processed = normalizationStep(processed)
    return processed


