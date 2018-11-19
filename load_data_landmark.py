import sys
import numpy as np
#from scipy.misc import imread
import cv2
import math
import pickle
import os
import matplotlib.pyplot as plt
import argparse
"""Script to preprocess the landmark dataset and pickle it into an array that's easy
    to index my character type"""

parser = argparse.ArgumentParser()
parser.add_argument("--path",help="Path where omniglot folder resides")
parser.add_argument("--save", help = "Path to pickle data to.", default=os.getcwd())
args = parser.parse_args()
data_path = args.path
train_folder = os.path.join(data_path, 'train')
val_folder = os.path.join(data_path, 'val')

save_path = args.save
image_per_landmark = 20

landmark_dict = {}

def distort_img(image):
    '''
    apply gaussian noise, ratation, crop to distort a image
    :param image:
    :return: distorted image
    '''
    # add gaussian noise
    offset = np.random.randint(-30, 30)
    scale = np.random.normal(1.0, 0.1)
    image = scale*(image + offset)
    row, col, channel = image.shape
    random_noise = np.random.normal(0, 10, [row, col])
    random_noise = random_noise[:, :, np.newaxis]
    image += random_noise

    # rotation
    angle = np.random.choice([-10, -5, 0, 5, 10])
    #radius = angle/360*2*np.pi
    M_rotation = cv2.getRotationMatrix2D((col / 2, row / 2), angle, 1)
    img_rotated = cv2.warpAffine(image, M_rotation, (col, row))

    # crop image to get rid of the black areas after rotation
    sintheta = math.sin(float(abs(angle))/360*2*np.pi)
    costheta = math.cos(float(abs(angle))/360*2*np.pi)
    crop_length = int(row/(sintheta+costheta))
    img_crop = img_rotated[int((row-crop_length)/2):int((row+crop_length)/2),
               int((col-crop_length)/2):int((col+crop_length)/2)]

    img_crop = cv2.resize(img_crop, (row, col), )


    return img_crop




def loadimgs(path,n=0):
    #if data not already unzipped, unzip it.

    X=[]
    y = []
    #category_images = []
    category_dict = {}
    curr_y = n #each different letter has defferent y as label
    #we load every alphabet seperately so we can isolate them later
    for category in os.listdir(path):
        print("loading category: " + category)
        category_dict[category] = [curr_y, None]
        landmark_path = os.path.join(path, category)
        #every letter/category has it's own column in the array, so  load seperately
        for landmark in os.listdir(landmark_path):
            category_images = []
            image_path = os.path.join(landmark_path, landmark)
            for image_name in os.listdir(image_path):
                image = cv2.imread(os.path.join(image_path, image_name), cv2.IMREAD_GRAYSCALE)
                image = cv2.resize(image, (105, 105))
                category_images.append(image) #shape of category_imagws for omnilogts: 20*105*105*1
                y.append(curr_y)
            try:
                X.append(np.stack(category_images))
            #edge case  - last one
            except ValueError as e:
                print(e)
                print("error - category_images:", category_images)
                continue
            curr_y += 1
            category_dict[category][1] = curr_y - 1
    y = np.vstack(y)
    X = np.stack(X)
    return X, y, landmark_dict


if __name__ == '__main__':

    X,y,c=loadimgs(train_folder) #shape of X: C*N*w*h*1. C:classes of letters; N:number of letters per class
    with open(os.path.join(save_path,"train_landmark.pickle"), "wb") as f:
        pickle.dump((X, c), f)


    X,y,c=loadimgs(val_folder)
    with open(os.path.join(save_path,"val_landmark.pickle"), "wb") as f:
        pickle.dump((X, c), f)


    #
    # train_list = os.listdir(os.path.join(train_folder, 'complex_nature'))
    # for image_name in train_list:
    #     if image_name[-3:] == 'bmp':
    #         image_id = image_name[:-4]
    #         image_dir = os.path.join(train_folder, 'complex_nature',image_id)
    #         if not os.path.isdir(image_dir):
    #             os.makedirs(image_dir)
    #         image = cv2.imread(os.path.join(train_folder, 'complex_nature', image_name))
    #         image = cv2.resize(image, (105, 105), interpolation=cv2.INTER_CUBIC)
    #         image = np.mean(image, -1, keepdims=True)
    #         for i in range(image_per_landmark):
    #             distorted_img = distort_img(image)
    #             cv2.imwrite(os.path.join(image_dir, image_id)+'_%02d.jpg'%i, distorted_img)




