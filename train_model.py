import tensorflow as tf
from sklearn. model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import  to_categorical
from sklearn.preprocessing import LabelEncoder
import pickle
import os
import cv2
import numpy as np
from sklearn import datasets
from sklearn.metrics import classification_report
from model import *


path = 'Images/'
#Đọc từng dữ liệu ảnh trong folder Images
images = []
classNo = []
folders = os.listdir(path)
for folder in folders:
	f = os.listdir(path+folder)
	for x in f:
		lisPic = os.listdir(path+folder+"/"+x)
		for img in lisPic:
			#Đọc file image với đường dẫn
			curImg = cv2.imread(path+folder+"/"+x+"/"+img)
			#Resize ảnh thành ảnh kích thước chuẩn 28
			curImg = cv2.resize(curImg, (28, 28),cv2.INTER_AREA)
			# đổi ảnh thành ảnh màu xám để xử lý dữ liệu
			curImg = cv2.cvtColor(curImg,cv2.COLOR_BGR2GRAY)
			# Format hình ảnh alphabet chuẩn theo threshhold với nền trắng và dữ liệu đen
			if folder == "character":
				curImg = cv2.threshold(curImg, 0, 255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
			
			# Thêm ảnh vào mảng
			images.append(curImg)
			# Gán label ảnh đó theo đúng tên kí tự của folder ảnh đó
			classNo.append(x)
		
    	
#Đổi ảnh vs label thành dạng numpy
images = np.array(images)
classNo = np.array(classNo)

print(len(images))
print(len(classNo))

#fit label ảnh thành dạng số
le = LabelEncoder()
label = classNo
classNo = le.fit_transform(classNo)

#tách dữ liệu train 80%, test 20%
X_train,X_test, y_train,y_test = train_test_split(images,classNo, test_size=0.2)

#tách tiếp dữ liệu train thành 80%, validation 20% (train 70%, validation 10%, test 20%)
X_train,X_validation, y_train,y_validation = train_test_split(X_train,y_train, test_size=0.1)

#Xử lý dùng hàm equalizeHistogram để cân bằng hình ảnh không quá sáng hoặc quá tối
def preProcessing(img):
    img = cv2.equalizeHist(img)
    img = img/255
    return img

#lấy các ảnh của 3 tập dữ liệu để xử lý cân bằng hình ảnh
X_train= np.array(list(map(preProcessing, X_train)))
X_test= np.array(list (map(preProcessing,X_test)))
X_validation= np.array(list(map(preProcessing, X_validation)))


#REshape 3 tập dữ liệu thành đúng dạng tổng size dữ liệu, size ảnh
X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],X_train.shape[2],1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1],X_test.shape[2],1)
X_validation = X_validation.reshape(X_validation.shape[0],X_validation.shape[1],
                                    X_validation.shape[2],1)

print(X_validation.shape)

#dùng ImageDataGenerator để tăng thêm dữ liệu của ảnh, xoay ảnh, tăng kích thước ảnh, zoom ảnh,...
#để đa dạng dữ liệu tránh overfitting
dataGen = ImageDataGenerator(width_shift_range=0.1,
                              height_shift_range=0.1,
                              zoom_range=0.1,
                              shear_range=0.1,
                              rotation_range=10,
                              fill_mode="nearest",
                              horizontal_flip=True)
#fit dataGen vào tập X_train
dataGen.fit(X_train)

#categorical 3 tập label 
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
y_validation = to_categorical(y_validation)

#lấy model từ file mode
model = myModel(36)
#in ra thông số của model
print (model.summary())

batchSizeVal= 20
epochsVal = 10
stepsPerEpochVal = 100
#train model với tập X_train vs y_train với fit với tập validation 
history = model.fit_generator(dataGen.flow(X_train,y_train,
                                  batch_size=batchSizeVal),
                                  steps_per_epoch=stepsPerEpochVal,
                                  epochs=epochsVal,
                                  validation_data=(X_validation, y_validation),
                                  shuffle=1)

#Dự đoán mô hình với tập test
score = model.evaluate(X_test, y_test, verbose=0) 
print('Test Score = ',score[0]) 
print('Test Accuracy =', score[1])
#Độ chính xác mô hình
#Lưu mô hình dạng h5 để sử dụng, ko cần phải train lại tốn thời gian	
model.save("my_Model.h5")#!/usr/bin/env python