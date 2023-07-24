from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import tensorflow as tf
import cv2
def myModel(noofClasses):

  #Nhận input đầu vào gồm số các label khác nhau
  noOfFilters= 50
  sizeOfFilter1=(5,5)
  sizeOfFilter2= (3,3)
  sizeOfPool= (2,2)
  noOfNode=512
  #Số node của layer Dense
  # sử dụng mô hình Sequential
  model = Sequential()
  model.add((Conv2D(noOfFilters, sizeOfFilter1, 
                       input_shape=(28,28,1),activation='relu')))
  #Đầu vào dữ liệu có dạng độ lớn 28
  #xài các hàm hoạt động relu
  model.add((Conv2D(noOfFilters, sizeOfFilter1, activation='relu')))
  model.add(MaxPooling2D(pool_size=sizeOfPool))
  model.add((Conv2D(noOfFilters//2, sizeOfFilter2, activation='relu')))
  model.add((Conv2D(noOfFilters//2, sizeOfFilter2, activation='relu')))
  model.add(MaxPooling2D(pool_size=sizeOfPool))
  model.add(Dropout(0.25))
  #Dropout dữ liệu 0.25

  model.add(Flatten())
  model.add(Dense(noOfNode, activation='relu'))
  model.add(Dropout (0.25))
  #Số các label đầu ra khác nhau = noofClasses
  #đầu ra sử dụng activation softmax
  model.add(Dense(noofClasses, activation='softmax'))
  model.compile(Adam(lr=0.001),loss='categorical_crossentropy',metrics=['accuracy'])

  #đầu ra là mô hình đã khởi tạo
  return model

