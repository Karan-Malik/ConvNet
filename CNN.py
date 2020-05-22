"""
@author: Karan-Malik
"""

#CNN to detect pneumonia from Chest X-rays
#Train accuracy ~ 96% and Test accuracy ~ 93%

import keras
from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,Flatten,Dropout,Dense,BatchNormalization,SpatialDropout2D

model=Sequential()

model.add(Conv2D(32,(3,3),padding='same',activation='relu',input_shape=(128,128,3)))
model.add(MaxPooling2D(2,2))
model.add(SpatialDropout2D(0.1))

model.add(Conv2D(32,(3,3),padding='same',activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(BatchNormalization())
model.add(SpatialDropout2D(0.2))

model.add(Conv2D(32,(3,3),padding='same',activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(BatchNormalization())
model.add(SpatialDropout2D(0.2))

model.add(Conv2D(32,(3,3),padding='same',activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(BatchNormalization())
model.add(SpatialDropout2D(0.3))

model.add(Conv2D(32,(3,3),padding='same',activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(BatchNormalization())
model.add(SpatialDropout2D(0.3))

model.add(Conv2D(32,(3,3),padding='same',activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(BatchNormalization())
model.add(SpatialDropout2D(0.3))

model.add(Conv2D(32,(3,3),padding='same',activation='relu'))
model.add(MaxPooling2D(2,2))
model.add(BatchNormalization())
model.add(SpatialDropout2D(0.5))

model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(output_dim=128,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(output_dim=128,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(output_dim=1,activation='sigmoid'))

adam=keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer=adam,loss='binary_crossentropy',metrics=['accuracy'])

model.summary()

from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory(
        'chest_xray/chest_xray/train',
        target_size=(128,128),
        batch_size=16 ,
        class_mode='binary')

val_set = test_datagen.flow_from_directory(
        'chest_xray/chest_xray/test',
        target_size=(128,128),
        batch_size=16,
        class_mode='binary')

model.fit_generator(
        training_set,
        steps_per_epoch=326,
        epochs=128,
        validation_data=val_set,
        validation_steps=39)


'''
#Checking for individual images

test_image = image.load_img('enter image name', target_size = (128, 128))

test_image

test_image = image.img_to_array(test_image)

test_image = np.expand_dims(test_image, axis = 0)

result = model.predict_classes(test_image)

print(result)
'''
