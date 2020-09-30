from keras.layers import Conv2D, Flatten, Dense, Dropout, BatchNormalization
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from keras import Sequential
from keras.preprocessing.image import ImageDataGenerator
from dataset import get_dataset

# автор архитектуры сети https://www.kaggle.com/cdeotte/how-to-choose-cnn-architecture-mnist
model = Sequential()

model.add(Conv2D(32,kernel_size=3,activation='relu',input_shape=(28, 28,1)))
model.add(BatchNormalization())
model.add(Conv2D(32,kernel_size=3,activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(32,kernel_size=5,strides=2,padding='same',activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Conv2D(64,kernel_size=3,activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(64,kernel_size=3,activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(64,kernel_size=5,strides=2,padding='same',activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.4))
model.add(Dense(14, activation='softmax'))


model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])


(train_images, train_labels), (test_images, test_labels) = get_dataset()

train_images = train_images.reshape(-1,28,28,1)
test_images = test_images.reshape(-1,28,28,1)

train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=0.333)

train_datagen = ImageDataGenerator(width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   zoom_range=0.1,
                                   rescale=1./255)

val_datagen = ImageDataGenerator(rescale=1./255)

es = EarlyStopping(monitor='val_accuracy', restore_best_weights=True, patience=3, min_delta=0.001)

batch_size = 64
model.fit(train_datagen.flow(train_images, train_labels, batch_size=batch_size),
          steps_per_epoch=len(train_images) / batch_size,
          validation_data=val_datagen.flow(val_images, val_labels),
          validation_steps=len(val_images) / batch_size,
          callbacks=[es],
          epochs=20)
model.evaluate(val_datagen.flow(test_images, test_labels))

model.save('model')