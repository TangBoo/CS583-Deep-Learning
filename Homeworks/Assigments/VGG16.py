from keras import models
from keras.datasets import cifar10
from keras import layers
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras.utils import to_categorical


nb_class=10
class_name = {0: 'airplane',1: 'automobile',2: 'bird',3: 'cat',4: 'deer',5: 'dog',6: 'frog',7: 'horse',8: 'ship',9: 'truck'}

(x_train,y_train),(x_test,y_test)=cifar10.load_data()
y_train=y_train.reshape(y_train.shape[0])
y_test=y_test.reshape(y_test.shape[0])
x_train=x_train.astype('float32')
x_test=x_test.astype('float32')
x_train /= 255
x_test /= 255
y_train=to_categorical(y_train,nb_class)
y_test=to_categorical(y_test,nb_class)

rand_indices = np.random.permutation(50000)
train_indices = rand_indices[0:40000]
valid_indices = rand_indices[40000:50000]

x_val = x_train[valid_indices, :]
y_val = y_train[valid_indices, :]
x_tr = x_train[train_indices, :]
y_tr = y_train[train_indices, :]

model=models.Sequential()
model.add(layers.Conv2D(64,(3,3),activation=None,strides=1,kernel_initializer='he_normal',padding='same',input_shape=(32,32,3)))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
model.add(layers.Conv2D(64,(3,3),strides=1,padding='same',activation=None,kernel_initializer='he_normal'))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(pool_size=2,strides=2,padding='valid'))


model.add(layers.Conv2D(128,(3,3),strides=1,padding='same',activation=None,kernel_initializer='he_normal'))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
model.add(layers.Conv2D(128,(3,3),strides=1,padding='same',activation=None,kernel_initializer='he_normal'))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(pool_size=2,strides=2,padding='valid'))

model.add(layers.Conv2D(256,(3,3),strides=1,padding='same',activation=None,kernel_initializer='he_normal'))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
model.add(layers.Conv2D(256,(3,3),strides=1,padding='same',activation=None,kernel_initializer='he_normal'))
model.add(layers.BatchNormalization())
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(pool_size=2,strides=2,padding='valid'))

model.add(layers.Flatten())
model.add(layers.Dense(128,activation='relu',kernel_initializer='he_normal'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(nb_class,activation='softmax',kernel_initializer='he_normal'))

model.compile(loss='categorical_crossentropy',optimizer='adadelta',metrics=['acc'])

print(model.summary())

epochs=100
batch_size=256
data_gener=ImageDataGenerator(featurewise_center=False, samplewise_center=False,featurewise_std_normalization=False,
                              samplewise_std_normalization=False, zca_whitening=False,rotation_range=0,width_shift_range=0.1,
                              height_shift_range=0.1,horizontal_flip=True,vertical_flip=False)
data_gener.fit(x_tr)
gen=data_gener.flow(x_tr,y_tr,batch_size=batch_size)

history=model.fit_generator(generator=gen,steps_per_epoch=40000//batch_size,epochs=epochs,validation_data=(x_val,y_val))
print(history)

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

loss,acc=model.evaluate(x_test,y_test,verbose=0)
print('Test acc:',acc)
print('Test loss',loss)

rand_id=np.random.choice(range(10000),size=10)
x_pre=np.array([x_test[i] for i in rand_id])
y_true=[y_test[i] for i in rand_id]
y_true=np.argmax(y_true,axis=1)
y_true=[class_name[name] for name in y_true]
y_pred=model.predict(x_pre)
y_pred=np.argmax(y_pred,axis=1)
y_pred=[class_name[index] for index in y_pred]

plt.figure(figsize=(15,7))
for i in range(10):
    plt.subplot(2,5,i+1)
    plt.imshow(x_pre[i].reshape(32,32,3),cmap='gray')
    plt.title('True: %s \n Pred: %s' % (y_true[i], y_pred[i]), size=15)
plt.show()