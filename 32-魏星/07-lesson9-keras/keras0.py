

'''

'''
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print('x_train.shape = ', x_train.shape)
print('y_train = ', y_train)
print('x_test.shape = ', x_test.shape)
print('y_test', y_test)


# digit_train = x_train[0]
# digit_test = x_test[0]
# import matplotlib.pyplot as plt
# plt.imshow(digit_train)
# plt.imshow(digit_test)
# plt.show()

from keras import models
from keras import layers

network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28*28,)))
network.add(layers.Dense(10, activation='softmax'))

network.compile(optimizer='rmsprop', loss='categorical_crossentropy',
               metrics=['accuracy'])


x_train = x_train.reshape((60000, 28*28))
x_train = x_train.astype('float32') / 255

x_test = x_test.reshape((10000, 28*28))
x_test = x_test.astype('float32') / 255


from keras.utils import to_categorical
print("before change:", y_train[0])
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print("after change: ", y_test[0])


network.fit(x_train, y_train, epochs=5, batch_size=128)


test_loss, test_acc = network.evaluate(x_test, y_test, verbose=1)
print(test_loss)
print('test_acc', test_acc)


(train_images, train_labels),(test_images,test_labels) = mnist.load_data()
test_images = test_images.reshape((10000, 28*28))
res = network.predict(test_images)
print(res[0])
print(res[0].shape)

for i in range(res[0].shape[0]):
    if (res[0][i] == 1):
        print("the number for the picture is : ", i)
        break






