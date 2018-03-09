from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import *
from keras.callbacks import *
from sklearn.model_selection import train_test_split
from keras.utils import multi_gpu_model
from datetime import datetime

from densenet_nascells import create_densenet


def load_datasets():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=0)

    generator_train = ImageDataGenerator(
        featurewise_center=True,
        samplewise_center=False,
        featurewise_std_normalization=True,
        samplewise_std_normalization=False,
        zca_whitening=False,
        zca_epsilon=1e-6,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.,
        zoom_range=0.2,
        channel_shift_range=0.,
        fill_mode='nearest',
        cval=0.,
        horizontal_flip=True,
        vertical_flip=False,
        rescale=None,
        preprocessing_function=None)

    generator_test = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True)

    generator_train.fit(x_train)
    generator_test.fit(x_train)
    return ((generator_train, generator_test), (x_train, y_train), (x_test, y_test), (x_val, y_val))


def create_callbacks(max_epochs):
    cbs = []

    def learningrate_schedule(epoch, lr):
        if epoch == int(max_epochs*0.5) or epoch == int(max_epochs*0.75):
            return lr*0.1
        else:
            return lr

    run_dir = datetime.today().strftime('%Y%m%d-%H%M%S')
    cbs.append(ReduceLROnPlateau(monitor='val_loss', factor=0.5, verbose=1, patience=10))
    cbs.append(TensorBoard(log_dir='./logs/%s' % run_dir, batch_size=64))
    cbs.append(ModelCheckpoint(filepath='./weights/weights.{epoch:02d}-{val_acc:.2f}.ckpt', verbose=1, period=10))
    return cbs


def main(max_epochs=300):
    (generator_train, generator_test), (x_train, y_train), (x_test, y_test), (x_val, y_val) = load_datasets()
    model = create_densenet(
        input_shape=(32, 32, 3), dense_layers=[3,3,4], nbr_classes=10,
        weight_decay=1e-4, filters_per_channel=24, compression=0.3, dropout=0.2,
        max_filt=48, min_filt=16)
        # Erkänt bra kombinationer:
        # Dense layers, compression, max_filt, min_filt,
        # [2,2,3], 0.5, 96, 32
        # [3,3,4], 0.3, 48, 16
    optimizer = SGD(lr=0.1, momentum=0.9, nesterov=True)
    cbs = create_callbacks(max_epochs)

    print(model.summary())

    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['acc'])

    model.fit_generator(generator_train.flow(x_train, y_train, batch_size=64, seed=0),
                        callbacks=cbs, epochs=max_epochs,
                        validation_data=generator_test.flow(x_val, y_val, seed=0),
                        verbose=1)

if __name__ == '__main__':
    main()
