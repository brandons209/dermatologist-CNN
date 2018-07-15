#general imports:
import numpy as np
import sys
import time

#pre-processing
from keras.preprocessing.image import ImageDataGenerator
import preprocess_data as preprocess

#model imports
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Model
from keras.applications.inception_v3 import InceptionV3
from keras import backend as K

#training
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras import optimizers as opt

#simple function to print trainable and nontrainable parameters
def print_trainable_params():
    trainable_count = int(np.sum([K.count_params(p) for p in set(model.trainable_weights)]))
    non_trainable_count = int(np.sum([K.count_params(p) for p in set(model.non_trainable_weights)]))

    print("Trainable params: {:,}".format(trainable_count))
    print("Non-trainable params: {:,}".format(non_trainable_count))
    print("Total params: {:,}".format(trainable_count + non_trainable_count))

answer = input("Would you like to pre-process the data? If this is your first time running this model, choose y. (y or n)\n")

if answer == 'y':
    #load in image paths and labels
    print("Loading in image paths and their labels...")
    test_images, test_targets = preprocess.load_image_path_set('test')
    valid_images, valid_targets = preprocess.load_image_path_set('valid')
    train_images, train_targets = preprocess.load_image_path_set('train')

    #load image values from paths, then convert img values 0-255 to values between 0 and 1
    print("Loading test images...")
    test_images = preprocess.load_images(test_images).astype('float32')/255
    print("Loading validation images...")
    valid_images = preprocess.load_images(valid_images).astype('float32')/255
    print("Loading train images...")
    train_images = preprocess.load_images(train_images).astype('float32')/255

    print("Saving processed data...")
    preprocess.save_images_array(test_images, 'test_images.npy')
    preprocess.save_images_array(test_targets, 'test_targets.npy')
    preprocess.save_images_array(valid_images, 'valid_images.npy')
    preprocess.save_images_array(valid_targets, 'valid_targets.npy')
    preprocess.save_images_array(train_images, 'train_images.npy')
    preprocess.save_images_array(train_targets, 'train_targets.npy')

elif answer == 'n':
    print("Loading data from numpy files...")
    test_images, test_targets = preprocess.load_images_and_labels_array('test')
    valid_images, valid_targets = preprocess.load_images_and_labels_array('valid')
    train_images, train_targets = preprocess.load_images_and_labels_array('train')
else:
    print("Please choose y or n")
    sys.exit(1)

print("There are {} training images, {} validation images, and {} testing images.".format(len(train_images), len(valid_images), len(test_images)))
#augument data by creating rotated versions of images, no need for translation or anything
#like that since images are taken at the same distance away with the lesion in the center
aug_gen = ImageDataGenerator(rotation_range=180)
aug_gen.fit(train_images)

aug_gen_valid = ImageDataGenerator(rotation_range=180)
aug_gen_valid.fit(valid_images)

"""
This model will use the inceptionv3 network trained on imagenet, with fine tuning the weights of inception, and my own classifer fc layers on the end.
"""
inception_transferred = InceptionV3(weights='imagenet', include_top=False)

#start with the output of the inception network and a global pooling layer:
#have to do this with Model library from keras, not Sequential. this is because we need to merge two networks into one.
model_builder = inception_transferred.output
model_builder = GlobalAveragePooling2D()(model_builder)

#classifier fully connected layers:
model_builder = Dense(328, activation='relu')(model_builder)
model_builder = Dropout(0.4)(model_builder)
model_builder = Dense(150, activation='relu')(model_builder)
model_builder = Dropout(0.3)(model_builder)

#predictions layer, 3 nodes for our 3 classes:
model_builder = Dense(3, activation='softmax')(model_builder)

#bring it together now:
model = Model(inputs=inception_transferred.inputs, outputs=model_builder)

#first train top layers we created only, since we just created the weights. so freeze inceptionv3 layers:
for layer in inception_transferred.layers:
    layer.trainable = False

#### Train classifier
#compile model using rmsprop optimizer, may try adagrad and adam.
model.compile(optimizer=opt.RMSprop(lr=0.00001), loss='categorical_crossentropy', metrics=['accuracy'])
print_trainable_params()

#train top layers of model using augmented data:
epochs = 20
batch_size = 5

input("Press enter to continue training top layers of model...")
checkpointer = ModelCheckpoint(filepath='saved_weights/inception.model.top.layers.only.best.hdf5', verbose=1, save_best_only=True)
#can run tensorboard --logdir=tensorboard_logs/ to be able to see logs. each run is stored in a folder named for its time ran, so you can compare multiple runs!
#TODO: fix issue with trying to display histograms of weights, where gradients return as None when tensorboard trys to calculate them. disabling tensorboard for now
start_time = time.strftime("%a_%b_%d_%Y_%H:%M", time.localtime())
ten_board = TensorBoard(log_dir='tensorboard_logs/{}_classifier'.format(start_time), write_images=True)

#reduce lr on plateau can allow better training results by dynamically reducing the learning rate when val loss does not improve
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, min_lr=0.00000000001)
model.fit_generator(aug_gen.flow(train_images, train_targets, batch_size=batch_size),
                    steps_per_epoch=train_images.shape[0] // batch_size,
                    epochs=epochs, verbose=1, callbacks=[checkpointer, reduce_lr, ten_board],
                    validation_data=aug_gen_valid.flow(valid_images, valid_targets, batch_size=batch_size),
                    validation_steps=valid_images.shape[0] // batch_size,
                    use_multiprocessing=True)

#load weights with lowest validation loss:
model.load_weights('saved_weights/inception.model.top.layers.only.best.hdf5')

#inception network has 310 layers, so instead of fine tuning the whole network, can fine tune the top two inception blocks, which is layers 250-310
#freeze top layers weights and unfreeze inception network's weights to fine tune the model:
for layer in inception_transferred.layers:
    layer.trainable = True
for layer in model.layers[311:]:
    layer.trainable = False

### Fine tune model
#compile model using rmsprop optimizer, may try different learnrate.
model.compile(optimizer=opt.RMSprop(lr=0.00001), loss='categorical_crossentropy', metrics=['accuracy'])
print_trainable_params()

#train top layers of model using augmented data:
epochs = 20
batch_size = 5
input("Press enter to continue training inception layers of model...")
checkpointer = ModelCheckpoint(filepath='saved_weights/inception.model.inception.layers.only.best.hdf5', verbose=1, save_best_only=True)
ten_board = TensorBoard(log_dir='tensorboard_logs/{}_inception'.format(start_time), write_images=True)
model.fit_generator(aug_gen.flow(train_images, train_targets, batch_size=batch_size),
                    steps_per_epoch=train_images.shape[0] // batch_size,
                    epochs=epochs, verbose=1, callbacks=[checkpointer, reduce_lr, ten_board],
                    validation_data=aug_gen_valid.flow(valid_images, valid_targets, batch_size=batch_size),
                    validation_steps=valid_images.shape[0] // batch_size,
                    use_multiprocessing=True)

#load weights with lowest validation loss:
model.load_weights('saved_weights/inception.model.inception.layers.only.best.hdf5')

#test accuracy of model:
print("Testing accuracy of model...")
evaluation = model.evaluate(test_images, test_targets)
print("Test accuracy is: {:.2f} and test loss is: {:.4f}".format(evaluation[1], evaluation[0]))

input("Press enter to save model...")
model.save('saved_models/skin_cancer_full_model.h5')
