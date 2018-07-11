# Dermatologist AI
I made this project based on the project idea in udacity's deep learning nanodegree, however I heavily expanded on the problem.

[LICENSE](LICENSE)
### Objective
The objective of this model is to determine if a supplied image of a skin aliment is either melanoma, nevus, or seborrheic keratosis based on nothing but the image itself.

### Libraries Used
I used Keras with the Tensorflow back end to train this model. I also use image processing tools from Keras for processing the data. And of course numpy. **Make sure to install the
required libraries in requirements.txt!**

### Performance
Right now, by fine tuning the weights of Google's InceptionV3 network with my own classifier fully connected layers, and training on 2500 images, I am able to get about 60 percent accuracy. With this, I also augmented my data by rotation images by varying degrees using Keras' ImageDataGenerator. However, validation loss seems to stop decreasing after around 10-15 epochs, so with more fine tuning of the parameters and the fully connected classifier layers, I can get the accuracy closer to 70 percent.

I believe trying the Inception ResNet V2 model and fine tuning it might give better results, however with the model being much more complex than InceptionV3 I have not tried training it. A good idea I have that I will try is just fine tune the top few "blocks" of the network instead of the entire network. This could yield better results without huge computation times.

### Running this model
1. Download data set (my own):
  * [All three data sets, with more data than udacity and scrubbed of bad data](https://drive.google.com/file/d/1d1IC_MQCiIYdwVTN4_-LwsOUf5X5C4kc/view?usp=sharing)


2. Download data sets (provided by udacity):

  * [Training Set](https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/skin-cancer/train.zip)

  * [Validation Set](https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/skin-cancer/valid.zip)

  * [Testing Set](https://s3-us-west-1.amazonaws.com/udacity-dlnfd/datasets/skin-cancer/test.zip)

2. Unzip each data set into their respective directory in the data folder. Leave them in the three separate folders (melanoma, nevus, seborrheic keratosis).

3. (Optional) Clean data and add more images (if you don't use my uploaded dataset).
  * More images for each category can be found at the [ISIC Archive](https://isic-archive.com/#images).

  * Most of the images found on that website don't appear in the initially download data, with the exception of seborrheic keratosis, since there are few images in the archive for it. So just make sure you are not copying the same data between sets. Also, for better accuracy try to include pictures without other objects in it, like drawn circles or arrows, or blue/yellow markers next to the lesion for size.

  * Some of the data initially provided has those drawn circles and markers in them, so for better results you can look through and delete them. Took me only a few minutes with file previews on.

4. Run the model.

  1. Open preprocess_data.py and change proccesses_num to the number of cores you are willing to give to process the data. This can take some time on one core, so I left it at 4. The higher the number the faster processing will go.
  2. Change image size if you are going to be changing the transferred network, otherwise leave it the same.
  3. Run cnn_network_skin_cancer.py and enter 'y' to pre-proccess the data.
  4. Let the model train! It will give information about trainable and non-trainable parameters, and at the end evaluate the model on the test data. The model will go through two training phases: one to train the classifier fully connected layers on the top of the network, then freeze those weights and fine tune the rest of the model.
  5. Once the model is done training, it will be saved under saved_models.

5. Test images.
  * You can test individual images by running test_model.py
  * It will return the name of the determined disease.
  * **This requires a saved model, which one is saved by running the steps above completely. I do provide my own trained model for testing if you want.**
  * Models are saved under the saved_models directory.
  * Usage is:

```bash
python test_model.py /path/to/image/file /path/to/saved/model
```

### Todo
Right now, I am trying to get TensorBoard callback to work when training, so model training can be visualized with TensorBoard. However, I am getting a ValueError because the returned gradients from the model are None. I believe this may be because of freezing the layers, but I am not sure.

I hope you can get better results than I! You can speed up training if you increase batch size and epochs to have smaller batch sizes per epoch. Good luck!
