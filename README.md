# Big-Data-Project
**Abstract**  
The project on the development of a new technique for land cover classification of satellite images
is useful because it provides a more accurate and precise method for identifying and classifying
the different types of land cover present in an area. The use of advanced image processing and
machine learning algorithms allows the technique to achieve a high level of accuracy and
precision, providing valuable information for various applications, such as environmental
monitoring, urban planning, and natural resource management. In terms of improving current
technology, the proposed technique represents a significant advance over existing methods, as it
is able to handle large amounts of data and complex land cover patterns with greater accuracy and
efficiency. Overall, the project has the potential to improve current technology and provide
valuable insights for a wide range of applications.

**DataSet**  
For this project we are using EuroSat Dataset.
EuroSAT is a dataset of satellite images of various land cover and land use classes across 10 European countries. It was created by using Sentinel-2 satellite images from the European Space Agency's Copernicus program. The dataset consists of 27,000 images, which are divided into 10 classes:

1. Sea and Lake: bodies of water, including oceans, seas, lakes, and reservoirs.
2. Highway: roads used for transportation, including highways and local roads.
3. River: natural channels of water flowing from higher to lower ground, including streams and creeks.
4. Pasture: fields of natural or sown grass used for grazing livestock.
5. Industrial: areas used for industrial purposes, such as factories and warehouses.
6. Residential: areas used for housing and other residential purposes, such as houses and apartment buildings.
7. Permanent Crop: fields of permanent crops, such as fruit trees, grapevines, and olive trees.
8. Annual Crop: fields of annual crops, such as corn, wheat, and soybeans.
9. Forest: land covered by trees, with at least 30% tree cover.
10. Herbaceous Vegetation: land covered by herbaceous plants, such as grasses, forbs, and other non-woody plants.

The images are labeled with one of the 10 classes described above, based on the dominant land cover or land use in the image. The EuroSAT dataset is widely used for land cover and land use classification tasks, as well as for training and evaluating deep learning models for image classification.  
![image](https://user-images.githubusercontent.com/54617669/209219038-abcb38a7-ec54-46c7-b187-b1d95858fa37.png)

**Implementation**  
In this project, we developed a convolutional neural network (CNN) model for classifying land
cover using satellite images. The dataset used for training and testing the model was the EuroSat
dataset, which contains 27000 images of 10 different classes of land cover: "sea lake,"
"highway," "river," "pasture," "industrial," "residential," "permanent crop," "annual crop,"
"forest," and "herbaceous vegetation."  

In the pre-processing step, we divided the original dataset into two parts: 21600 images for
training and 5400 images for testing. We then created a train_generator and a test_generator to
pre-process and augment the images before feeding them to the CNN models. This involved
several steps, such as resizing the images to the required dimensions, applying random
transformations to the images (such as rotation, shearing, and flipping) to increase the diversity
of the dataset, and normalizing the pixel values to have zero mean and unit variance.  

We trained four different pre-trained Keras models on the EuroSat dataset: ResNet50,
ResNet50V2, VGG16, and VGG19. These models have been trained on large datasets of natural
images and have achieved state-of-the-art performance in many image classification tasks. We
used the pre-trained weights of these models as the starting point for our own training and fine-
tuned them on the EuroSat dataset.  

To train the models, we first pre-trained them on the training data using the train_generator and
then fine-tuned them end-to-end on the same data. We used a variety of techniques to improve
the performance of the models, such as reducing the learning rate and applying regularization to
prevent overfitting. We also experimented with different hyperparameters, such as the batch size
and the number of epochs, to find the optimal settings for each model.  

After training the models, we evaluated their performance on the test dataset using the
test_generator. We calculated a variety of metrics to assess the quality of the predictions, such as
accuracy, precision, recall, F1 and F2 scores, and confusion matrices. These metrics provide a
comprehensive view of the performance of the models and allow us to compare them with each
other.  

Finally, we saved all the trained models for future testing. This allows us to use the models in
future projects or even deploy them in real-world applications, such as land cover mapping or
10environmental monitoring. Overall, the CNN models performed well on the EuroSat dataset and
achieved good results in terms of classification accuracy and other metrics.  

**Results**

**(a) ResNet50**
![resnet50 result](https://user-images.githubusercontent.com/54617669/209216815-9567197a-a487-4f17-8599-401ad9496bee.png)
![resnet50 CM](https://user-images.githubusercontent.com/54617669/209216841-f5aefceb-6708-487a-858e-c7413e0a8e99.png)


**(b) ResNet50V2**
![resnet50v2 result](https://user-images.githubusercontent.com/54617669/209216983-21bd1a04-5729-4e2d-8871-2b63160e8524.png)
![resnet50v2 Cm](https://user-images.githubusercontent.com/54617669/209216992-47cba241-2cdd-4819-8f23-de75494ceda4.png)


**(c) VGG16**
![vgg16 result](https://user-images.githubusercontent.com/54617669/209217031-4e7e7cae-c84a-48f0-a405-c6a62dff8c68.png)
![vgg16 CM](https://user-images.githubusercontent.com/54617669/209217041-14a8d329-892b-4df6-8dde-ba6e01aab9f8.png)

**(d) VGG19**
![vgg19 result](https://user-images.githubusercontent.com/54617669/209217066-2a00dbb7-0e70-4616-a004-75fffa72d63a.png)
![vgg19 CM](https://user-images.githubusercontent.com/54617669/209217082-af4d21d6-fd60-4900-b052-977a4478aef8.png)

