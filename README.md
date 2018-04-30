# **Traffic Sign Recognition** 

[image1]: ./report_images/5x5randomimages.png
[image2]: ./report_images/bar_chart.png
[test_img1]: ./test_data/web1.png
[test_img2]: ./test_data/web2.png
[test_img3]: ./test_data/web3.png
[test_img4]: ./test_data/web4.png
[test_img5]: ./test_data/web5.png

## Data Set Summary & Exploration

In the following section, I will go through the details of the dataset.

### Dataset overview

The provided dataset is a large set of images of german traffic signs.
The dataset consists of 51839 images, where each image has the dimensions (32x32x3), e.g. 32-by-32 pixels and three channels (RGB).

Within the dataset there are 43 different classes of images. 
The names of the different classes can be found in the `signnames.csv` file

The dataset is divided into three parts: Training, Validation and Test.
The size of these subsets is shown in the table below.

| Data set   | Size  | Percentage | 
| -----------|:-----:|:----------:|
| Training   | 34799 | 67.13      |
| Validation | 4410  |  8.50      |
| Test       | 12630 | 24.36      |
        
Which means that the network will only ever learn from 75.5 % of the data, and will then be tested against 24.36 % of the entire data.


### Visualisation

To visualise the dataset, a 25 randomly chosen images are plotted in a 5x5 grid below.
![alt text][image1]


To further visualize the dataset, a bar chart below is shown to give a visual overview of the distribution of image classes in the dataset.
![TEST TEXT][image2]

At a first glance the validation set seemingly is much more uniform in its distribution than e.g. the training dataset, which can contain vastly different numbers of each label. An example would be the labels `0` and `2`, where there are almost 10 times (approx. 2000 images) the amount of images of label `2` than that of label `0` (around 220 images).

However, on further inspection, all the datasets seem to have the same "shape" of the distribution across labels, this distribution is just scaled.
This make sense, if all the datasets are completely randomly sampled for the same pool of images, each subset of would contain more or less the same distribution (in an ideal world).



## Design and Test of Model Architecture

### Data processing

To process the data I chose the following steps:

* Convert to grayscale
* Normalize image data between [-1, 1]

By converting to grayscale, we are reducing the data complexity by a factor of 3, by going from three channels of data to one channel.
Also, by doing this, we are removing any color dependency/importance of the traffic signs, which should force the network to learn from the shapes.

By normalizing the data, the corresponding gains and biases in the network are also scaled down, and are should be distributed around 0.


### Model Architechture

The model, was built ontop of the LeNet image classifier, and consisted of the following layers:

| Layer         		|     Description	        					| Input Size  | Output Size |
|:---------------------:|:---------------------------------------------:|:-----------:|:-----------:|
| Convolution        	| 1x1 stride, valid padding, outputs 28x28x6 	| 32x32x1     | 28x28x6     |
| RELU					| Activation function                           | 28x28x6     | 28x28x6     |
| Max pooling	      	| 2x2 stride                                    | 28x28x6     | 14x14x6     |
| Convolution           | 1x1 stride, valid padding, outputs 28x28x6	| 14x14x6     | 10x10x16    |
| RELU                  | Activation function                           | 10x10x16    | 10x10x16    |
| Max pooling	      	| 2x2 stride                    				| 10x10x16    | 5x5x16      |
| Flatten       		| Reshapes data for fully connected layer       | 5x5x16      | 400x1       |
| Fully connected		| Hidden layer, with dropout                    | 400x1       | 120x1       |
| Fully connected		| Hidden layer, with dropout                    | 120x1       | 84x1        |
| Fully connected		| Hidden layer, with dropout                    | 84x1        | 43x1        |
| Softmax				| Final activation function to get logits       | 43x1        | 1x1         |


### Model Parameters


To train the network, an [Adam Optimizer](https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer) was used,
with the following hyper parameters:

| Parameter        | Value  |
| -----------------|:------:|
| Mu               | 0      |
| Sigma            | 0.1    |
| Learning rate    | 0.001  |
| Epochs           |  10    |
| Batch Size       | 128    |
| Keep probability | 0.6    |


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 98.5%
* validation set accuracy of 94.2%
* test set accuracy of 93.8%

As stated previously, the model used was based of the LeNet architechture.
The reasoning behind this, was that the LeNet architechture was shown to be a good image classifier, when used on the MNIST dataset, which could easily be translated to the traffic sign problem, as LeNet used convolutional layers.
The convolutional layers ensures that the network will learn patterns/shapes, which is important in both recognising/classifying hand written digits as well as traffic signs.

However, when using the bare implementation of LeNet, the results were not really that great.
The first results showed a test set accuracy in the range of 83-85% of correctly classified images, with a training accuracy of +95% accurate.
This indicated that the model might have an overfitting problem.
To remedy this, I tried the following:

* Decrease the learning rate
* Add dropout

By decreasing the learning rate, thus, slowing the model parameters change, the results got better, but still the test set accuracy were not above 90%.

As a last step, the I added dropout to the model. 
This should ensure that the network would generate a very general model, as it cannot use specific nodes or parts in the network, to indicate or represent certain traffic signs.
The dropout was added to all the fully connected layers, including the flattening layer.
This will result in not all parts of the image being used in the training

The dropout percentage was experimented with, and found to be efficient around the 40%, e.g. 60% of all the nodes in the network would be active at any given training operation.

To conclude, the test set and validation set accuracy are roughly the same value.
This should indicate that the model is valid.


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

To verify the models accuracy further, I test the model on 5 images found on the web. 
Here are the results of the prediction:

| Image			 | Class                                 |     Prediction	        					| 
|:--------------:|:-------------------------------------:|:--------------------------------------------:| 
| ![][test_img1] | General caution                       | Wild animals crossing                        | 
| ![][test_img2] | Priority road                         | Priority road                                |
| ![][test_img3] | Yield                                 | Yeild                                        |
| ![][test_img4] | Right-of-way at the next intersection | Go straight or left                          |
| ![][test_img5] | Speed limit (70km/h)                  | Speed limit (70km/h)                         |


The model was able to correctly guess 2 of the 5 traffic signs, which gives an accuracy of 60%.

#### Analysis of the image predictions

To give some insights into the networks predictions, the top 5 predictions and their softmax probabilities have been calculated.

Below are five tables, each with the probability and classification for each of the five images.

First image:

| Probability         	|     Prediction	           | 
|:---------------------:|:----------------------------:| 
| .958         			| Wild animals crossing        | 
| .035    				| General caution              |
| .006					| Double curve				   |
| .000	      			| Road narrows on the right	   |
| .000				    | Bicycles crossing            |


Second image:

| Probability         	|     Prediction	    | 
|:---------------------:|:---------------------:| 
| .999         			| Priority road         | 
| .000     				| Roundabout mandatory  |
| .000					| Speed limit (100km/h) |
| .000	      			| Speed limit (50km/h)  |
| .000				    | Speed limit (80km/h)  |


Third image:

| Probability         	|     Prediction	    | 
|:---------------------:|:---------------------:| 
| .999         			| Yield                 | 
| .000     				| Speed limit (60km/h)  |
| .000					| Ahead only            |
| .000	      			| Children crossing     |
| .000				    | Speed limit (30km/h)  |


Fourth image:

| Probability         	|     Prediction	    | 
|:---------------------:|:---------------------:| 
| .915         			| Go straight or left   | 
| .069     				| General caution       |
| .010					| Road work             |
| .002	      			| No entry              |
| .000				    | Double curve          |

Fifth image:

| Probability         	|     Prediction	    | 
|:---------------------:|:---------------------:| 
| .994         			| Speed limit (70km/h)  | 
| .003     				| Speed limit (20km/h)  |
| .002					| Speed limit (30km/h)  |
| .000	      			| Speed limit (120km/h) |
| .000				    | Roundabout mandatory  |


In the cases where the network guessed the image class correctly, the network has a very high probability (+ 0.99),
compared to the cases where the network guess incorrectly, e.g. on the first image, where the network also considers the correct class (General caution) to be probable.

In the first image, all the classes have the same shape and color, and further the second highest guess is the correct class.

In the fourth image, the network is the least sure of all the five images, and it also has guessed completely wrong on all five guesses.
