**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./examples/placeholder.png "Grayscaling"
[image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"
[model]: ./examples/model.png "Model Visualization"
[center]: ./examples/center_2017_07_23_13_15_54_229.jpg "Center driving"
[recover1]: ./examples/center_2017_07_23_13_16_05_579.jpg "Recovering from left 1"
[recover2]: ./examples/center_2017_07_23_13_16_06_673.jpg "Recovering from left 2"
[recover3]: ./examples/center_2017_07_23_13_16_07_151.jpg "Recovering from left 3"
[recoverCurve1]: ./examples/center_2017_07_23_13_16_23_267.jpg "Recovering in curve 1"
[recoverCurve2]: ./examples/center_2017_07_23_13_16_23_786.jpg "Recovering in curve 2"
[recoverBridge1]: ./examples/center_2017_07_23_13_16_38_009.jpg "Recovering in bridge 1"
[recoverBridge2]: ./examples/center_2017_07_23_13_16_38_958.jpg "Recovering in bridge 2"
[center2]: ./examples/center_2017_07_23_13_16_58_104.jpg "Center driving 2"
[centerinv]: ./examples/center_2017_07_23_13_15_54_229_inv.jpg "Inverse if centerd image"


---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results
* data1_60.mp4 video file of the first track
* model_2.h5 containing a trained convolution neural network to drive the second track (it has been generated with python 2.7, in contrast with model.h5 which is generated for python 3.5)
* data2_60.mp4 video file of the second track

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```
sh
python drive.py model.h5
```

Fir the second track:
```
sh
python2.7 drive.py model_2.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network based on the arquitecture used in NVIDIA's end-to-end deep learning explained in https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/. I've tested a few different models and this is the one that obtained better results.

The first step is to transform the images to grayscale and then normalize the data with a Lambda layer in Keras. After this, the images contains some data that is not needed in the top and bottom, corresponding to sky and the hood of the car, so we can cut this part.

Then, the model consists of 5 convolutional layers, with a kernel of 5x5 and 3x3. After the convolutional layers, there are three fully connected layers.

#### 2. Attempts to reduce overfitting in the model

To reduce overfitting, especially when we increase the amount of data, it is necessary to introduce Dropout layers. I have used two Dropout layers of 0.25 between the convolutional layers, and a Dropout layer of 0.5 after the first fully connected layers. 

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

The training data was obtained by driving the car through the simulations as centered as possible. It consists of a few laps in each direction, plus some additional data in curves and the bridge. Also, the data contains some cases when the car needs to recover, because the car goes to the left or right side of the roads. I've also tried to drive through the curves differently each time, taking the curve by the inside and outside.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture, was to try models used in some similar problems, and test how it perform on the simulation, and when I was happy with the results obtained with this model, I modified it a little, to reduce overfitting and try to improve the results further.

My first step was to use a convolution neural network model similar to LeNet's, because it works well with images and its used in a lot of different problems, but in this case it didn't work very well, the car sometimes turned right when it should go left, or just kept going straight in some curve. Maybe it works better for classification than regression. After this, I tried the model developed by NVIDIA for the end-to-end deep learning for autonomous driving. This one worked much better and it's the one used in the end. Additionally, I tried to use a Dueling Network arquitecture, because it has been used by Deep Mind to play some Atari games, some of which are similar to the simulation, but I had the same problem as with the LeNet's, it worked mostly all right, but I could not complete a lap.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model adding three Dropout layers, two between the convolutional layers with a drop of 0.25 of the data, and another one after the first fully connected layer with 0.5.

To improve the results, instead of using color images, I transformed the images to grayscale, which improve the results a lot, finally been able to driving through the simulation without going out of the road. 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track, in the first curve and the curve after the bridge. Also, in the bridge, the car sometimes went to the left and got stuck. To prevent this I took data of this places a few times trying to change the driving each time. For example, for the curves, I sometimes took the curve in the inside and others on the outside, correcting when I could go out of the road, and the same whith the bridge, sometimes I went over the bridge on the center, and other times more on the left/right, and correcting the movement of the car.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture consisted 5 convolutional layers, 3 droput layers, and 4 fully connected layers, with an initial normalization layer. 

Here is a visualization of the architecture

![alt text][model]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][center]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover when some prediction is a little of, and makes the model more stable. These images show what a recovery looks like starting from the left :

![alt text][recover1]
![alt text][recover2]
![alt text][recover3]

Here recovering in a curve when I'm going out of the road:

![alt text][recoverCurve1]
![alt text][recoverCurve2]

And here in the bridge:

![alt text][recoverBridge1]
![alt text][recoverBridge2]

Then I repeated this process on track two in order to get more data points.

![alt text][center2]

To augment the data sat, I also flipped images and angles thinking that this would duplicate the data, because even though I drove in both directions, both are different, so this effectively augment the data. For example, here is an image that has then been flipped:

![alt text][center]
![alt text][centerinv]

After the collection process, I had around 30K images. I then preprocessed this data by first converting to grayscale, after this there is a normalization layer in Keras, and a cropping layer to cut unwanted parts of the images.

I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by the validation loss, I used an adam optimizer so that manually training the learning rate wasn't necessary. Although to learn for the second circuit were I needed more data, I used Adam's optimizer with a learning rate of 0.00001 and 120 epochs to obtain acceptable results.

Because the training data consumed a lot of memory, I tried to use generators to prevent having all the images in memory, but the performance was very bad, and the results were not great, so instead I optimized the memory consumption of the algorithm, removing all the duplications of the data structures.

#### 4. Comments on the second track
The model provided can drive through almost all the second track, but there were two curves were I had to control the car manually. I think the main reason for the model to have trouble with this track, is that we predict only the angular orientation, but not the speed, and for example, there is one curve in a slope, where you can see very little of the road ahead,  and the car should slow down a lot in order to know what it should do. Also, this track is difficult to drive manually too, so the quality of the training data is worse.
