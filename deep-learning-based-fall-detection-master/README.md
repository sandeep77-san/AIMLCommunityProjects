# deep-learning-based-fall-detection
### Scope
Injuries causing due to falling down are common in elderly people. 
Automating falls detection lead to quicker medical care which can in turn reduce the subsequent medical complications.
Hence, smart fall detection system based on videos data set is need of time.
![Intro](https://user-images.githubusercontent.com/63051209/90305280-a88adc00-dede-11ea-90a5-3c5ca8815fbf.PNG)

### The following steps involved in Fall detection:

#### Creating a skeleton of human body
A poseNet model (posnet python  https://github.com/tensorflow/tfjs-models/tree/master/posenet) is used to capture the key points of human body and makes a skeleton.
Using the skeleton co-ordinates, a model is trained to recognize different body postures which are ultimately categorized into falling and non-falling category.
Instead of using a single frame, a sequence of consecutive frames are used to fine-tune the model and to improve the accuracy.
Test the model by categorizing body postures into 2, i.e., falling and non-falling category, based on human posture captured from live camera.

#### Gathering the data and prepocessing
- Capturing the data from human skeleton coordinates for each frame of an input video and labeling the body postures with action names(ex: Sitting, Walking , Falling .. etc)
- Standardizing input features and labeling the actions (on hot ending of labels).
- making slices (timeStepSize * noOfFeatures) of input data to feed it to Conv2D and LSTM models.

#### Training the model
Three Neural network models have been considered, compared the results and the best model is choosen to deploy it.
- Sequential neural networks, 
- Convolution 2D neural Networks
- LSTM

#### Deploying the model
The best model among all the model(LSTM ) is deployed in streamlit with good user interface.
![streamlit](https://user-images.githubusercontent.com/63051209/90305317-dcfe9800-dede-11ea-97fa-9cc2aeadff61.PNG)

#### pip install packages
- opencv-python 3.4.5.20
- streamlit     0.63.1

####  Conda install pakages
- tensorflow 1.12.0
- python  3.6.10
- pyyrml 5.3.1
- scipy 1.4.1
- numpy 1.18.5	
- pandas 1.0.4
- keras 2.2.4
- scikit-learn 0.23.1
- matplotlib 3.2.2

#### Pipeline for the project
Week1 - Understanding the deep learning models based on neural networks such convolution, RNN and LSTM and collecting video data set for non-falling and falling actions.

Week2 - Understanding how posenet captures skeleton co-ordinates using opencv

Week3 - Creating a data set of all skeleton co-ordinates(17 key points of human body) for each frame in input video data sets.

Week4 - Making the skeleton data prepared for learning (features and lables). Redudant and dummy data is removed from data set. Training with sequential neural networks and Convolution 2D models.

Week5 - Training with LSTM. Compare the results of all the models and pick the best among all.

Week6 - Deploying the model in streamlit with good user interface.

## references
- Weiming Chen , Zijie Jiang , Hailin Guo and Xiaoyang Ni 'Fall Detection Based on Key Points of Human-Skeleton Using OpenPose'article published in symmetry.
- Zhe Cao, Tomas Simon,  Shih-En Wei, Yaser Sheikh 'Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields'
