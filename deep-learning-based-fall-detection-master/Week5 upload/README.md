## Week5

Training with LSTM and compare the results of all the model and pick the best among all.

## Comparative study of all the methods

#### Sequential NN:

Accuracy: 0.95


              precision    recall  f1-score   

     Sitting       0.85      0.99      0.92     
     Walking       0.99      0.83      0.90      
     PushUps       1.00      1.00      1.00      
     Falling       1.00      1.00      1.00      

#### CNN 2D model:

Accuracy: 0.98

Classification Report

              precision    recall  f1-score  

     Sitting       0.98      0.94      0.96       
     Walking       1.00      0.98      0.99       
     PushUps       0.94      1.00      0.97        
     Falling       0.97      1.00      0.99     

#### RNN/LSTM model:

Accuracy: 0.99

Classification Report

              precision    recall  f1-score   

     Sitting       1.00      0.98      0.99        
     Walking       1.00      1.00      1.00        
     PushUps       0.97      1.00      0.99        
     Falling       1.00      1.00      1.00     


All the models have better overall acuracies and performed well on data set. But sequential model has low f1score  for 'sitting' and 'walking' , which performed not well to descrimiate both the actions. 
Where as CNN model has shown some better f1score for both actions and perfomed well. 
But RNN/LSTM model has discriminated both actions perfectly and showing good f1score for both the actions.

So RNN/LSTM is the best model among all.

  



