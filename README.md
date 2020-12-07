# unifyid_challenge
Instructions to run the code:
Requirements: To successsfully run the code the following packages are required:
1. Pandas
2. Numpy
3. sklearn
4. Tensorflow 2.2.0
5. keras

To execute the code just run the main.ipynb jupyter notebook in the correct environment with all the required packages. The code will output the following:
1. The final prediction of the user hash is written in a json file final_pred.json. The json file contains a list of strings where each string is the user id 
corresponding to the data in test json file.
2. user_level_perf.csv : Contains the performance of the model on the train data aggregated at a per user level. The descriptive statistics of the user level 
performance are also printed in the jupyter notebook.
3. model_checkpoints : The model checkpoints corresponding to the best val accuracy are stored in the folder. Ideally this would be crucical if later the code 
has to be run directly in test mode using the pre trained model. Currently this feature is under development in the run.py file.

Answers to the questions asked in the challenge are given along with the code in the jupyter notebook markdown format.

## Additional Questions
1. If you had one additional day, what would you change or improve to your submission?
First of all I would complete the run.py so that the train data and test data urls could be changed and read by the system accordingly. Secondly ability to run the system in test mode directly if the training has already been done.
I would improve the model architecture and try using LSTM. Currently in feature extraction I have deleted all the backspaces and the actual mistakes. This information could be used by using an LSTM which would find richer relationships between the different mistakes made and the time required.
Some of the other baseline models that I would try with the existing features are: svm, logistic regression for multi-class to gain more insights into basic relationships in the data.

2. How would you modify your solution if the number of users was 1,000 times larger?
If the number of users were 1000x larger than the current softmax solution would pose significant challenges. In this case I would change the loss function to negative sampling instead of softmax crossentropy to reduce the computational complexity on the final layer. 

3. What insights and takeaways do you have on the distribution of user performance?
While the model performs extremely well on some users, there is a significant percentage of users for use the model basically learns nothing. I would further analyse this by looking into the distribution of overall typing time and number of bacckspaces used per user to gain an insight if it has any correlation. 
