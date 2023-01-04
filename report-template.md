# Report: Predict Bike Sharing Demand with AutoGluon Solution
#### NAME: SUKRUTA PRASANNA PARDESHI

## Initial Training
### What did you realize when you tried to submit your predictions? What changes were needed to the output of the predictor to submit your results?
While submitting the predictions, we can observe the initial score of 1.810. This score could be improved by incorporating more parameters while predicting the values. As the TabularPrediction model iterated the training data of bike predictors through baseline conditions, it was observed that the validation score of models could have been improved better by optimizing the models with some additional hyperparameters. 

### What was the top ranked model that performed?
The top model that performed well in the AutoGluon models list was 'WeightedEnsemble_L3'. It had a validation score i.e. a score_val of -52.771 and it was also noted that this model had the highest fit_time of 434s.

## Exploratory data analysis and feature creation
### What did the exploratory analysis find and how did you add additional features?
Additional features were added to the data by splitting the datetime into the year, month, and hour features. Also, the data types of season and weather were changed to categories so that AutoGluon can efficiently fit the models on these categorical data points. We get a clear understanding after splitting the datetime and grabbing insights after plotting histogram plot against their frequencies and all features. We can now visualize and see the bike demands per hour, year, and month after additional features were included in the dataset.

### How much better did your model preform after adding additional features and why do you think that is?
After including the additional features, the new score of the predictions was leveraged to 0.6892. Though the score_val of the best model was -30.136, still the score is considered acceptable as per the submission rules of the competition. A score greater than 0.5 is assumed to be better fitted by the model. By increasing the number of features in your dataset the model can be better trained with new data points, therefore by adding features we can study how the model predicts and determine the performance metrics.

## Hyper parameter tuning
### How much better did your model preform after trying different hyper parameters?
As the previous score was 0.689 after tuning hyperparameters and changing their values, the score after hyperparameter tuning was 0.541. The score decreased slightly after adding the ‘hyperparameters’ and ‘hyperparameter_tune_kwargs’ attributes. Here, in this project, I’ve considered the hyperparameters as ‘very_light’ as it trained only fewer models. This was achieved to reduce disk storage consumption. The hyperparameter_tune_kwargs parameter ‘num_trails’ wasn’t included as the models were not trained efficiently. Here the hyperparameters that were used to train fewer models were – num_bag_folds, num_bag_sets, num_stack_levels (these all three come under the Stack configuration of models), scheduler, and searcher. Although the score of 0.541 is a good score in terms of the Kaggle leader scoreboard. 

### If you were given more time with this dataset, where do you think you would spend more time?
I would spend more time optimizing the predictions by tuning more hyperparameters to the model. I still think the model score could be improved and the predictions could be boosted by adding more parameters and iterating the training data for a longer duration to achieve better precision and recall scores. As in this project, I’ve attempted to use very_light hyperparameters which can be enhanced by manually configuring the individual model’s parameters for example – hyperparameters={'NN_TORCH': {'num_epochs': 2}, 'GBM': {'num_boost_round': 20}}. 

### Create a table with the models you ran, the hyperparameters modified, and the kaggle score.
|model|num_stack_levels|num_bag_folds|num_bag_sets|score|
|--|--|--|--|--|
|initial|1|8|20|1.81053|
|add_features|1|8|20|0.68929|
|hpo|3|5|8|0.54162|

### Create a line plot showing the top model score for the three (or more) training runs during the project.


![model_train_plot](https://user-images.githubusercontent.com/67223554/210557519-654e228c-be66-4448-9d8b-0c06f8210e7f.png)


### Create a line plot showing the top kaggle score for the three (or more) prediction submissions during the project.


![model_test_plot](https://user-images.githubusercontent.com/67223554/210557551-1958219b-e7a8-49cc-a62e-62e492013f5b.png)


## Summary
This project gave predictions on bike sharing demands based on several features like date, season, weather, temperature, humidity, and many more. The predictions were obtained by training the dataset over several AutoGluon models using the TabularPrediction model. Here, the dataset was trained first through baseline conditions to achieve a prediction score of 1.81, then some additional features were included to obtain a 0.68 prediction score. Finally, hyperparameters such as scheduler, and searcher were tuned with the model which decreased the prediction score up to 0.54. Although, this score is considered better from the perspective of the Kaggle leader scoreboard submission criterion. The features were also visualized using histogram plots.  
