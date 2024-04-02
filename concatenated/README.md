# Concatenated Model
This folder contains the model consiting of an LSTM, that uses a GloVe embedding layer, and BERT. The results of both models are concatenated and then passed through some fully connected layers to form the output

## Files
- lauter.py contains the model class of the concatenated model - BERT and LSTM  
- train.py contains the code that has been used to train the lauter model. Every epoch the model is trained and then validated directly afterwards so one can check if the models performance is increasing  
- eval.py contains code for the validation of the generated model. Note that this code is basically already integrated in train.py  
- draw_plots.py contains code to use a file where the stdout of the training is captured and plots those results  
- requirements.txt contains the necessary python libraries that have been used in this code  
- data\ contains the training and validation dataset from NumerSense. 
  - data\validation.masked.categorized.txt contains the validation dataset. This dataset has been modified to contain the category of each respective sentence
