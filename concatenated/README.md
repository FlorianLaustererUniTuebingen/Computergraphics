# Concatenated Model
This folder contains the model consiting of an LSTM, that uses a GloVe embedding layer, and BERT. The results of both models are concatenated and then passed through some fully connected layers to form the output

## Files
- lauter.py contains the model class of the concatenated model - BERT and LSTM  
- train.py contains the code that has been used to train the lauter model. Every epoch the model is trained and then validated directly afterwards so one can check if the models performance is increasing  
- eval.py contains code for the validation of the generated model. Note that this code is basically already integrated in train.py  
- draw_plots.py contains code to use a file where the stdout of the training is captured and plots those results  
- data\ contains the training and validation dataset from NumerSense. 
  - data\validation.masked.categorized.txt contains the validation dataset. This dataset has been modified to contain the category of each respective sentence

## Training the model
- Install the requirements from the top level readme file
- Download GloVe embeddings
- Adjust the paths for the training and validation dataset
- then run:
```
python train.py
```

## Evaluating the model
Within the training loop the model gets evaluated every epoch. If you still wish to do a seperate evaluation adjust the paths in the eval.py file, then run:
```
python eval.py
```

## Plots
In our case the stdout was captured and is then being used for the plots. You could just copy and paste it into a file, adjust the path in the code and then run:
```
python draw_plots.py
```
