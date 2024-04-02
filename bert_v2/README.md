# Finetuning BERT
This folder contains the code for finetuning BERT with the NumerSense dataset. In every epoch the model is once trained and once evaluated. By evaluating it every epoch one can see the progress that the model is doing. 

## Files
- bert_masked_tut.py contains the working version of the model where only BERT has been used and is being trained as well as being evaluated  
- bert_masked_tut.ipynb contains the same as bert_masked_tut.py but in the notebook format
- plot.py contains the code that has been used to plot the accuracies of the model for each epoch. The stdout of bert_masked_tut.py was captured and then saved as the files that are being read by plot.py
- The dataset folders contain the data that is being used for training and validation of the model. Those are formatted versions of the original NumerSense dataset.
