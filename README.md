# A novel approach for NumerSense
We created an approach to solve the NumerSense task (https://inklab.usc.edu/NumerSense/) using visually grounded information. 
The goal of NumerSense is to increase the performance of PTLMs for the task of numerical commonsense, because those models, altough generally performing good, weren't able to perform with high accuracy in these tasks in the past. 
E.g. the output of a sentence like "A bird usually has <MASK> legs." is "four" if evaluated by BERT. That's why the team from NumerSense has created a dataset so models can be finetuned to predict those numbers correctly.
With our code we propose one possibilty for a model that uses this dataset in training

## concatenated
This model consits of two seperate models. On one side a LSTM that uses visually grounded GloVe embeddings, on the other side a BERT. 
The result of both models is concatenated and inserted into Fully connected layers that should generate a numerical result.
![ConcatenatedArchitecture_with_background](https://github.com/FlorianLaustererUniTuebingen/Computergraphics/assets/165826773/0a56e475-8726-4471-896f-99911d07aa3b)
The model can be found in the "concatenated" folder and the training and validation data can be found in the "concatenated/data" folder

## BERT V2
Since our concatenated model did not train as well as we had hoped, we created a regular BERT model that should be finetuned using the NumerSense dataset. 
The code for this can be found in the "bert_v2" folder. The training data is in the "bert_v2/masked_training_dataset" folder and the validation data is in the "bert_v2/masked_validation_dataset" folder
