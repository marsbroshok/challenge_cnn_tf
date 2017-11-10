# Solution for the Challenge on ECG Classification

This is a python code to train CNN model, and run evaluation or prediction on ECG (Electrocardiography) data challenge. 
Dataset is availble at git repository [Détection d'inversions ECG](https://github.com/liyongsea/challenge-data)

CNN model defined with Keras framework and used Tensorflow backend.

## Model Architecture
Model architecture motivated by the current state-of-the-art in image processing - Convolutional Neural Networks. 

Contrary to images, ECG signal is 1D signal. So convolutions are applied as 1D filters.

This model has a simple architecture:

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input (InputLayer)           (None, 750, 12)           0         
_________________________________________________________________
conv1d_1 (Conv1D)            (None, 748, 64)           2368      
_________________________________________________________________
max_pooling1d_1  (MaxPooling (None, 374, 64)           0         
_________________________________________________________________
conv1d_2 (Conv1D)            (None, 372, 128)          24704     
_________________________________________________________________
max_pooling1d_2  (MaxPooling (None, 186, 128)          0         
_________________________________________________________________
conv1d_3 (Conv1D)            (None, 184, 256)          98560     
_________________________________________________________________
max_pooling1d_3  (MaxPooling (None, 92, 256)           0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 23552)             0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 23552)             0         
_________________________________________________________________
output (Dense)               (None, 1)                 23553     
=================================================================
Total params: 149,185.0
Trainable params: 149,185.0
``` 

Model trained with cross entropy loss function, max of 20 epochs and early stopping option when there are no improvement in loss function for more that 3 epochs.

On MacBook Pro 13" (2015) CPU training takes around 3 min.

*Notes on training:* training dataset contains around 1K samples and to improve generalisation capabilities of the network there is a simple data augmentation method in `model.py` script. For every original sample there are 5 more samples generated with random horizontal shift (we can say 'in timeline'). So the total volume of training data is higher that the original input data.

## Requirements
This script tested in the following environment:

* Python 2.7
* TensorFlow 1.0 (CPU-only)
* Keras 2.0.2
* Numpy 1.12.0
* h5py 2.6.0
* MacOS Sierra

## How To

*Note:* Before running the `model.py`script be sure to have the challenge data from git repository [Détection d'inversions ECG](https://github.com/liyongsea/challenge-data).

Install python requirements:

`pip install --requirement requirements.txt`

Usage:

```
python  model.py [-h] --run-mode {TRAIN,EVAL,PRED} --data-csv DATA_CSV 
						  --labels-csv LABELS_CSV [--model-name MODEL_NAME]
```
where arguments are:

```
  -h, --help            show this help message and exit
  --run-mode {TRAIN,EVAL,PRED}
                        Perform one of the following operations on model use
                        these commands: TRAIN : train model, EVAL : evaluate
                        model PRED : make prediction with model
  --data-csv DATA_CSV   Raw data CSV file
  --labels-csv LABELS_CSV
                        Labels CSV file. Labels are ignored in PRED mode
  --model-name MODEL_NAME
                        Optional model name to be added as suffix to output
                        files
```

For example, if you have your train data in the file `input_training.csv` and target labels in `output_training.csv`, then this command will train model and save it with suffix `my_cnn`:

```
python model.py --run-mode TRAIN \
                --data-csv input_training.csv \
                --labels-csv output_training.csv \
     
     
                --model-name my_cnn
```

## Credits & Links

[https://www.tensorflow.org/]()

[https://keras.io/]()

______________
Alexander Usoltsev 2017
