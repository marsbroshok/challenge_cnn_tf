#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals

import argparse
import csv
import numpy as np
from keras.models import Model
from keras.layers import Flatten, Input, Dense, Dropout
from keras.layers import Conv1D, MaxPooling1D
from keras.models import load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard

MODEL_NAME = 'cnn3'
np.random.seed(42)


def read_csv(filename, skip_header_lines=1, skip_cols=1):
    """
    Read csv file and return numpy array

    :param filename: full path to the csv file
    :param skip_header_lines: number of lines to skip from the beginning of file
    :param skip_cols: number of columns to skip from the beginning of file
    :return: data as numpy float array
    """

    if filename:
        with open(filename, 'r') as f:
            data = csv.reader(f)
            data = list(data)
            try:
                data = np.array(data[skip_header_lines:], dtype=float)
            except ValueError as e:
                print("Error while puttin csv data into numpy array")
                print("ValueError: {}".format(e))
                raise ValueError
            return data[:, skip_cols:]
    else:
        raise IOError('Non-empty filename expected.')


def save_csv(filename, data):
    """
    Save prediction data to csv file with header and rows ID

    :param filename: full path to output csv file
    :param data: 1D data array to be saved into csv
    :return: saved filename
    """

    data = [(i, r) for i, r in enumerate(data)]
    data.insert(0, ('ID', 'TARGET'))
    with open(filename, 'w') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerows(data)


def transform_features(data, feature_dim=750, features_number=12):
    """
    Take array of samples in a shape of N x M, where N - number of samples, M - raw features
    values (must be equal to `feature_step X features_number`)

    :param data: numpy array with raw data
    :param feature_dim: split step to cut raw data into individual feature's data
    :param features_number: number of individual features in raw data
    :return: numpy array of shape (N, feature_dim, features_number)
    """

    data_stacked = np.array([r.reshape(features_number, feature_dim).transpose() for r in data])
    return data_stacked


def enrich_data(data, labels):
    """
    Generate more data samples by shifting data with random step

    :param data: numpy array with transformed features
    :param labels: target labels for data
    :return: numpy array with original and generated samples
    """

    data_gen = []
    labels_gen = []
    for i, lbl in enumerate(labels):
        # Store original data
        data_gen.append(data[i])
        labels_gen.append(lbl)
        for j in range(5):
            # Shift data
            shift = np.random.randint(10, 740)
            data_mod = np.roll(data[i], shift, axis=0)
            # Add generated data
            data_gen.append(data_mod)
            labels_gen.append(lbl)
    data_gen = np.array(data_gen)
    labels_gen = np.array(labels_gen)
    return data_gen, labels_gen


def input_func(data_file, labels_file, mode, generate_more=True):
    """
    Read CSV and prepare data for consuming by model.

    :param data_file: input data CSV file
    :param labels_file: input labels CSV file
    :param mode: one of the mode for model: {TRAIN, EVAL, PRED}
    :param generate_more: bool flag to create synthetic samples from original one
    :return: x, y - input data formatted to use by model and labels for that data
    """
    if mode == 'PRED':
        # Prediction needs only data, not labels
        x = transform_features(read_csv(data_file))
        return x, []
    else:
        # Train and Eval needs both data and labels
        x = transform_features(read_csv(data_file))
        y = read_csv(labels_file)
        if generate_more:
            x, y = enrich_data(x, y)
        return x, y


def build_model(show_summary=False):
    """
    Build and return a CNN model

    :param show_summary: boole flag to show built model summary
    :return: tensorflow keras model
    """
    input_layer = Input(batch_shape=(None, 750, 12), name='input')
    x = Conv1D(64, 3, activation='relu')(input_layer)
    x = MaxPooling1D()(x)
    x = Conv1D(128, 3, activation='relu')(x)
    x = MaxPooling1D()(x)
    x = Conv1D(256, 3, activation='relu')(x)
    x = MaxPooling1D()(x)
    x = Flatten()(x)
    x = Dropout(0.5)(x)
    output_layer = Dense(1, activation='sigmoid', name='output')(x)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['binary_accuracy'])
    if show_summary:
        model.summary()
    return model


def run_model(model, features, labels, mode):
    """
    if mode is TRAIN take input data with its labels and train new model .
    If mode is EVAL load model from saved state and run evaluation.
    If mode is PRED load model from saved state, run prediction and save predictions to CSV file.

    :param model: keras model
    :param features: input data for model
    :param labels: labels for input data
    :param mode: one of the modes from {TRAIN, EVAL, PRED}
    """
    if mode == 'PRED':
        csv_out_file = './output_predictions_{}.csv'.format(MODEL_NAME)
        scores = model.predict(x=features)
        labels = map(lambda score: 1 if score >= 0.5 else 0, scores)
        save_csv(csv_out_file, labels)
        msg = "Saved prediction results to {}".format(csv_out_file)

    elif mode == 'EVAL':
        loss, accuracy = model.evaluate(x=features,
                                        y=labels)
        msg = "\nModel evaluation finished\nLoss: {}\tAccuracy: {}".format(loss, accuracy)

    else:
        # ok, let's train then!
        saved_model_file = './trained_model_{}.h5'.format(MODEL_NAME)
        # We use early stopping to avoid spending time on overfitting our model
        early_stopping = EarlyStopping(monitor='val_loss', patience=3)
        # save model at checkpoints when loss function improved
        checkpoint = ModelCheckpoint(saved_model_file, monitor='val_loss', save_best_only=True, verbose=1)
        # and keep logs for visualisation with TensorBoard
        tensorboard = TensorBoard('./tensorboard_logs', histogram_freq=1)
        # Train model
        model.fit(x=features,
                  y=labels,
                  epochs=20,
                  validation_split=0.25,
                  callbacks=[tensorboard, checkpoint, early_stopping])

        msg = "Model training finished"
    print(msg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-mode',
                        required=True,
                        type=str,
                        choices=['TRAIN', 'EVAL', 'PRED'],
                        help="""
                          Perform one of the following operations on model use these commands:
                          TRAIN : train model,
                          EVAL : evaluate model
                          PRED : make prediction with model
                      """)
    parser.add_argument('--data-csv',
                        required=True,
                        type=str,
                        help='Raw data CSV file')
    parser.add_argument('--labels-csv',
                        required=True,
                        type=str,
                        help='Labels CSV file. Labels are ignored in PRED mode')
    parser.add_argument('--model-name',
                        required=False,
                        type=str,
                        help='Optional model name to be added as suffix to output files')

    parse_args, _ = parser.parse_known_args()

    if parse_args.model_name:
        MODEL_NAME = parse_args.model_name

    # Prepare input data from CSV files
    input_data, input_labels = input_func(parse_args.data_csv, parse_args.labels_csv, parse_args.run_mode)

    if parse_args.run_mode == 'TRAIN':
        # create model
        model_cnn = build_model()
        # run model with mode params
        run_model(model_cnn, input_data, input_labels, parse_args.run_mode)
        pass
    else:
        try:
            # load model
            model_cnn = load_model('./trained_model_{}.h5'.format(MODEL_NAME))
            #  and run prediction
            run_model(model_cnn, input_data, input_labels, parse_args.run_mode)
        except Exception as e:
            print("Can't found model, check that model was trained and input data is correct.\n".format(e))
