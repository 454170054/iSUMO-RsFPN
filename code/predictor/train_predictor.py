from code.feature_extraction.aaindex import AAIndex
from code.feature_extraction.zsf import ZScale
from code.models.RsFPN import Res_FPN
import tensorflow as tf
import numpy as np
import os


def load_features_labels():
    """
    load AAF, ZSF, PBF and labels
    """
    trainfilepath = r'../../dataset/five_fold_cross_validation.csv'
    testfilepath = r'../../dataset/independent.csv'
    # load AAF
    aaindex_train = AAIndex(trainfilepath)
    aaindex_test = AAIndex(testfilepath)
    AAF = np.concatenate((aaindex_train, aaindex_test), axis=0)
    # load ZSF
    zscale_train, y = ZScale(trainfilepath, 1)
    zscale_test, y_test = ZScale(testfilepath, 1)
    ZSF = np.concatenate((zscale_train, zscale_test), axis=0)
    # load PBF
    protein_bert_train = np.load('../../protein_bert_f/train_features.npy')
    protein_bert_train = np.float32(protein_bert_train)
    protein_bert_test = np.load('../../protein_bert_f/test_features.npy')
    protein_bert_test = np.float32(protein_bert_test)
    PBF = np.concatenate((protein_bert_train, protein_bert_test), axis=0)
    return AAF, ZSF, PBF, np.concatenate([y, y_test], axis=0)


def train_model(features, labels, filepath):
    model = Res_FPN(features)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_output3_acc', patience=50)
    best_saving = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_output3_acc', mode='auto',
                                                     verbose=1, save_best_only=True, save_weights_only=True)
    model.fit(features, labels, epochs=1000, batch_size=128, validation_split=0.05,
              shuffle=True, callbacks=[early_stopping, best_saving], verbose=0)


def train():
    AAF, ZSF, PBF, labels = load_features_labels()
    output_dir = '../../weights'

    # train model1 based on AAF
    modelName = 'model1.h5'
    filepath = os.path.join(output_dir, modelName)
    train_model(AAF, labels, filepath)

    # train model2 based on ZSF
    modelName = 'model2.h5'
    filepath = os.path.join(output_dir, modelName)
    train_model(ZSF, labels, filepath)

    # train model2 based on PBF
    modelName = 'model3.h5'
    filepath = os.path.join(output_dir, modelName)
    train_model(PBF, labels, filepath)