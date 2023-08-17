from code.feature_extraction.aaindex import AAIndex
from code.feature_extraction.zsf import ZScale
from code.models.RsFPN import Res_FPN
import tensorflow as tf
import numpy as np
from code.feature_extraction.pbf import extract_embedding_features
import pandas as pd
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
    # load ZSF
    zscale_train, y = ZScale(trainfilepath, 1)
    zscale_test, y_test = ZScale(testfilepath, 1)
    # load PBF
    train_seqs = pd.read_csv(trainfilepath, sep=',')['Sequence']
    protein_bert_train = extract_embedding_features(train_seqs.values.tolist())
    protein_bert_train = np.float32(protein_bert_train)

    test_seqs = pd.read_csv(testfilepath, sep=',')['Sequence']
    protein_bert_test = extract_embedding_features(test_seqs.values.tolist())
    protein_bert_test = np.float32(protein_bert_test)
    return aaindex_train, aaindex_test, zscale_train, zscale_test, protein_bert_train, protein_bert_test, y, y_test


def train_model(train_feature, test_feature, train_label, test_label, filepath):
    model = Res_FPN(train_feature)
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_output3_acc', patience=50)
    best_saving = tf.keras.callbacks.ModelCheckpoint(filepath + '.h5', monitor='val_output3_acc', mode='auto',
                                                     verbose=1, save_best_only=True, save_weights_only=True)
    model.fit(train_feature, train_label, epochs=1000, batch_size=128, validation_data=(test_feature, test_label),
              shuffle=True, callbacks=[early_stopping, best_saving], verbose=0)
    model.load_weights(filepath + '.h5')
    model.save(filepath)


def train():
    aaindex_train, aaindex_test, zscale_train, zscale_test, protein_bert_train, protein_bert_test, y, y_test = load_features_labels()
    output_dir = '../../weights'

    # train model1 based on AAF
    modelName = 'model1'
    filepath = os.path.join(output_dir, modelName)
    train_model(aaindex_train, aaindex_test, y, y_test, filepath)

    # train model2 based on ZSF
    modelName = 'model2'
    filepath = os.path.join(output_dir, modelName)
    train_model(zscale_train, zscale_test, y, y_test, filepath)
    # train model2 based on PBF
    modelName = 'model3'
    filepath = os.path.join(output_dir, modelName)
    train_model(protein_bert_train, protein_bert_test, y, y_test, filepath)