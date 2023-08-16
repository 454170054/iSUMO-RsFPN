import tensorflow as tf
from tensorflow.keras import layers, optimizers, regularizers
from tensorflow.keras.models import Model


def res_net_block(input_data, filters, strides=1):
    x = tf.keras.layers.Conv1D(filters, kernel_size=3, strides=strides, padding='same')(input_data)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Conv1D(filters, kernel_size=3, strides=1, padding='same')(x)
    x = layers.BatchNormalization()(x)
    if strides != 1:
        downsample = layers.Conv1D(filters, kernel_size=1, strides=strides)(input_data)
    else:
        downsample = input_data
    x = layers.Add()([x, downsample])
    output = layers.Activation('relu')(x)
    return output


def Res_FPN(encode):
    inputs = tf.keras.Input(shape=(encode.shape[1], encode.shape[2]))
    x = layers.Conv1D(128, kernel_size=3)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('gelu')(x)
    x = layers.MaxPool1D(pool_size=2, strides=1, padding='same')(x)
    x = layers.Dropout(0.4)(x)

    x = res_net_block(x, 128)
    x = layers.MaxPool1D(2)(x)
    x = layers.Dropout(0.4)(x)

    x = res_net_block(x, 128)
    x = layers.MaxPool1D(2)(x)
    x = layers.Dropout(0.4)(x)

    x = layers.Flatten()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(128, activation=tf.nn.gelu)(x)
    x_shortcut1 = layers.Dense(64, activation=tf.nn.gelu, kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01))(x)
    x_shortcut2 = layers.Dense(32, activation=tf.nn.gelu, kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01))(x_shortcut1)
    x_shortcut3 = layers.Dense(16, activation=tf.nn.gelu, kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01))(x_shortcut2)
    x_shortcut4 = layers.Dense(8, activation=tf.nn.gelu, kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01))(x_shortcut3)
    x_feature_pyramid_1 = layers.Dense(16, activation=tf.nn.gelu,
                                       kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01))(x_shortcut4)
    x_feature_pyramid_2 = layers.Dense(32, activation=tf.nn.gelu,
                                       kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01))(x_feature_pyramid_1)
    x_feature_pyramid_3 = layers.Dense(64, activation=tf.nn.gelu,
                                       kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01))(x_feature_pyramid_2)

    feature_map_1 = layers.add([x_shortcut3, x_feature_pyramid_1])
    feature_map_2 = layers.add([x_shortcut2, x_feature_pyramid_2])
    feature_map_3 = layers.add([x_shortcut1, x_feature_pyramid_3])

    output1 = layers.Dense(1, activation=tf.nn.sigmoid, name="output1")(feature_map_1)
    output2 = layers.Dense(1, activation=tf.nn.sigmoid, name="output2")(feature_map_2)
    output3 = layers.Dense(1, activation=tf.nn.sigmoid, name="output3")(feature_map_3)
    model = Model(inputs=[inputs], outputs=[output1, output2, output3], name="iSUMO-FPN")
    model.compile(optimizer=optimizers.experimental.AdamW(),
                     loss={
                         'output1': 'binary_crossentropy',
                         'output2': 'binary_crossentropy',
                         'output3': 'binary_crossentropy'},
                     metrics={
                         'output1': 'acc',
                         'output2': 'acc',
                         'output3': 'acc'
                     },
                     experimental_run_tf_function=False)
    return model