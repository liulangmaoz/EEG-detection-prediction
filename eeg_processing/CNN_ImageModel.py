# -*- coding: utf-8 -*-
# CNN Image Classification Model (InceptionV3)
# For EEG Time-Frequency Image Classification

import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD

def build_cnn_image_model(img_size=(224, 224), num_classes=3, lr=1e-4):
    """
    构建时频图分类 CNN 模型（InceptionV3）
    """
    base_model = InceptionV3(
        weights="imagenet",
        include_top=False,
        input_shape=(*img_size, 3)
    )

    # 冻结前10层
    for layer in base_model.layers[:10]:
        layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.6)(x)
    out = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs=base_model.input, outputs=out)

    model.compile(
        optimizer=SGD(learning_rate=lr, momentum=0.9),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model