import numpy as np
import os
import requests
from PIL import Image
import io

import tensorflow as tf

print(tf.__version__)

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


class ResNetEmbeding:
    def __init__(self, path):
        self.model = tf.keras.applications.ResNet50(include_top=False, weights=None)
        self.load_model(path)

    def load_model(self, path):
        self.model.load_weights(path)

    def extract_feature(self, url, distant=True):
        if distant:
            content = requests.get(url, stream=True).content
            byteStream = io.BytesIO(content)
            image = Image.open(byteStream)
        else:
            image = Image.open(url)
        image = image.resize([224, 224]).convert('RGB')
        y = tf.keras.preprocessing.image.img_to_array(image)
        y = np.expand_dims(y, axis=0)
        y = tf.keras.applications.resnet50.preprocess_input(y)
        y = self.model(y)
        result = tf.keras.layers.GlobalAveragePooling2D()(y)
        feature = [x for x in result.numpy()[0].tolist()]
        return feature

        # Image.open(BytesIO(data))

    def extract_feature_data(self, data):
        image = Image.open(io.BytesIO(data)).resize([224, 224]).convert('RGB')
        y = tf.keras.preprocessing.image.img_to_array(image)
        y = np.expand_dims(y, axis=0)
        y = tf.keras.applications.resnet50.preprocess_input(y)
        y = self.model(y)
        result = tf.keras.layers.GlobalAveragePooling2D()(y)
        feature = [x for x in result.numpy()[0].tolist()]
        return feature


if __name__ == '__main__':
    resnet = ResNetEmbeding("../models/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5")
    v = resnet.extract_feature(
        "http://127.0.0.1:9000/picture/ILSVRC2012_val_00001195.JPEG?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=PWRF69G12NIGKASVSMIU%2F20240426%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20240426T080543Z&X-Amz-Expires=604800&X-Amz-Security-Token=eyJhbGciOiJIUzUxMiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3NLZXkiOiJQV1JGNjlHMTJOSUdLQVNWU01JVSIsImV4cCI6MTcxNDE2MDQ1MSwicGFyZW50IjoibWluaW9hZG1pbiJ9.cMrwB1mRJhD5kURUr06mC-ffxBC-PvY4aE7NCyOPxy9vA9BJoqGi5CRCPTkDsu700L1_hrhuVsJRYR25ePP4ww&X-Amz-SignedHeaders=host&versionId=null&X-Amz-Signature=495148349161b61c4345e49d4f631f369a3d9b195f11e8110c8dc29112697309")
    print(v)
