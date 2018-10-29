## Utils.py -- Some utility functions
##
## Copyright (C) 2018, IBM Corp
##                     PaiShun Ting <paishun@umich.edu>
##                     Pin-Yu Chen <Pin-Yu.Chen@ibm.com>
##                     Chun-Chen Tu <timtu@umich.edu>
## Copyright (C) 2017, Huan Zhang <ecezhang@ucdavis.edu>.
## Copyright (C) 2016, Nicholas Carlini <nicholas@carlini.com>.
##
## Licensed under the Apache License, Version 2.0 (the "License");
## you may not use this file except in compliance with the License.
## You may obtain a copy of the License at
##
##     http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS,
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
## See the License for the specific language governing permissions and
## limitations under the License.

from keras.models import Model, model_from_json, Sequential
from PIL import Image

import os
import numpy as np

def save_img(img, name = "output.png"):

    np.save(name, img)
    fig = np.around((img + 0.5)*255)
    fig = fig.astype(np.uint8).squeeze()
    pic = Image.fromarray(fig)
    pic.save(name)

def generate_attack_data_set(data, model, MGR):

    num_sample = MGR.parSet['nFunc']
    target_label = MGR.parSet['target_label']

    orig_imgs = []
    orig_labels = []
    orig_imgs_id = []

    pred_labels = np.argmax(model.model.predict(data.test_data), axis=1)
    true_labels = np.argmax(data.test_labels, axis=1)
    correct_data_indices = np.where([1 if (x==y and y == target_label) else 0 for (x,y) in zip(pred_labels, true_labels)])
    correct_data_indices = correct_data_indices[0]

    data.test_data = data.test_data[correct_data_indices]
    data.test_labels = data.test_labels[correct_data_indices]
    true_labels = true_labels[correct_data_indices]

    class_num = data.test_labels.shape[1]

    for sample_index in range(num_sample):
        orig_imgs.append(data.test_data[sample_index])
        orig_labels.append(data.test_labels[sample_index])
        orig_imgs_id.append(correct_data_indices[sample_index])

    return np.array(orig_imgs), np.array(orig_labels), np.array(orig_imgs_id)
