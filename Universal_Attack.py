## Universal_Attack.py -- The main entry file for attack generation
##
## Copyright (C) 2018, IBM Corp
##                     PaiShun Ting <paishun@umich.edu>
##                     Pin-Yu Chen <Pin-Yu.Chen@ibm.com>
##                     Sijia Liu <sijia.liu@ibm.com>
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

import sys
sys.path.append('models/')
sys.path.append('optimization_methods/')

import os
import numpy as np
import argparse

from setup_mnist import MNIST, MNISTModel
import Utils as util
import ObjectiveFunc
import ZO_SVRG as svrg
import ZO_SGD as sgd
from SysManager import SYS_MANAGER



MGR = SYS_MANAGER()
def main():

    data, model =  MNIST(), MNISTModel(restore="models/mnist", use_log=True)
    origImgs, origLabels, origImgID = util.generate_attack_data_set(data, model, MGR)

    delImgAT_Init = np.zeros(origImgs[0].shape)
    objfunc = ObjectiveFunc.OBJFUNC(MGR, model, origImgs, origLabels)

    MGR.Add_Parameter('eta', MGR.parSet['alpha']/origImgs[0].size)
    MGR.Log_MetaData()

    if(MGR.parSet['optimizer'] == 'ZOSVRG'):
        delImgAT = svrg.ZOSVRG(delImgAT_Init, MGR, objfunc)
    elif(MGR.parSet['optimizer'] == 'ZOSGD'):
        delImgAT = sgd.ZOSGD(delImgAT_Init, MGR, objfunc)
    else:
        print('Please specify a valid optimizer')


    for idx_ImgID in range(MGR.parSet['nFunc']):
        currentID = origImgID[idx_ImgID]
        orig_prob = model.model.predict(np.expand_dims(origImgs[idx_ImgID], axis=0))
        advImg = np.tanh(np.arctanh(origImgs[idx_ImgID]*1.9999999)+delImgAT)/2.0
        adv_prob  = model.model.predict(np.expand_dims(advImg, axis=0))

        suffix = "id{}_Orig{}_Adv{}".format(currentID, np.argmax(orig_prob), np.argmax(adv_prob))
        util.save_img(advImg, "{}/Adv_{}.png".format(MGR.parSet['save_path'], suffix))
    util.save_img(np.tanh(delImgAT)/2.0, "{}/Delta.png".format(MGR.parSet['save_path']))

    sys.stdout.flush()
    MGR.logHandler.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-optimizer' , default='ZOSVRG', help="choose from ZOSVRG and ZOSGD")
    parser.add_argument('-q', type=int, default=1, help="Number of random vectors to average over for each gradient estimation")
    parser.add_argument('-alpha', type=float, default=1.0, help="Optimizer's step size being (alpha)/(input image size)")
    parser.add_argument('-M', type=int, default=50, help="Length of each stage/epoch")
    parser.add_argument('-nStage', type=int, default=1000, help="Number of stages/epochs")
    parser.add_argument('-const', type=float, default=5, help="Weight put on the attack loss")
    parser.add_argument('-nFunc', type=int, default=10, help="Number of images being attacked at once")
    parser.add_argument('-batch_size', type=int, default=5, help="Number of functions sampled for each iteration in the optmization steps")
    parser.add_argument('-mu', type=float, default=0.01, help="The weighting magnitude for the random vector applied to estimate gradients in ZOSVRG")
    parser.add_argument('-rv_dist', default='UnitSphere', help="Choose from UnitSphere and UnitBall")
    parser.add_argument('-target_label', type=int, default=1, help="The target digit to attack")
    args = vars(parser.parse_args())

    for par in args:
        MGR.Add_Parameter(par, args[par])

    MGR.Add_Parameter('save_path', 'Results/' + MGR.parSet['optimizer'] + '/')
    MGR.parSet['batch_size'] = min(MGR.parSet['batch_size'], MGR.parSet['nFunc'])

    main()
