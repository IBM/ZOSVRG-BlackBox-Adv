## ZO_SGD.py -- Perform ZOSGD Optimization Algorithm
##
## Copyright (C) 2018, IBM Corp
##                     PaiShun Ting <paishun@umich.edu>
##                     Pin-Yu Chen <Pin-Yu.Chen@ibm.com>
##                     Sijia Liu <sijia.liu@ibm.com>
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

import numpy as np

np.random.seed(2018)

def ZOSGD(delImgAT_Init, MGR, objfunc):


    T = MGR.parSet['nStage']*MGR.parSet['M']

    best_Loss = 1e10;
    best_delImgAT = delImgAT_Init
    curret_delImgAT = delImgAT_Init

    for T_idx in range(T):


        randBatchIdx = np.random.choice(np.arange(0, MGR.parSet['nFunc']), MGR.parSet['batch_size'], replace=False)
        F_estimate  = objfunc.gradient_estimation(curret_delImgAT, MGR.parSet['mu'], MGR.parSet['q'], randBatchIdx)
        curret_delImgAT = curret_delImgAT - MGR.parSet['eta'] * F_estimate


        objfunc.evaluate(curret_delImgAT, np.array([]), False)
        if(T_idx%100 == 0):
            print('Iteration Index: ', T_idx)
            objfunc.print_current_loss()
        if(objfunc.Loss_Attack <= 1e-20 and objfunc.Loss_Overall < best_Loss):
            best_Loss = objfunc.Loss_Overall
            best_delImgAT = curret_delImgAT
            #print('Updating best delta image record')

        MGR.logHandler.write('Iteration Index: ' + str(T_idx))
        MGR.logHandler.write(' Query_Count: ' + str(objfunc.query_count))
        MGR.logHandler.write(' Loss_Overall: ' + str(objfunc.Loss_Overall))
        MGR.logHandler.write(' Loss_Distortion: ' + str(objfunc.Loss_L2))
        MGR.logHandler.write(' Loss_Attack: ' + str(objfunc.Loss_Attack))
        MGR.logHandler.write(' Current_Best_Distortion: ' + str(best_Loss))
        MGR.logHandler.write('\n')
    return best_delImgAT
