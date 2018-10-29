## ZO_SVRG.py -- Perform ZOSVRG Optimization Algorithm
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

def ZOSVRG(delImgAT_Init, MGR, objfunc):

    best_Loss = 1e10
    best_delImgAT = delImgAT_Init


    delImgAT_kp1_S = delImgAT_Init
    for S_idx in range(1, MGR.parSet['nStage']+1):

        delImgAT_M_Sm1 = delImgAT_kp1_S
        g_S = objfunc.gradient_estimation(delImgAT_M_Sm1, MGR.parSet['mu'], MGR.parSet['q'])
        delImgAT_0_S = delImgAT_M_Sm1

        objfunc.evaluate(delImgAT_0_S, np.array([]), False)

        delImgAT_kp1_S = delImgAT_0_S
        for k in range(0, MGR.parSet['M']):
            delImgAT_k_S = delImgAT_kp1_S
            randBatchIdx = np.random.choice(np.arange(0, MGR.parSet['nFunc']), MGR.parSet['batch_size'], replace=False)
            v_k_S  = objfunc.gradient_estimation(delImgAT_k_S, MGR.parSet['mu'], MGR.parSet['q'], randBatchIdx)
            v_k_S -= objfunc.gradient_estimation(delImgAT_0_S, MGR.parSet['mu'], MGR.parSet['q'], randBatchIdx)
            v_k_S += g_S
            delImgAT_kp1_S = delImgAT_k_S - MGR.parSet['eta'] * v_k_S

            objfunc.evaluate(delImgAT_kp1_S, np.array([]), False)
            if((S_idx*MGR.parSet['M']+k)%100 == 0):
                    print('Stage Index: ', S_idx, '       M Index: ', k)
                    objfunc.print_current_loss()
            if(objfunc.Loss_Attack <= 1e-20 and objfunc.Loss_Overall < best_Loss):
                best_Loss = objfunc.Loss_Overall
                best_delImgAT = delImgAT_kp1_S
                #print('Updating best delta image record')

            MGR.logHandler.write('S_idx: ' + str(S_idx))
            MGR.logHandler.write(' m_idx: ' + str(k))
            MGR.logHandler.write(' Query_Count: ' + str(objfunc.query_count))
            MGR.logHandler.write(' Loss_Overall: ' + str(objfunc.Loss_Overall))
            MGR.logHandler.write(' Loss_Distortion: ' + str(objfunc.Loss_L2))
            MGR.logHandler.write(' Loss_Attack: ' + str(objfunc.Loss_Attack))
            MGR.logHandler.write(' Current_Best_Distortion: ' + str(best_Loss))
            MGR.logHandler.write('\n')

    return best_delImgAT
