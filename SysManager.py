## SysManager.py -- Manager for system parameters and logger
##
## Copyright (C) 2018, IBM Corp
##                     PaiShun Ting <paishun@umich.edu>
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
import Utils as util
import os

class SYS_MANAGER:

    def __init__(self):
        self.parSet = {}
        self.logHandler = 'None'

    def Add_Parameter(self, key, value):
        self.parSet[key] = value

        if(key == 'save_path'):
            os.system("mkdir -p {}".format(value))
            self.logHandler = open(value + '/log.txt', 'w+')


    def Log_MetaData(self):
        self.logHandler.write('Optimizer: ' + self.parSet['optimizer'] + '\n')
        self.logHandler.write('q: ' +  str(self.parSet['q']) + '\n')
        self.logHandler.write('eta: ' + str(self.parSet['eta']) + '\n')
        self.logHandler.write('M: ' +  str(self.parSet['M']) + '\n')
        self.logHandler.write('nStage: ' +  str(self.parSet['nStage']) + '\n')
        self.logHandler.write('const: ' + str(self.parSet['const']) + '\n')
        self.logHandler.write('number of functions: ' + str(self.parSet['nFunc']) + '\n')
        self.logHandler.write('batch size: ' + str(self.parSet['batch_size']) + '\n')
        self.logHandler.write('mu: ' + str(self.parSet['mu']) + '\n')
        self.logHandler.write('rv_dist: ' + str(self.parSet['rv_dist']) + '\n')
        self.logHandler.write('target_label: ' + str(self.parSet['target_label']) + '\n')
