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
        for par, par_value in self.parSet.items():
            self.logHandler.write(par + ' ' + str(par_value) + '\n')
