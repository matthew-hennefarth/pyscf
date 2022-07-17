#!/usr/bin/env python
# Copyright 2014-2022 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Matthew Hennefarth <matthew.hennefarth@gmail.com>


import numpy as np

from pyscf.lib import logger, SinglePointScanner
from pyscf.grad import rhf as rhf_grad

def grad_elec(mc_grad, atmlst=None, verbose=None):
    log = logger.new_logger(mc_grad, verbose)
    mol = mc_grad.mol

    # Generate a list of mols, and then call the energy function on each of them for parallelization
    # Then, we store all of the data in the resulting shared_gradient object which has the same dimension as the mol.atms


def as_scanner(grad):
    pass


class Gradients(rhf_grad.GradientsMixin):

    as_scanner = as_scanner
    grad_elec = grad_elec

    def __init__(self, method, displacement=0.01):
        self.displacement = displacement

        if isinstance(method, SinglePointScanner):
            self.scanner = method
            mc = method.base
        
        elif isinstance(method, rhf_grad.GradientsMixin):
            self.scanner = method.base.as_scanner()
            mc = method.base

        elif getattr(method, "as_scanner", None):
            self.scanner = method.as_scanner()
            mc = method

        else:
            raise NotImplementedError('Electronic energy scanner of %s not available' %
                                      method)

        rhf_grad.GradientsMixin(mc)

    def kernel(self, displacement=None, atmlst=None, verbose=None):
        log = logger.new_logger(self, verbose)
        if atmlst is None:
            atmlst = self.atmlst
        else:
            self.atmlst = atmlst

        if self.verbose >= logger.WARN:
            self.check_sanity()
        
        if self.verbose >= logger.INFO:
            self.dump_flags()

        self.de = de = grad_elec(atmlst, log)

        self._finalize()
        return self.de

