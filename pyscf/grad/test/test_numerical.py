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

import unittest
from pyscf import gto, scf, grad

def setUpModule():
    global h2
    h2 = gto.M(verbose=3,
                output='/dev/null',
                atom=[['H', 0, 0, 0],
                      ['H', 0, 0, 0.8]],
                basis='def2-svp')

def tearDownModule():
    pass

class KnownValues(unittest.TestCase):

    def test_h2_hf(self):
        hf = scf.RHF(h2)
        hf.run()

        numerical_method = grad.numerical.Gradients(hf)
        analytical_method = hf.nuc_grad_method()

        ana_g = analytical_method.kernel()
        num_g = numerical_method.kernel()

        print(num_g)
        print(ana_g)


if __name__ == "__main__":
    print("Full Tests for Numerical Gradients")
    unittest.main()