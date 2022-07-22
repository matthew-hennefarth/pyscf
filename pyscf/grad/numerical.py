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

def grad_num(mc_grad, displacement=None, atmlst=None, verbose=logger.INFO):
    log = logger.new_logger(mc_grad, verbose)
    mol = mc_grad.mol
    if atmlst is None:
        atmlst = range(mol.natm)

    de = np.zeros((len(atmlst), 3))

    # This is clearly trivially parallelizable and this loop can be unraveled.
    # I am going to leave this as it is now though and deal with
    # parallelization later. First, tests and general debug for usage.

    # TODO loop over the atmlst variables here instead
    for atomi, vec in enumerate(mol.atom_coords(unit='Bohr')):
        for dimj, dim in enumerate(vec):
            perturbed_coords = mol.atom_coords()

            forward = dim + displacement
            perturbed_coords[atomi][dimj] = forward

            forward_mol = mol.set_geom_(perturbed_coords, unit="Bohr", inplace=False)
            forward_mol.build()
            perturbed_energy_forward = mc_grad.scanner(forward_mol)

            backward = dim - mc_grad.displacement
            perturbed_coords[atomi][dimj] = backward

            backward_mol = mol.set_geom_(perturbed_coords, unit="Bohr", inplace=False)
            backward_mol.build()
            perturbed_energy_backward = mc_grad.scanner(backward_mol)

            de[atomi][dimj] = (perturbed_energy_forward - perturbed_energy_backward)/(2*mc_grad.displacement)

    return de

def as_scanner(grad):
    pass


class Gradients(rhf_grad.GradientsMixin):

    as_scanner = as_scanner
    grad_num = grad_num

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

        super().__init__(mc)

    # Hook in to the method to dump out the displacement as well (not super necessary but nice to have..)
    # def dump_flags(self):
    #     pass

    def kernel(self, displacement=None, atmlst=None, verbose=None):
        log = logger.new_logger(self, verbose)
        if displacement is not None: 
            self.displacement = displacement

        if atmlst is None:
            atmlst = self.atmlst

        if self.verbose >= logger.WARN:
            self.check_sanity()
        
        if self.verbose >= logger.INFO:
            self.dump_flags()

        self.de = self.grad_num(atmlst, log)

        if self.mol.symmetry:
            self.de = self.symmetrize(self.de, atmlst)

        self._finalize()
        return self.de

