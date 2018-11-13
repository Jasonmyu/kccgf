'''
CCSD with k-point sampling
'''

import unittest
import numpy as np
from pyscf.pbc.tools.pbc import super_cell
from pyscf.pbc import gto, scf, cc
from pyscf import cc as mol_cc
import kpts_gf_ip
import gf
import kpts_gf_ea

class Test(unittest.TestCase):
    def test_ip(self):
        nmp = [1, 2, 1]
        cell = gto.Cell()
        cell.atom='''
        C 0.000000000000   0.000000000000   0.000000000000
        C 1.685068664391   1.685068664391   1.685068664391
        '''
        cell.basis = 'gth-szv'
        cell.pseudo = 'gth-pade'
        cell.a = '''
        0.000000000, 3.370137329, 3.370137329
        3.370137329, 0.000000000, 3.370137329
        3.370137329, 3.370137329, 0.000000000'''
        cell.unit = 'B'
        cell.verbose = 5
        cell.precision = 1e-12
        cell.build()

        cell.rcut *= 2.0
        cell.build()

        omegas = [-10.99479191]

        kpts = cell.make_kpts(nmp)
        kpts -= kpts[0]
        kmf = scf.KRHF(cell, kpts, exxdiv=None)
        kmf.kpts = kpts
        kmf.diis = None
        kmf.conv_tol_grad = 1e-8
        kmf.conv_tol = 1e-8
        ehf = kmf.kernel()
        kmf.analyze()
        mycc = cc.KRCCSD(kmf)
        mycc.ip_partition = None
        mycc.ea_partition = None
        mycc.conv_tol_normt = 1e-8
        mycc.conv_tol = 1e-8
        mycc.max_cycle=100
        mycc.kernel()
        p=[0,1,2,3]
        q=[0,1,2,3]
        gfunccc = kpts_gf_ip.OneParticleGF(mycc)
        kpts_gf = gfunccc.kernel(kpts,p,q,omegas)

        val = 0.0
        for k in range(2):
            for iocc in range(4):
                val += kpts_gf[k, iocc, iocc, 0]

        print "trace kpts gf ", val
        print 'molectrace     (-0.679067423896-0.000595898244016j)'

        assert (val.real)-(-0.679067423986) < 1e-5
        assert (val.imag)-(-0.000595898244016) < 1e-5

    def test_ea(self):
        nmp = [1, 2, 1]
        cell = gto.Cell()
        cell.atom='''
        C 0.000000000000   0.000000000000   0.000000000000
        C 1.685068664391   1.685068664391   1.685068664391
        '''
        cell.basis = 'gth-szv'
        cell.pseudo = 'gth-pade'
        cell.a = '''
        0.000000000, 3.370137329, 3.370137329
        3.370137329, 0.000000000, 3.370137329
        3.370137329, 3.370137329, 0.000000000'''
        cell.unit = 'B'
        cell.verbose = 5
        cell.precision = 1e-12
        cell.build()

        cell.rcut *= 2.0
        cell.build()

        omegas = [-10.99479191]

        kpts = cell.make_kpts(nmp)
        kpts -= kpts[0]
        kmf = scf.KRHF(cell, kpts, exxdiv=None)
        kmf.kpts = kpts
        kmf.diis = None
        kmf.conv_tol_grad = 1e-8
        kmf.conv_tol = 1e-8
        ehf = kmf.kernel()
        kmf.analyze()
        mycc = cc.KRCCSD(kmf)
        mycc.ip_partition = None
        mycc.ea_partition = None
        mycc.conv_tol_normt = 1e-8
        mycc.conv_tol = 1e-8
        mycc.max_cycle=100
        mycc.kernel()
        p=[4,5,6,7]
        q=[4,5,6,7]
        gfunccc = kpts_gf_ea.OneParticleGF(mycc)
        kpts_gf = gfunccc.kernel(kpts,p,q,omegas)

        val = 0.0
        for k in range(2):
            for iocc in range(4):
                val += kpts_gf[k, iocc, iocc, 0]

        print "trace kpts gf ", val
        print 'molectrace     (-0.629996672535-0.000512440754188j)'

        assert (val.real)-(-0.629996672535) < 1e-5
        assert (val.imag)-(0.000512440754188) < 1e-5


