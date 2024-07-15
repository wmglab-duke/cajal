import math

import numpy as np

from cajal.nrn import Section
from cajal.nrn.cells._cells import Axon
from cajal.units import um, uF, mm, mV, ms, C, ohm, cm, unitdispatch


# pylint: disable=attribute-defined-outside-init
class MRG(Axon):  # pylint: disable=too-many-instance-attributes
    """
    MRG Myelinated Axon (McIntyre et al. 2002).
    """

    name = "mrg"

    @unitdispatch
    def __init__(
        self,
        diameter: "um" = 5.7,
        length: "mm" = 35,
        x: "um" = 0,
        y: "um" = 0,
        z: "um" = 0,
        v_init: "mV" = -80.0,
        axonnodes=None,
        passive_end_nodes=True,
        geom_fit="combined",
        lemella_fit=0,
        try_exact=True,
        piecewise_geom_interp=True,
        gid=None,
        ap_count_t: "ms" = None,
        threshold: "mV" = 0,
        enforce_odd_axonnodes=False,
        interpolation_method=2,
        passive=False,
        passive_g=0.007,
        include_AP_monitors=True,
    ):
        self.try_exact = try_exact
        self.v_init = v_init
        self.ap_count_t = ap_count_t
        self.threshold = threshold
        self.enforce_odd_axonnodes = enforce_odd_axonnodes
        self.is_passive = passive
        self.passive_g = passive_g

        self.paranodes1 = None
        self.paranodes2 = None
        self.axoninter = None

        self.mycm = None
        self.mygm = None
        self.e_pas_Vrest = None

        # -- check valid fit method --
        valid_fit = {"fazan_proximal", "fazan_distal", "berthold", "friede", "combined"}
        if geom_fit in valid_fit:
            self.geom_fit = geom_fit
        else:
            raise ValueError(
                f"geom_fit option {geom_fit} not recognized. "
                f"Choose from: {valid_fit}"
            )

        # check valid fit method
        valid_lemella = {0, 1}
        if lemella_fit in valid_lemella:
            self.lemella_fit = lemella_fit
        else:
            raise ValueError(
                f"lemella_fit option {lemella_fit} not recognized. "
                f"Choose from: {valid_lemella}"
            )

        # -- set geometric params --
        gp = self.geometric_params(
            float(diameter),
            geom_fit,
            lemella_fit,
            try_exact,
            piecewise_geom_interp,
            interpolation_method,
        )

        self.axonD = gp["axonD"]
        self.nodeD = gp["nodeD"]
        self.paraD1 = gp["paraD1"]
        self.paraD2 = gp["paraD2"]
        self.deltax = gp["deltax"]
        self.paralength2 = gp["paralength2"]
        self.nl = gp["nl"]
        self.rhoa = gp["rhoa"]
        self.nodelength = gp["nodelength"]
        self.paralength1 = gp["paralength1"]
        self.space_p1 = gp["space_p1"]
        self.space_p2 = gp["space_p2"]
        self.space_i = gp["space_i"]
        self.Rpn0 = gp["Rpn0"]
        self.Rpn1 = gp["Rpn1"]
        self.Rpn2 = gp["Rpn2"]
        self.Rpx = gp["Rpx"]
        self.interlength = gp["interlength"]

        if self.deltax <= 0:
            raise ValueError(
                "Infeasible internode length {}um ".format(self.deltax)
                + "for fibre diameter {}um".format(diameter)
            )

        self.axonnodes = axonnodes or self._to_axonnodes(float(length.to("um")))
        self.passive_end_nodes = passive_end_nodes

        # -- compartment subsets --
        self.node = []
        self.MYSA = []
        self.FLUT = []
        self.STIN = []

        # -- complete build --
        super().__init__(x, y, z, diameter, length, gid, include_AP_monitors)

    def _to_axonnodes(self, length):
        n_inter = length / self.deltax
        if self.enforce_odd_axonnodes:
            return int(np.ceil(n_inter) // 2 * 2 + 1)
        return int(n_inter)

    @staticmethod
    def geometric_params(
        fd, geom_fit="combined", lemella_fit=0, try_exact=True, piecewise=True, interp=2
    ):
        """Generate geometric parameters for fiber model."""
        try:
            fd_try = fd if try_exact else 0
            params = MRG._classic_geometric_params(fd_try)
        except ValueError:
            if piecewise:
                if fd > 5.7:
                    params = MRG._geometric_params_1(fd)
                else:
                    params = MRG._geometric_params_2(fd, geom_fit, lemella_fit)
            else:
                if interp == 3:
                    params = MRG._geometric_params_schiefer(fd)
                elif interp == 2:
                    params = MRG._geometric_params_2(fd, geom_fit, lemella_fit)
                elif interp == 1:
                    params = MRG._geometric_params_1(fd)
                else:
                    raise ValueError("Choose interpolation method from 1, 2, or 3.")

        params.update(MRG._complete_geometric_params(**params))
        return params

    @staticmethod
    def _classic_geometric_params(fiber_d):
        if fiber_d == 1.0:
            axonD = 0.8
            nodeD = 0.7
            paraD1 = 0.7
            paraD2 = 0.8
            deltax = 100
            paralength2 = 5
            nl = 15
        elif fiber_d == 2.0:
            axonD = 1.6
            nodeD = 1.4
            paraD1 = 1.4
            paraD2 = 1.6
            deltax = 200
            paralength2 = 10
            nl = 30
        elif fiber_d == 5.7:
            axonD = 3.4
            nodeD = 1.9
            paraD1 = 1.9
            paraD2 = 3.4
            deltax = 500
            paralength2 = 35
            nl = 80
        elif fiber_d == 7.3:
            axonD = 4.6
            nodeD = 2.4
            paraD1 = 2.4
            paraD2 = 4.6
            deltax = 750
            paralength2 = 38
            nl = 100
        elif fiber_d == 8.7:
            axonD = 5.8
            nodeD = 2.8
            paraD1 = 2.8
            paraD2 = 5.8
            deltax = 1000
            paralength2 = 40
            nl = 110
        elif fiber_d == 10.0:
            axonD = 6.9
            nodeD = 3.3
            paraD1 = 3.3
            paraD2 = 6.9
            deltax = 1150
            paralength2 = 46
            nl = 120
        elif fiber_d == 11.5:
            axonD = 8.1
            nodeD = 3.7
            paraD1 = 3.7
            paraD2 = 8.1
            deltax = 1250
            paralength2 = 50
            nl = 130
        elif fiber_d == 12.8:
            axonD = 9.2
            nodeD = 4.2
            paraD1 = 4.2
            paraD2 = 9.2
            deltax = 1350
            paralength2 = 54
            nl = 135
        elif fiber_d == 14.0:
            axonD = 10.4
            nodeD = 4.7
            paraD1 = 4.7
            paraD2 = 10.4
            deltax = 1400
            paralength2 = 56
            nl = 140
        elif fiber_d == 15.0:
            axonD = 11.5
            nodeD = 5.0
            paraD1 = 5.0
            paraD2 = 11.5
            deltax = 1450
            paralength2 = 58
            nl = 145
        elif fiber_d == 16.0:
            axonD = 12.7
            nodeD = 5.5
            paraD1 = 5.5
            paraD2 = 12.7
            deltax = 1500
            paralength2 = 60
            nl = 150
        else:
            raise ValueError(
                "Please choose an axon diameter from the following: 5.7, "
                + "7.3, 8.7, 10.0, 11.5, 12.8, 14.0, 15.0, 16.0"
            )
        return {
            "axonD": axonD,
            "nodeD": nodeD,
            "paraD1": paraD1,
            "paraD2": paraD2,
            "deltax": deltax,
            "paralength2": paralength2,
            "nl": nl,
        }

    @staticmethod
    def _geometric_params_1(fd):
        # g = 1.473348e-04*fd**2 + 1.394606e-02*fd + 5.235060e-01
        axonD = 1.876226e-02 * fd**2 + 4.787487e-01 * fd + 1.203613e-01
        if fd > 1.5:
            nodeD = 6.303781e-03 * fd**2 + 2.070544e-01 * fd + 5.339006e-01
        else:
            nodeD = axonD * 1.9 / 3.4
        paraD1 = 6.303781e-03 * fd**2 + 2.070544e-01 * fd + 5.339006e-01
        paraD2 = 1.876226e-02 * fd**2 + 4.787487e-01 * fd + 1.203613e-01
        if fd >= 5.26:
            deltax = -8.215284e00 * fd**2 + 2.724201e02 * fd + -7.802411e02
        else:
            deltax = (500 / 5.7) * fd
        paralength2 = -1.989947e-02 * fd**2 + 3.016265e00 * fd + 1.743604e01
        nl = -3.889695e-01 * fd**2 + 1.487838e01 * fd + 9.721088e00
        return {
            "axonD": axonD,
            "nodeD": nodeD,
            "paraD1": paraD1,
            "paraD2": paraD2,
            "deltax": deltax,
            "paralength2": paralength2,
            "nl": nl,
        }

    @staticmethod
    def _geometric_params_2(fd, geom_fit, lemella_fit):
        if geom_fit == "fazan_proximal":
            axonD = 0.553 * fd + -0.024
        elif geom_fit == "fazan_distal":
            axonD = 0.688 * fd + -0.337
        elif geom_fit == "berthold":
            axonD = 0.0156 * fd**2 + 0.392 * fd + 0.188
        elif geom_fit == "friede":
            axonD = 0.684 * fd + 0.0821
        else:
            axonD = 0.621 * fd - 0.121
        nodeD = 0.321 * axonD + 0.37
        paraD1 = nodeD
        paraD2 = axonD
        deltax = -3.22 * fd**2 + 148 * fd + -128
        paralength2 = -0.171 * fd**2 + 6.48 * fd + -0.935
        if lemella_fit == 0:
            nl = math.floor(17.4 * axonD + -1.74)
        else:
            nl = math.floor(-1.17 * axonD**2 + 24.9 * axonD + 17.7)
        return {
            "axonD": axonD,
            "nodeD": nodeD,
            "paraD1": paraD1,
            "paraD2": paraD2,
            "deltax": deltax,
            "paralength2": paralength2,
            "nl": nl,
        }

    @staticmethod
    def _geometric_params_schiefer(fd):
        axonD = 0.889 * fd - 1.9104
        nodeD = 0.3449 * fd - 0.1484
        paraD1 = 0.3527 * fd - 0.1804
        paraD2 = 0.889 * fd - 1.9104
        deltax = 969.3 * np.log(fd) - 1144.6
        paralength2 = 2.5811 * fd + 19.59
        nl = 65.897 * np.log(fd) - 32.666
        return {
            "axonD": axonD,
            "nodeD": nodeD,
            "paraD1": paraD1,
            "paraD2": paraD2,
            "deltax": deltax,
            "paralength2": paralength2,
            "nl": nl,
        }

    @staticmethod
    def _complete_geometric_params(
        nodeD, paraD1, paraD2, axonD, deltax, paralength2, nl
    ):
        rhoa = 0.7e6  # [ohm-um]
        nodelength = 1.0  # Length of node of ranvier [um]
        paralength1 = 3  # Length of MYSA [um]
        space_p1 = 0.002  # Thickness of periaxonal space in MYSA [um]
        space_p2 = 0.004  # Thickness of periaxonal space in FLUT [um]
        space_i = 0.004  # Thickness of periaxonal space in STIN [um]
        Rpn0 = (rhoa * 0.01) / (
            np.pi * ((((nodeD / 2) + space_p1) ** 2) - ((nodeD / 2) ** 2))
        )
        Rpn1 = (rhoa * 0.01) / (
            np.pi * ((((paraD1 / 2) + space_p1) ** 2) - ((paraD1 / 2) ** 2))
        )
        Rpn2 = (rhoa * 0.01) / (
            np.pi * ((((paraD2 / 2) + space_p2) ** 2) - ((paraD2 / 2) ** 2))
        )
        Rpx = (rhoa * 0.01) / (
            np.pi * ((((axonD / 2) + space_i) ** 2) - ((axonD / 2) ** 2))
        )
        interlength = (deltax - nodelength - (2 * paralength1) - (2 * paralength2)) / 6
        return {
            "rhoa": rhoa,
            "nodelength": nodelength,
            "paralength1": paralength1,
            "space_p1": space_p1,
            "space_p2": space_p2,
            "space_i": space_i,
            "Rpn0": Rpn0,
            "Rpn1": Rpn1,
            "Rpn2": Rpn2,
            "Rpx": Rpx,
            "interlength": interlength,
            "nl": nl,
        }

    def create_sections(self):
        self.paranodes1 = 2 * (self.axonnodes - 1)  # MYSA paranodes
        self.paranodes2 = 2 * (self.axonnodes - 1)  # FLUT paranodes
        self.axoninter = 6 * (self.axonnodes - 1)  # STIN internodes

        for i in range(self.axonnodes):
            self.node.append(Section(name="node[%d]" % i, cell=self))

        for i in range(self.paranodes1):
            self.MYSA.append(Section(name="MYSA[%d]" % i, cell=self))

        for i in range(self.paranodes2):
            self.FLUT.append(Section(name="FLUT[%d]" % i, cell=self))

        for i in range(self.axoninter):
            self.STIN.append(Section(name="STIN[%d]" % i, cell=self))

    def build_topology(self):
        for i in range(self.axonnodes - 1):
            self.MYSA[2 * i].connect(self.node[i])
            self.FLUT[2 * i].connect(self.MYSA[2 * i])
            self.STIN[6 * i].connect(self.FLUT[2 * i])
            self.STIN[6 * i + 1].connect(self.STIN[6 * i])
            self.STIN[6 * i + 2].connect(self.STIN[6 * i + 1])
            self.STIN[6 * i + 3].connect(self.STIN[6 * i + 2])
            self.STIN[6 * i + 4].connect(self.STIN[6 * i + 3])
            self.STIN[6 * i + 5].connect(self.STIN[6 * i + 4])
            self.FLUT[2 * i + 1].connect(self.STIN[6 * i + 5])
            self.MYSA[2 * i + 1].connect(self.FLUT[2 * i + 1])
            self.node[i + 1].connect(self.MYSA[2 * i + 1])

    def define_geometry(self):
        for sec in self.node:
            sec.nseg = 1
            sec.diam = self.nodeD
            sec.L = self.nodelength

        for sec in self.MYSA:
            sec.nseg = 1
            sec.diam = float(self.diameter)
            sec.L = self.paralength1

        for sec in self.FLUT:
            sec.nseg = 1
            sec.diam = float(self.diameter)
            sec.L = self.paralength2

        for sec in self.STIN:
            sec.nseg = 1
            sec.diam = float(self.diameter)
            sec.L = self.interlength

    def define_biophysics(self):
        self.mycm = 0.1  # [uF/cm2]; lemella membrane
        self.mygm = 0.001  # [S/cm2]; lemella membrane
        self.e_pas_Vrest = -80.0
        diameter = float(self.diameter)

        for sec in self.node:
            sec.Ra = self.rhoa / 10000
            # Define biophysics of passive end nodes
            if (
                sec is self.node[0] or sec is self.node[self.axonnodes - 1]
            ) and self.passive_end_nodes:
                sec.cm = 2
                sec.insert("pas")
                sec.g_pas = 0.0001
                sec.e_pas = self.e_pas_Vrest
                sec.insert("extracellular")
                sec.xg[0] = self.mygm / (self.nl * 2)
                sec.xc[0] = self.mycm / (self.nl * 2)
            else:
                if self.is_passive:
                    sec.cm = 2
                    sec.insert("pas")
                    sec.g_pas = self.passive_g * 0.001
                    sec.e_pas = self.e_pas_Vrest
                    sec.insert("extracellular")
                    sec.xraxial[0] = self.Rpn0
                    sec.xg[0] = 1e10
                    sec.xc[0] = 0
                else:
                    sec.cm = 2
                    sec.insert("axnode_myel")
                    sec.insert("extracellular")
                    sec.xraxial[0] = self.Rpn0
                    sec.xg[0] = 1e10
                    sec.xc[0] = 0

        for sec in self.MYSA:
            self._apply_biophysics(sec, 0.001, self.paraD1, self.Rpn1, diameter)

        for sec in self.FLUT:
            self._apply_biophysics(sec, 0.0001, self.paraD2, self.Rpn2, diameter)

        for sec in self.STIN:
            self._apply_biophysics(sec, 0.0001, self.axonD, self.Rpx, diameter)

    def _apply_biophysics(self, sec, scale, secd, rp, diameter):
        sec.Ra = self.rhoa * (1 / (secd / diameter) ** 2) / 10000
        sec.cm = 2 * secd / diameter
        sec.insert("pas")
        sec.g_pas = scale * secd / diameter
        sec.e_pas = self.e_pas_Vrest
        sec.insert("extracellular")
        sec.xraxial[0] = rp
        sec.xg[0] = self.mygm / (self.nl * 2)
        sec.xc[0] = self.mycm / (self.nl * 2)

    def init_voltages(self):
        for sec in self.all:
            sec.v = self.v_init

    def init_AP_monitors(self, distance=None, threshold=None, t=None):
        length = float(self.length.to("um"))
        distance = (
            distance if distance is not None else [-0.4 * length, 0.4 * length] * um
        )
        threshold = threshold if threshold is not None else self.threshold
        t = t if t is not None else self.ap_count_t
        self.set_AP_monitors(distance=distance, threshold=threshold, t=t)


class Sundt(Axon):
    """
    Sundt. Provide length in mm.
    """

    name = "sundt"

    @unitdispatch
    def __init__(self, diameter: 'um' = 1.0, length: 'mm' = 10.05, x: 'um' = 0,
                 y: 'um' = 0, z: 'um' = 0, dx: 'um' = 50, temp: 'C' = 37,
                 v_init: 'mV' = -65, ap_count_t: 'ms' = None, threshold: 'mV' = 0,
                 axonnodes=None, gid=None, enforce_odd_axonnodes=False, 
                 check_single=False):

        self.dx = dx
        self.temp = temp
        self.v_init = v_init
        self.ap_count_t = ap_count_t
        self.threshold = threshold
        self.enforce_odd_axonnodes = enforce_odd_axonnodes
        self.ncompartments = axonnodes or self._to_axonnodes(float(length.to('um')))

        # -- compartment subsets --
        self.node = []

        super().__init__(x, y, z, diameter, length, gid, check_single=check_single)

    def _to_axonnodes(self, length):
        n_inter = int(length / self.dx)
        if self.enforce_odd_axonnodes:
            return int(np.ceil(n_inter) // 2 * 2 + 1)
        return int(n_inter)

    def create_sections(self):
        for i in range(self.ncompartments):
            self.node.append(Section(name='node[%d]' % i, cell=self))

    def build_topology(self):
        for i in range(self.ncompartments-1):
            self.node[i+1].connect(self.node[i])

    def define_geometry(self):
        for sec in self.node:
            sec.nseg = 1
            sec.diam = self.diameter
            sec.L = self.dx

    def define_biophysics(self):
        for node in self.node:
            node.insert('nahh')
            node.insert('borgkdr')  # insert delayed rectified K channels
            node.insert('pas')      # insert leak channels

            node.gnabar_nahh = 0.04
            node.mshift_nahh = -6        # NaV1.7/1.8 channelshift
            node.hshift_nahh = 6         # NaV1.7/1.8 channelshift
            node.gkdrbar_borgkdr = 0.04  # density of K channels
            node.ek = -90                # K equilibrium potential
            node.g_pas = 1 / 10000       # set Rm = 10000 ohms-cm2
            node.Ra = 100                # intracellular resistance
            node.e_pas = -65.0

            node.insert('extracellular')
            node.xg[0] = 1e10
            node.xc[0] = 0

    def init_voltages(self):
        for sec in self.all:
            sec.v = self.v_init

    def init_AP_monitors(self, distance=None, threshold=None, t=None):
        distance = distance if distance is not None else [-2, 2]*mm
        threshold = threshold if threshold is not None else self.threshold
        t = t if t is not None else self.ap_count_t
        self.set_AP_monitors(distance=distance, threshold=threshold, t=t)


class Linear(Axon):
    @unitdispatch
    def __init__(
        self,
        diameter: "um" = 1.0,
        length: "mm" = 10.05,
        x: "um" = 0,
        y: "um" = 0,
        z: "um" = 0,
        dx: "um" = 50,
        v_init: "mV" = 0,
        v_rest: "mV" = 0,
        cm: "uF" = 1,
        Ra: "ohm*cm" = 35.4,
        rm: "ohm*cm**2" = 1000,
        neumann_bcs=True,
        axonnodes=None,
        gid=None,
        enforce_odd_axonnodes=True,
    ):
        self.dx = dx
        self.v_init = v_init
        self.v_rest = v_rest
        self.enforce_odd_axonnodes = enforce_odd_axonnodes
        self.neumann_bcs = neumann_bcs
        self.cm = cm
        self.Ra = Ra
        self.rm = rm
        self.ncompartments = axonnodes or self._to_axonnodes(float(length.to("um")))

        # -- compartment subsets --
        self.node = []

        super().__init__(x, y, z, diameter, length, gid)

    def _to_axonnodes(self, length):
        n_inter = int(length / self.dx)
        if self.enforce_odd_axonnodes:
            return int(np.ceil(n_inter) // 2 * 2 + 1)
        return int(n_inter)

    def create_sections(self):
        for i in range(self.ncompartments):
            self.node.append(Section(name="node[%d]" % i, cell=self))

    def build_topology(self):
        for i in range(self.ncompartments - 1):
            self.node[i + 1].connect(self.node[i])

    def define_geometry(self):
        for sec in self.node:
            sec.nseg = 1
            sec.diam = self.diameter
            sec.L = self.dx

    def define_biophysics(self):
        for sec in self.node:
            sec.Ra = self.Ra
            sec.cm = self.cm

            sec.insert("pas")
            sec.g_pas = 1 / self.rm
            sec.e_pas = self.v_rest

            sec.insert("extracellular")
            sec.xg[0] = 1e10
            sec.xc[0] = 0

    def init_voltages(self):
        for sec in self.all:
            sec.v = self.v_init

    def init_AP_monitors(self):
        pass


# -- ALIASES --
MyelinatedAxon = MRG
CFiber = Sundt
