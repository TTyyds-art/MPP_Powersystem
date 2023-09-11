from dataclasses import dataclass
from dataclasses import dataclass
from typing import Union, List, Optional, Tuple

import numpy
from numpy import exp, conj, r_

from pandapower.pypower.idx_bus import PD, QD
from pandapower.pypower.idx_gen import PG, QG
from pandapower.pypower.makeSbus import makeSbus
from pandapower.pypower.makeYbus import makeYbus
from .critical_region import CriticalRegion
from .mpQCQP_program_0731 import MPQCQP_Program
from .mplp_program import MPLP_Program
from .mpmilp_program import MPMILP_Program
from .mpmiqp_program import MPMIQP_Program
from .mpqp_program import MPQP_Program


def pow_flow_error(program, x_star, theta_point):
    '''
    通过计算功率流方程的误差来判断是解析式计算的解的误差有多大
    :param x_star: [θ, V, P, Q]
    :param theta_point:  load
    :return: error
    '''

    ppc = program.om.get_ppc()
    baseMVA, bus, gen, branch, gencost = \
        ppc["baseMVA"], ppc["bus"], ppc["gen"], ppc["branch"], ppc["gencost"]
    # 将x_star分解为θ, V, P, Q，其中它们各占四分之一
    Va = x_star[0:int(len(x_star) / 4)]
    Vm = x_star[int(len(x_star) / 4):int(len(x_star) / 2)]
    Pg = x_star[int(len(x_star) / 2):int(len(x_star) / 4 * 3)]   ## active generation in p.u.
    Qg = x_star[int(len(x_star) / 4 * 3):len(x_star)]   ## reactive generation in p.u.

    ## ----- evaluate constraints -----
    V = Vm * exp(1j * Va)

    bus[:, PD]  = theta_point[:int(len(theta_point) / 2)]  ## active load in p.u.
    bus[:, QD] = theta_point[int(len(theta_point) / 2):len(theta_point)]  ## reactive load in p.u.
    ## build admittance matrices
    Ybus, Yf, Yt = makeYbus(baseMVA, bus, branch)

    ## put Pg & Qg back in gen
    gen[:, PG] = Pg * baseMVA  ## active generation in MW
    gen[:, QG] = Qg * baseMVA  ## reactive generation in MVAr
    Sbus = makeSbus(baseMVA, bus, gen, vm=None)
    ## evaluate power flow equations
    mis = V * conj(Ybus * V) - Sbus

    g = r_[ mis.real,            ## active power mismatch for all buses
            mis.imag ]           ## reactive power mismatch for all buses

    return numpy.linalg.norm(g) # 返回误差的二范数,公式：||g||_2 = sqrt(g1^2 + g2^2 + ... + gn^2)
@dataclass
class SolutionTorch:
    """The Solution object is the output of multiparametric solvers, it contains all the critical regions as well
    as holds a copy of the original problem that was solved. """

    program: Union[MPLP_Program, MPQP_Program, MPMIQP_Program, MPMILP_Program, MPQCQP_Program]
    critical_regions: List[CriticalRegion]

    def __init__(self, program: Union[MPLP_Program, MPQCQP_Program], critical_regions: List[CriticalRegion],
                 is_overlapping=False):
        """
        The explicit solution associated with

        :param program: The multiparametric program that is considered here
        :param critical_regions: The list of critical regions in the solution
        :param is_overlapping: A Flag that tells the point location routine that there are overlapping critical regions
        """
        self.program = program
        self.critical_regions = critical_regions
        self.is_overlapping = is_overlapping

    def add_regions(self, region: Union[CriticalRegion, List[CriticalRegion], Tuple[CriticalRegion, ...]]) -> None:
        """
        Adds a region to the solution

        :param region: region to add to the solution
        :return: None
        """
        if isinstance(region, (list, tuple)):  # Check if b is a list or tuple
            self.critical_regions.extend(region)
        else:
            self.critical_regions.append(region)


    def evaluate(self, theta_point: numpy.ndarray) -> Optional[numpy.ndarray]:
        """
        returns the optimal x* from the solution, if it exists

        :param theta_point: an uncertainty realization
        :return: the calculated x* from theta
        """

        cr = self.get_region(theta_point)

        if cr is None:
            return None
        elif isinstance(cr, dict):
            return cr

        return cr.evaluate(theta_point)

    def get_region(self, theta_point: numpy.ndarray) -> Optional[CriticalRegion]:
        """
        Find the critical region in the solution that corresponds to the theta provided

        The method finds all critical regions that the solution is inside and returns the solutions, x*, with the lowest
        objective function of all of these regions.

        In the case of no overlap we can make a shortcut

        :param theta_point: an uncertainty realization
        :return: the region that contains theta
        """
        if self.is_overlapping:
            return self.get_region_overlap(theta_point)
        else:
            return self.get_region_no_overlap(theta_point)

    def get_region_no_overlap(self, theta_point: numpy.ndarray) -> Optional[CriticalRegion]:
        """
        Find the critical region in the solution that corresponds to the provided theta, assumes that no critical regions overlap

        :param theta_point:
        :return:
        """
        for region in self.critical_regions:
            if region.is_inside(theta_point):
                return region

        # if no regions, find the smallest distance between the points and regions
        smallest_distance = {'': float(numpy.inf)}
        for region in self.critical_regions:
            distance_cr_point = region.distance(theta_point)
            if list(distance_cr_point.values())[0] < list(smallest_distance.values())[0]:
                smallest_distance = distance_cr_point
        if list(smallest_distance.values())[0] >= 1e3:
            return None
        else:
            return smallest_distance


    def get_region_overlap(self, theta_point: numpy.ndarray) -> Optional[CriticalRegion]:
        """
        Find the best critical region in the solution that corresponds to the provided theta

        :param theta_point: realization of uncertainty
        :return: the critical region that that theta is in with the lowest objective value or none
        """

        # start with the worst value possible for the best objective and without a selected cr
        best_objective = float("inf")
        best_cr = None

        for region in self.critical_regions:
            # check if theta is inside the critical region
            if region.is_inside(theta_point):
                # we are inside the critical region now evaluate x* and f*
                x_star = region.evaluate(theta_point)
                error = pow_flow_error(program=self.program, x_star=x_star, theta_point=theta_point)
                # if better then update
                if error <= best_objective:  # 选择最小的误差的解
                    best_cr = region
                    best_objective = error

        return best_cr

    # def save(self, file_path):
    #     with open(file_path, 'wb') as f:
    #         pickle.dump(self.__dict__, f)
    #
    # @classmethod
    # def load(cls, file_path):
    #     with open(file_path, 'rb') as f:
    #         instance_dict = pickle.load(f)
    #         return cls(**instance_dict)

