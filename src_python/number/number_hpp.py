import numpy as np
from qnnum import Qnnum
from qnvec import Qnvec
import qnvec as qnv
#import geometry_n_hpp

from typing import TypeVar

type_ = TypeVar('type_',float, Qnnum)

# class number includes conventional number and qnnumber
# interface
class Number:
    @staticmethod
    def num(a, type_):
        if type_ == float:
            return float(a)
        elif type_ == Qnnum:
            return Qnnum([a, 0, 1])

    @staticmethod
    def eps(type_):
        if type_ == float:
            return np.finfo(float).eps
        elif type_ == Qnnum:
            return Qnnum.zero()

    @staticmethod
    def zero(type_):
        if type_ == float:
            return 0.0
        elif type_ == Qnnum:
            return Qnnum.zero()

    @staticmethod
    def inf(type_):
        if type_ == float:
            return np.inf
        elif type_ == Qnnum:
            return Qnnum.inf()

    @staticmethod
    def nan(type_):
        if type_ == float:
            return np.nan
        elif type_ == Qnnum:
            return Qnnum.nan()

    @staticmethod
    def is_inf(a, type_):
        if type_ == float:
            return a == Number.inf(float)
        elif type_ == Qnnum:
            return a == Qnnum.inf()

    @staticmethod
    def abs(a, type_):
        if type_ == float:
            return abs(a)
        elif type_ == Qnnum:
            return Qnnum.abs(a)

    @staticmethod
    def min(a, b, type_):
        if type_ == float:
            return min(a, b)
        elif type_ == Qnnum:
            return Qnnum.min(a, b)

    @staticmethod
    def max(a, b, type_):
        if type_ == float:
            return max(a, b)
        elif type_ == Qnnum:
            return Qnnum.max(a, b)

    @staticmethod
    def unique(qnv_, type_):
        print("type_ ",type_)
        shape=qnv_.shape
        print("qnv_.shape() ",shape)
        if type_ == float:
            return np.unique(qnv_)
        elif type_ == Qnnum :
            return Qnvec.uniquev(qnv_)

