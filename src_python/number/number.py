import numpy as np
from qnnum import Qnnum
import qnnum

class Number:
    @staticmethod
    def num_double(a: int) -> float:
        c = float(a)
        return c

    @staticmethod
    def num_qnnum(a: int) -> Qnnum:
        d = [a, 0, 1]
        c = Qnnum(d)
        return c

    @staticmethod
    def eps_double() -> float:
        return np.finfo(float).eps

    @staticmethod
    def eps_qnnum() -> Qnnum:
        return Qnnum.zero()

    @staticmethod
    def zero_double() -> float:
        return 0.0

    @staticmethod
    def zero_qnnum() -> Qnnum:
        return qnnum.zero()

    @staticmethod
    def inf_double() -> float:
        return np.inf

    @staticmethod
    def inf_qnnum() -> Qnnum:
        return qnnum.inf()

    @staticmethod
    def nan_double() -> float:
        return np.nan

    @staticmethod
    def nan_qnnum() -> Qnnum:
        return qnnum.nan()

    @staticmethod
    def is_inf_double(a: float) -> bool:
        return a == Number.inf_double()

    @staticmethod
    def is_inf_qnnum(a: Qnnum) -> bool:
        return a == qnnum.inf()

    @staticmethod
    def abs_double(a: float) -> float:
        return abs(a)

    @staticmethod
    def abs_qnnum(a: Qnnum) -> Qnnum:
        return qnnum.abs(a)

    @staticmethod
    def min_double(a: float, b: float) -> float:
        return min(a, b)

    @staticmethod
    def min_qnnum(a: Qnnum, b: Qnnum) -> Qnnum:
        return qnnum.min(a, b)

    @staticmethod
    def max_double(a: float, b: float) -> float:
        return max(a, b)

    @staticmethod
    def max_qnnum(a: Qnnum, b: Qnnum) -> Qnnum:
        return qnnum.max(a, b)

