import time
#from crsys import crsys_init
from qnnum import qnnum_init, Qnnum, zero, one, any, inf, nan, abs, printqnn, copy
import crsys as crs

if __name__ == "__main__":
    isys = 4  # for octagonal
    crs.crsys_init(isys)
    qnnum_init()  # qnnum init
    N = crs.N
    print("N", N)
    begin = time.time()

    qnn0 = zero()
    printqnn("qnn0", qnn0)
    qnn1 = one()
    printqnn("qnn1", qnn1)
    qnn2 = qnn1
    d1 = [2, 0, 1]
    qnn3 = Qnnum(d1)
    printqnn("qnn2", qnn2)
    printqnn("qnn3", qnn3)

    print("true" if qnn1 == qnn2 else "false")  # for boolean true and false
    b0 = (qnn1 == qnn2)
    b1 = (qnn0 == qnn1)
    b2 = (qnn0 > qnn1)
    b3 = (qnn0 < qnn1)
    print("qnn1==qnn2:", b0)
    print("qnn0==qnn1:", b1)
    print("qnn0>qnn1:", b2)
    print("qnn0<qnn1:", b3)
    d2 = [1, 1, 2]
    qnn4 = any(d2)
    printqnn("qnn4", qnn4)
    b4 = (qnn4 > qnn1)
    b5 = (qnn4 < qnn1)
    b6 = (qnn4 == qnn1)
    print("qnn4>qnn1:", b4)
    print("qnn4<qnn1:", b5)
    print("qnn4==qnn1:", b6)
    print("qnn4>=qnn1:", (qnn4 >= qnn1))
    print("qnn4<=qnn1:", (qnn4 <= qnn1))
    print("qnn4>=qnn4:", (qnn4 >= qnn4))
    print("qnn4<=qnn4:", (qnn4 <= qnn4))
    print()

    qnn5 = -qnn4
    qnn6 = qnn1 + qnn2
    qnn7 = qnn1 - qnn2
    qnn8 = qnn1 * qnn3
    qnn9 = qnn1 / qnn3
    printqnn("qnn5=-qnn4", qnn5)
    printqnn("qnn1+qnn2", qnn6)
    printqnn("qnn1-qnn2", qnn7)
    printqnn("qnn1*qnn3", qnn8)
    printqnn("qnn1/qnn3", qnn9)
    d3 = [1, 1, 2]
    qnn10 = any(d3)
    qnn11 = qnn1 / qnn5
    qnn12 = qnn5 / qnn5
    qnn13 = qnn5 * 2
    qnn14 = qnn5 / 2
    printqnn("qnn10", qnn10)
    printqnn("qnn1/qnn5", qnn11)
    printqnn("qnn5/qnn5", qnn12)
    printqnn("qnn5*2", qnn13)
    printqnn("qnn5/2", qnn14)
    qnn15 = qnn5
    qnn15 += qnn5
    printqnn("qnn5+=qnn5", qnn15)
    qnn16 = qnn5
    qnn16 -= qnn5
    printqnn("qnn5-=qnn5", qnn16)
    qnn17 = qnn5
    qnn17 *= qnn5
    printqnn("qnn5*=qnn5", qnn17)
    qnn18 = qnn5
    qnn18 /= qnn5
    printqnn("qnn5/=qnn5", qnn18)

    infp = inf()
    infn = -inf()
    nan0 = nan()
    printqnn("inf", infp)
    printqnn("-inf", infn)
    printqnn("nan", nan0)
    absinfn = abs(infn)
    printqnn("abs(-inf)", absinfn)
    print("inf<qnn1", (infp < qnn1))
    print("inf>qnn1", (infp > qnn1))
    print("-inf<qnn1", (infn < qnn1))
    print("-inf>qnn1", (infn > qnn1))

    qnt1 = zero(); qnt2 = zero(); qnt3 = zero(); qnt4 = zero()
    for i in range(100000):
        qnt1 = qnn1+qnn2
        qnt2 = qnn1-qnn2
        qnt3 = qnn1*qnn3
        qnt4 = qnn1/qnn3

        #qnt1 = copy(qnn1 + qnn2)
        #qnt2 = copy(qnn1 - qnn2)
        #qnt3 = copy(qnn1 * qnn3)
        #qnt4 = copy(qnn1 / qnn3)

    end = time.time()

    print("elapsed time =", (end - begin) * 1000, "[ms]")
    print("elapsed time =", (end - begin) * 1000000, "[µs]")
    # print("elapsed time =", (end - begin) * 1000000000, "[ns]")

