import os

def wt_off(pnt, np, trg, nt, path, filename):
    with open(os.path.join(path, f"{filename}.off"), 'w') as f:
        f.write("OFF\n")
        f.write(f"{np} {nt} 0\n")  # number of independent points
        f.write("\n")
        for i in range(np):  # i1-th point
            f.write(f"{pnt[i][0]} {pnt[i][1]} 0.0\n")
        f.write("\n")
        for j in range(nt):  # j-th triangle
            f.write(f"3 {trg[j][0]} {trg[j][1]} {trg[j][2]}\n")

