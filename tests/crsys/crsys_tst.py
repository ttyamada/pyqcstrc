import sys
import crsys as crs

#octagonal
isys=4
crs_o=crs.crsys_init(isys)
N=crs_o.N
print("isys=",isys,"N=",N)

#decagonal
isys=3
crs_d=crs.crsys_init(isys)
N=crs_d.N
print("isys=",isys,"N=",N)

#dodecagonal
isys=5
crs_dd=crs.crsys_init(isys)
N=crs_dd.N
print("isys=",isys,"N=",N)
