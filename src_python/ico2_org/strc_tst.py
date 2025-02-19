#if __name__ == '__main__':
import numpy as np
from numpy.typing import NDArray
import random
try:
    from ico2.numericalc import (numerical_vector,
                                          numerical_vectors,
                                          projection_numerical_par,
                                          projection_par_numerical,
                                          #inside_outside_obj,
                                          inside_outside_tetrahedron,
                                          #inside_outside_tetrahedron_tau_v2,
                                          inside_outside_tetrahedron_rough,
                                          projection3_numerical,
                                          projection3_sets_numerical,
                                          projection_numerical,
                                          projection_sets_numerical,
                                          projection_sets_par_numerical,
                                          projection_sets_par_numerical_normalized,
                                          get_internal_component_numerical,
                                          get_internal_component_sets_numerical,
                                          length_numerical,
                                          )
    from ico2.symmetry_numerical import (generator_obj_symmetric_vector_specific_symop,
                                                  generator_obj_symmetric_vector_specific_symop_1,
                                                  generator_obj_symmetric_vectors_specific_symop_1,
                                                  generator_obj_symmetric_vectors_specific_symop,
                                                  generator_equivalent_numeric_vector_specific_symop,
                                                  )
    from ico2.symmetry import (generator_obj_symmetric_obj,
                                        generator_obj_symmetric_obj_specific_symop,
                                        site_symmetry_and_coset,
                                        icosasymop_array,
                                        icosasymop3_array,
                                        #equivalent_sites_unit_cell,
                                        equivalent_sites_with_centring,
                                        generator_equivalent_vec,
                                        get_index_of_symmetry_operation_for_equivalent_vectors,
                                        symmetry_operations_axial_vector,
                                        symmetry_operations_axial_vectors,
                                        )
    from ico2.projection import projection
    from ico2.math1 import (mul_vector,
                                     mul_vectors
                                     )
    from ico2.utils import (shift_object,
                                    )
except ImportError:
    print('import error in strc.py\n')

v0 = np.array([[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1],[ 0, 0, 1]])
v1 = np.array([[ 1, 0, 2],[-1, 0, 2],[-1, 0, 2],[-1, 0, 2],[-1, 0, 2],[-1, 0, 2]])
v2 = np.array([[ 1, 0, 2],[-1, 0, 2],[-1, 0, 2],[ 1, 0, 2],[-1, 0, 2],[-1, 0, 2]])
v3 = np.array([[ 1, 0, 2],[-1, 0, 2],[-1, 0, 2],[ 0, 0, 2],[-1, 0, 2],[ 0, 0, 2]])
od0 = np.vstack([v0,v1,v2,v3]).reshape(1,4,6,3)


x1= np.array([0, 0, -1, 0, 0, 0],dtype=np.float64) #5f
x2= np.array([0, 1, -1, 1, 0, 0],dtype=np.float64) #3f
x3= np.array([0, 1, -1, 0, 0, 0],dtype=np.float64) #2f



indx_site_sym,indx_coset=site_symmetry_and_coset(v0,'p',0)

#
# symmetry operation on x1,x2,x3 for each subdevided OD. 
#
# in the independent OD (obj)
v1_=generator_obj_symmetric_vector_specific_symop(x1,V1,indx_site_sym) # 5f
v2_=generator_obj_symmetric_vector_specific_symop(x2,V1,indx_site_sym) # 3f
v3_=generator_obj_symmetric_vector_specific_symop(x3,V1,indx_site_sym) # 2f
#
# in the ODs at equivalent positions
v1_=generator_obj_symmetric_vectors_specific_symop(v1_,V1,indx_coset)
v2_=generator_obj_symmetric_vectors_specific_symop(v2_,V1,indx_coset)
v3_=generator_obj_symmetric_vectors_specific_symop(v3_,V1,indx_coset)

ve1=projection_sets_par_numerical_normalized(v1_)
ve2=projection_sets_par_numerical_normalized(v2_)
ve3=projection_sets_par_numerical_normalized(v3_)

for v in ve1[0]:
    print('%8.6f %8.6f %8.6f (%8.6f)'%(v[0],v[1],v[2],np.linalg.norm(v)))
    
    
print('===============')
#
# symmetry operation on x1,x2,x3 for each subdevided OD. 
#
ve1=projection_sets_par_numerical_normalized(x1)
ve2=projection_sets_par_numerical_normalized(x2)
ve3=projection_sets_par_numerical_normalized(x3)
#
# in the independent OD (obj)
v1_=generator_obj_symmetric_vector_specific_symop_1(ve1,V2,indx_site_sym,'normal') # 5f
v2_=generator_obj_symmetric_vector_specific_symop_1(ve2,V2,indx_site_sym,'normal') # 3f
v3_=generator_obj_symmetric_vector_specific_symop_1(ve3,V2,indx_site_sym,'normal') # 2f
#
# in the ODs at equivalent positions
v1_=generator_obj_symmetric_vectors_specific_symop_1(v1_,V2,indx_coset,'normal') # 5f
v2_=generator_obj_symmetric_vectors_specific_symop_1(v2_,V2,indx_coset,'normal') # 3f
v3_=generator_obj_symmetric_vectors_specific_symop_1(v3_,V2,indx_coset,'normal') # 2f

i2=0
i3=0
mxe1_=v1_[i2][i3]
mxe2_=v2_[i2][i3]
mxe3_=v3_[i2][i3]
vec=np.array([mxe1_,mxe2_,mxe3_])
print('vec:',vec)
mu=np.array([1,0,0])
mu_=vec@mu
print(mu_)
