# -*- coding: utf-8 -*-

"""

Created on Sun May 20 14:01:12 2018



@author: john

"""

import numpy as np

cimport numpy as np



def load_grid(np.ndarray [np.int64_t,ndim=2] coords, np.ndarray [np.float64_t,ndim=1] weights, np.ndarray [np.float64_t,ndim=3] grid, int num):


    cdef Py_ssize_t i

    

    for i in range(num):

        

        grid[coords[i,0],coords[i,1],coords[i,2]] +=  weights[i]

        

    return grid
