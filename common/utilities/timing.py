# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 16:51:43 2019

@author: Tejas
"""

import functools
import time


def timeit(func):
    
    @functools.wraps(func)
    
    def newfunc(*args, **kwargs):
        startTime = time.time()
        retargs = func(*args, **kwargs)
        elapsedTime = time.time() - startTime
        print('function [{}] finished in {} ms'.format(
            func.__name__, int(elapsedTime * 1000)))
        return retargs

    return newfunc