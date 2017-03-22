# -*- coding:utf-8 -*-
"""
    This module provide several combine method
"""

def combine_avg(X, Y, Cxy, times):
    return (X*float(times)+Cxy.dot(Y))/float(times+1)
