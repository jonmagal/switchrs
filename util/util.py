# -*- coding: utf8 -*-
'''
Created on 09/09/2014

@author: Jonathas Magalhães
'''


def save_file(file_name, content):
    with open(file_name, 'w') as file_:
        file_.write(content)