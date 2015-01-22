# -*- coding: utf8 -*-
'''
Created on 09/09/2014

@author: Jonathas Magalhães
'''


def save_file(file_name, content):
    with open(file_name, 'w') as file_:
        file_.write(content)
        
def save_sheet(file_name, content, title):        
    import csv
    with open(file_name, 'w') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(title)
        for c in content:
            csv_writer.writerow(c)