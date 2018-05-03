#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 19:28:11 2018

@author: kevin
Extract each file read its NXML and place figures in correct spot.
Since we don't know the class we can set its class to -1
"""

import fnmatch
import os
import tarfile
from bs4 import BeautifulSoup
from HTMLParser import HTMLParser
import pandas as pd
#import FTP_download
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import Session
from sqlalchemy import create_engine
WORKING_DIRECTORY = '/media/kevin/disk/Pubmed_extract_test/'
def extract(fname):
    if (fname.endswith("tar.gz")):
        tar = tarfile.open(fname, "r:gz")
        tar.extractall(path='/'.join(fname.split('/')[0:-1]) + '/')
        tar.close()

class MLStripper(HTMLParser):
    def __init__(self):
        self.reset()
        self.fed = []
    def handle_data(self, d):
        self.fed.append(d)
    def get_data(self):
        return ''.join(self.fed)

def strip_tags(html):
    s = MLStripper()
    s.feed(html)
    return s.get_data()

#find all tars
def extract_tars():
    matches = []
    for root, dirnames, filenames in os.walk(WORKING_DIRECTORY):
        for filename in fnmatch.filter(filenames, '*.tar.gz'):
            matches.append(os.path.join(root, filename))
    #extract and delete tar
    for match in matches:
        filename = match
        extract(filename)
        os.remove(filename)

def nxmls_to_list():
    #find all nxmls
    nxmls = []
    for root, dirnames, filenames in os.walk(WORKING_DIRECTORY):
        for filename in fnmatch.filter(filenames, '*.nxml'):
            nxmls.append(os.path.join(root, filename))
    return nxmls
def images_to_list():
    #get images        
    images = []
    for root, dirnames, filenames in os.walk(WORKING_DIRECTORY):
        for filename in fnmatch.filter(filenames, '*.jpg'):
            images.append(os.path.join(root, filename))
        for filename in fnmatch.filter(filenames, '*.tif'):
            images.append(os.path.join(root, filename))
        for filename in fnmatch.filter(filenames, '*.tiff'):
            images.append(os.path.join(root, filename))        
    return images

def extract_meta_images(nxmls, images):
    #get info from figures
    articles = pd.DataFrame(columns=['PMC_id', 'fid','label', 'caption', 'location'])
    image_count = len(articles)#0
    for nxml in nxmls:
        blob = open(nxml).read()
        data = BeautifulSoup(blob, "lxml")
        PMC = data.find('article-id', {'pub-id-type':'pmc'}).text#move this somewhere else
        for fig in data.find_all('fig'):
            try:
                fid = fig.graphic.attrs['xlink:href']
            except:
                fid = "-1"
            try:
                caption = fig.caption.text
            except:
                caption = "query(Film.id, func.count(FilmComment.id))-1"
            location = [s for s in images if str(fid) in s]
            try:
                label = fig.label.text
            except:
                label = "-1"# label not found check in the caption
            articles.loc[image_count] = {'PMC_id': PMC, 'fid': fid,'label':label, 'caption': caption, 'location': location[0]}
            image_count += 1
    return articles

def insert_database(fig_df):
    Base = automap_base()
    # engine, suppose it has two tables 'user' and 'address' set up
    engine = create_engine('mysql+mysqldb://root:toor@127.0.0.1/Pubmed_Images?charset=utf8mb4', pool_recycle=3600) # connect to server
    # reflect the tables
    Base.prepare(engine, reflect=True)
    # mapped classes are now created with names by default
    # matching that of the table name.
    Pubmed_Images = Base.classes.images
    #Start Session
    session = Session(engine)
    for fig in fig_df.iterrows():
        fig = fig[1]
        session.add(Pubmed_Images(PMC_id=int(fig['PMC_id']), fid=fig['fid'], label=fig['label'], caption=fig['caption'], location=fig['location'], class_id=-1))
    session.commit()

def get_PMCIDS(nxmls):
    pmcids = []
    for nxml in nxmls:
        pmcids.append(nxml.split('/')[-2].replace('PMC', ""))
    return pmcids
    

if __name__ == "__main__":
    #download via ftp
    #import ftplib
    #ftp = ftplib.FTP(mysite, username, password)
    #download_ftp_tree(ftp, remote_dir, WORKING_DIRECTORY)
    
    extract_tars()
    print("extracted Tars")
    nxml_list = nxmls_to_list()
    print("get unlabeled PMCIDS")
    Unlabeled_PMCIDS = get_PMCIDS(nxml_list)
    
    print("gathered nxmls to list")
    image_list = images_to_list()
    print("gathered images to list")
    image_df = extract_meta_images(nxml_list, image_list)
    print("Created dataframe")
    insert_database(image_df)
    print("Inserted in to database")
    
    