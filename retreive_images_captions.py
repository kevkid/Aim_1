#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 17:23:20 2017

@author: kevin
"""
import os
from xml.dom import minidom
from bs4 import BeautifulSoup
import pandas as pd
from glob import glob
import urllib2
from HTMLParser import HTMLParser
import re
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import Session
from sqlalchemy import create_engine
#import lucene
#from org.apache.lucene.index import IndexOptions
#from org.apache.lucene.store import IOContext
#from org.apache.lucene.store import Directory
#from org.apache.lucene.analysis.standard import StandardAnalyzer
#from org.apache.lucene.document import Document, Field, FieldType
#from org.apache.lucene.search import IndexSearcher
#from org.apache.lucene.index import FieldInfo, IndexWriter, IndexWriterConfig, IndexReader, DirectoryReader
#from org.apache.lucene.queryparser.classic import QueryParser, MultiFieldQueryParser, QueryParserBase
#from org.apache.lucene.store import SimpleFSDirectory
#from org.apache.lucene.util import Version
#from java.io import File
#os.chdir('/home/kevin/Downloads/ftp.ncbi.nlm.nih.gov/pub/pmc/oa_package/06/00')
from HTMLParser import HTMLParser

CLASSES = {'bar': 0, 'gel': 1, 'map': 2, 'network': 3,  'plot': 4, 'text': 5, 
           'box': 6, 'heatmap': 7, 'medical': 8, 'nxmls': 9, 'screenshot': 10, 
           'topology': 11, 'diagram': 12, 'histology': 13, 'microscopy': 14, 'photo': 15, 
           'sequence': 16, 'tree': 17, 'fluorescence': 18, 'line': 19, 'molecular': 20, 
           'pie': 21, 'table': 22}

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

def untar(fname):
    import tarfile
    if (fname.endswith("tar.gz")):
        tar = tarfile.open(fname, "r:gz")
        tar.extractall()
        tar.close()

'''------------New methods to get caption data
steps:
    get all images, make set of pmcid
    for each pmcid find all images associated with it
    for each image get caption, label, location etc
What we will do is for each image, pull the nxml and get the meta data
'''
def get_pmcid(images):
    PMCIDs = []
    for img in images:
        PMCIDs.append(re.search(r'(?<=PMC)(.*)(?=__)', img).group())
    return list(set(PMCIDs))

def get_nxml(pmcids):#input: list of pmcids, output: list of nxmls that have the PMCid as their name.nxml
    #set dir to save nxmls
    num_pmcids = len(pmcids)
    directory = '/media/kevin/disk/convNetPanelsPaneled_sorted/nxmls/'
    import os
    if not os.path.exists(directory):
        os.makedirs(directory)
    nxmls = {}
    for i, pmcid in enumerate(pmcids):
        if i%10 == 0:
            print ("On article {} out of {}".format(i, num_pmcids))
        if not os.path.exists(str(directory + pmcid + '.nxml')):#check if exists
            f = urllib2.urlopen('https://www.ncbi.nlm.nih.gov/pmc/oai/oai.cgi?verb=GetRecord&identifier=oai:pubmedcentral.nih.gov:' + str(pmcid) + '&metadataPrefix=pmc')
            blob = f.read()
            data = BeautifulSoup(blob, "lxml")
            article_id = pmcid#data.find('article-id', {'pub-id-type':'pmcid'}).text
            #save nxml
            with open(str(directory + article_id + '.nxml'), "w") as code:
                code.write(str(data))
        else:
            print ("File already exists. Skipping...")
        nxmls[pmcid] = str(directory + pmcid + '.nxml')
    return nxmls
def extract_meta_images(nxmls, images):
    #get info from figures
    articles = pd.DataFrame(columns=['PMC_id', 'fid','label', 'caption', 'location', 'class'])
    image_count = len(articles)#0
    num_images = len(images)
    prev_PMC = ""
    for i, image in enumerate(images):
        if i%20==0:
            print ("On image {} of {}".format(i, num_images))
        PMC = re.search(r'(?<=PMC)(.*)(?=__)', image).group()
        if PMC != prev_PMC:#if the previous one is the same, use the data in memory instead of opening it again
            nxml = nxmls[PMC]#[s for s in nxmls if str(PMC) in s][0]#this needs to be made more efficient!
            blob = open(nxml).read()
            data = BeautifulSoup(blob, "lxml")
            prev_PMC = PMC#set it
        #fid = (re.search(r'.+(?=_process)',image.split('/')[-1]).group().replace("__", "/")).split('/')[-1]#figure id
        fid = re.search(r'(?<=\_\_)(.*?)(?=\_process)',image.split('/')[-1]).group()
        try:
            fig = data.find('graphic', {"xlink:href" : fid} ).parent
        except:
            print (fid)
            print (PMC)
            print (nxml)
            print (image)
            error = data.find('error').text
            print (error)
            if error == "The metadata format 'pmc' is not supported by the item or by the repository.":
                #https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi?id=PMC2581515
                f = urllib2.urlopen('https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi?id=' + str(PMC))
                blob = f.read()
                data = BeautifulSoup(blob, "lxml")
                try:
                    print (data.find('error').text)
                except:
                    print ("Something unexpected happened when trying to find the error tag... Check the file.")
        try:
            caption = fig.caption.text
        except:
            caption = "-1"
        try:
            label = fig.label.text
        except:
            label = "-1"# label not found check in the caption
        img_class = image.split('/')[-2]
        articles.loc[image_count] = {'PMC_id': PMC, 'fid': fid,'label':label, 
                    'caption': caption, 'location': image, 'class': img_class}
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
        session.add(Pubmed_Images(PMC_id=int(fig['PMC_id']), fid=fig['fid'], label=fig['label'], caption=fig['caption'], location=fig['location'], class_id=CLASSES[fig['class']]))
    session.commit()
    
if __name__ == '__main__':
    '''
    If the caption is the same then this means that it is from the same image.
    '''
    #set working dir
    working_directory = '/media/kevin/disk/convNetPanelsPaneled_sorted/'
    #get image list
    results = [y for x in os.walk(working_directory) for y in glob(os.path.join(x[0], '*.png'))]
    results.sort()
    #get pmcids
    print ("Get list of pmcid")
    PMCIDs = get_pmcid(results)
    #get nxmls
    print ("Get nxmls")
    nxmls = get_nxml(PMCIDs)
    print ("Extract meta data from images")
    images = extract_meta_images(nxmls, results)
    #error here need to fix or catch
    #for some reason this id has an extra underscore! Why I have no clue
    images['PMC_id'].iloc[11111] = 3153361
    insert_database(images)
    