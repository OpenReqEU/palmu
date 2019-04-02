#!/usr/bin/env python
# coding: utf-8

# In[144]:


import pandas as pd 
import numpy as np

import json

import os


# In[145]:


# for embeddings
EMB_D = 200 
def loadGloveModel(gloveFile):
    print("Loading Glove Model")
    f = open(gloveFile,'r')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print("Done.",len(model)," words loaded!")
    return model

model_glove = loadGloveModel( "./data/glove.6B.200d.txt")


# In[146]:


files = os.listdir( "./data/")
files_json = [ "./data/"+f for f in files if ".json" in f ]


# In[147]:


files_json


# In[148]:


# list of qtfiles

def get_reqs( file ):
    
    data = ""
    with open( file , "r") as f:
        data = f.read()
        
    data = json.loads( data )
    print(" getting requirements from {}  - number of reqs: {}".format(  file , len(data["requirements"])) )
    return data["requirements"]

def get_embedding_txt( txt  , model ):
    
    #txt = req["text"]
    # here return the embedding
    txt = txt.lower() 
    words = txt.split(" ")
    embds = np.zeros( ( EMB_D))
    for w in words:
        if w in model.keys():
            emb = model[w]
        else:
            emb = np.zeros( (EMB_D) )
            
        embds += emb
        
    embds = embds / len( words )
    return embds

def get_embedding_components( req , model ):
    
    components_dict = req["requirementParts"][-1]
    #print( components_dict)
    if "text" not in components_dict.keys():
        return np.zeros( (EMB_D))
    
    text_list = components_dict["text"] #.replace( '"' , "")[1:-1] #.split(",") #  .split('"' )
    #print( "asdasda" , text_list )
    embs = np.zeros( (EMB_D))
    for word in text_list:
        if word in model.keys():
            emb = model[word]
            embs += emb 
    
    embs = embs / (len( text_list ) + 1 )
    return embs 
        

def get_embedding_com( req , model ):
    
    embs = np.zeros( ( EMB_D ))
    for comment in req["comments"]:
        
        txt = comment["text"]
        emb = get_embedding_txt( txt , model  )
        embs += emb 
        
    embs = embs/(len( req["comments"]  ) + 1 )
    return embs 

def get_embeddings( files_json ):
    # return the 
    # id - > embeddings correspondence 
    all_embeddings = []
    comment_embeddings = [] 
    index = 0
    mapping = {}
    print( files_json )
    for file in files_json:
        print( file )
        requirements = get_reqs( file )
        
        for req in requirements:
            
            if "text" in req.keys():
                #print( req["text"])
                name_emb = get_embedding_txt( req["name"] , model_glove )
                embedding = get_embedding_txt( req["text"]  , model_glove )
                comment_emb = get_embedding_com( req , model_glove )
                component_emb = get_embedding_components( req , model_glove )
                
                embedding =  0.1*name_emb + 0.5*embedding + 0.3*comment_emb + 0.1*component_emb 
                #embedding is a vector 
                mapping[req["id"]] = index
                index  = index + 1
                all_embeddings.append( embedding )
            
    return all_embeddings , mapping 
        # project is given by file
embs , mapp = get_embeddings( files_json )


# In[149]:


len(embs)


# In[150]:


embs = np.array( embs )


# In[151]:


embs.shape


# In[152]:


import pickle 


# In[153]:



pickle.dump( mapp ,   open( "./data/mappings200.map", "wb" ) , protocol=2 )


# In[154]:


np.save( "./data/embbedings200.npy" , embs   )


# In[127]:


mapp




