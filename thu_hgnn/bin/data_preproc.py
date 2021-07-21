'''
author: bg
goal: load datasets 
type:   
how: preprocessing pipeline using sklearn 
ref: zbase module - using sub-elements; not proper structure 
refactors: TODOs for non-fundus type of data
'''

import pickle
import abc 

import numpy as np 
from skimage import transform
from sklearn.preprocessing import StandardScaler 
from sklearn.base import BaseEstimator, TransformerMixin 

class FmapFileLoader(BaseEstimator, TransformerMixin):
    '''
    Read feature map from pkl file, resize if need be, feature select if desired  
    @input:     List of file paths 
    @output:    list of preprocessed results 
    @actions:   sklearn @ {fit, transform } 
    TODO: requirements/asserts ???? 
    '''     
    def __init__(self, resize_tuple=None):
        super().__init__()
        self.resize_tuple = resize_tuple 

    def fit(self, X, y=None): 
        return self  

    def transform(self, X, y=None):
        ### X is a list of file paths to load. File content is in pkl format 
        O_ = [] 
        for fp in X: 
            try:
                with open( fp, 'rb') as fd:
                    o = pickle.load(fd) 
                if self.resize_tuple is not None: # resize is 2D tuple; not checking but will fail otherwise
                    o = transform.resize(o, self.resize_tuple, anti_aliasing=True) 
                O_.append( o ) 
            except Exception as e: 
                pass 
        # print( "@FmapLoader ", len(O_))
        return O_

class FmapChannelSelector(BaseEstimator, TransformerMixin):
    '''
    Select fmap channelz 
    @input:     list of fmapz
    @output:    list of fmapz with selected channels only
    @actions:   slice fmapz array 
    TODO: requirements/asserts ???? 
    '''     
    def __init__(self, list_of_channels):
        super().__init__()
        self.list_of_channels = list_of_channels 

    def fit(self, X, y=None):
        return self  

    def transform(self, X, y=None):
        if self.list_of_channels  is None: ##NOOP 
            return X 
        O_ = []
        for x in X:
            try:
                o = [ x[:,:,c] for c in self.list_of_channels ] ## will raise error if x not 3D                
                O_.append( np.dstack(o) )
            except:
                pass 
        # print( "@FmapChannelSelector ", len(O_))
        return O_

class FlattenAndScale(BaseEstimator, TransformerMixin):
    '''
    @input:     data list or array, X and y=None (supervised or unsupervised ) 
    @output:    list of preprocessed results 
    @actions:   flatten input and run provided scaler 
    TODO: requirements/asserts ???? 
    '''     
    def __init__(self, sk_scaler=StandardScaler):
        super().__init__()
        self.sk_scaler = sk_scaler() 

    def fit(self, X, y=None):
        return self  

    def transform(self, X, y=None):
        O_ = [ np.array(x).reshape(-1) for x in X]
        # print(f"TO Scaler: {len(O_)} from {len(X)} @i=0.shape = {O_[0].shape}")#[0].shape
        O_ = self.sk_scaler.fit_transform(O_)  if (self.sk_scaler is not None and len(O_) > 0) else O_ 
        # print( "@Flatten_Scaler: ", len(O_))
        return O_ 

