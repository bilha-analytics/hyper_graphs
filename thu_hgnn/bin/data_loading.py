'''
author: bg
goal: load fundus data set as fmapz and as HGNN data types 
type: util  
how: 
ref: 
refactors: 
'''
import itertools
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.pipeline import Pipeline  
from sklearn.preprocessing import StandardScaler 

import data_preproc as dp 

from collections import defaultdict

import seaborn as sns 

'''
Config stuff 
'''
DSET_FILE = '../data/stare_fundus_graph_data.txt'
_CLASSEZ = ['Undef', 'Normal', 'DR', 'Other']


##======= 1. File IO ====================

def load_dset_listing():
    '''
    Fetch fmap item from listing 
    @return list of fmap metadata tuples of (fpath, disease, attr_code, y_label)
    '''
    with open(f"{DSET_FILE}", 'r') as fd:
        for ln in fd.readlines():
            yield list(ln.strip().split("\t"))


def show_fmap(fmap, has_origi=True, cmap=None, titlez=None, spacer=0.01,tfont=10):    
    '''
    Display the images/channels within a feature map image/file 
    '''
    # 1. generate list of images
    c = fmap.shape[2] ## origi image goes first 
    if (c >= 3) and has_origi:
        img_list =[ fmap[:,:,:3],] + [fmap[:,:, i+3] for i in range(c-3) ]
    else:
        img_list =[fmap[:,:, i] for i in range(c) ]
    
    # 2. plot the list
    n = len(img_list)
    nc = c 
    nr = n//nc + ( 0 if n%nc == 0 else 1) 
    ## image rows
    for i, img in enumerate(img_list):
        plt.subplot(nr, nc, (i+1) )
        plt.imshow( img, cmap=cmap)  #.astype('uint8')
        plt.axis('off')
        if titlez and (i<len(titlez)):
            plt.title( f"{titlez[i]}", fontsize=tfont ) #min(i, len(titlez)-1)
    plt.subplots_adjust(wspace=spacer, hspace=spacer)
    plt.show();


def fetch_fmap_images(X_paths, channz_ls=(0,), resize_tuple=(16,16), flatten=True, scaler=StandardScaler):
    '''
    Read image files; using Pipeline for now b/c was part of something else ELSE could be generator
    @input:
        - X-paths:      list of fmap paths from load_dset listing, 
        - channz_ls:    the fmap channels to use. default is green channel 
        - resize_tuple: resize Image or None . default is 16x16 img. Set to None for no resizing 
        - scaler:       rescale/normalize values. default is sklearn.StandardScaler. set to None for NOOP
    '''

    dpipe_load = [('loader', dp.FmapFileLoader(resize_tuple=resize_tuple) ), ]
    dpipe_chanz = [ (channz_ls[0], dp.FmapChannelSelector( list_of_channels=channz_ls[1] ) ),] if channz_ls is not None else [ ('c_all', dp.FmapChannelSelector( list_of_channels=None ) ),]
    dpipe_flat = [  ('flat', dp.FlattenAndScale(sk_scaler=scaler) ),] 

    dpipe = dpipe_load + dpipe_chanz + dpipe_flat if flatten else dpipe_load + dpipe_chanz
    # print(X_paths)
    # print(dpipe, "\n*****\n") 
    O_ = Pipeline( dpipe ).fit_transform(X_paths)
    return O_ 


def fetch_test_train(N=None, n_classes=2, perc_test=.3, 
                    mode_transductive=False,   
                    rseed=None):
    '''
    train_test split + generated X_features, X_node_attributes, y_labels + test split
    @input: - content listing txt file 
            - percent to make test cases
            - if transductive or inductive 
    @output:    2 tuples of (x_paths, y_lbls, x_attrs), one for test cases and other for training cases
    @action: load_dset_listing + mask y_labels for transductive
    '''
    np.random.seed(rseed)
    X_train, y_train, X_attr_train = [], [], []
    X_test, y_test, X_attr_test = [], [], []
    
    #all_D = list(load_dset_listing() ) if N is None else list([i for i in itertools.islice(load_dset_listing(), N)])
    all_D = list( load_dset_listing() )
    np.random.shuffle( all_D  )
    if N is not None:
        #all_D = all_D[:min(N, len(all_D))] 
        # force class balance @ y_lb has 2 classes 
        _DATA = []
        class_counters = defaultdict(int) 
        nn = N//n_classes
        for i in range(n_classes):
            class_counters[i] = nn 
        for rec in all_D:
            y_ = int(rec[-1]) 
            if class_counters[y_] != 0:
                _DATA.append( rec ) 
                class_counters[y_] -= 1
            if len(_DATA) >= N:
                break 
        all_D = _DATA      
    
    
    n = len(all_D)
    tn = int(n*.3) 
    get_x_path = lambda x: [r[0] for r in x]
    get_x_attr = lambda x: [r[1:-1] for r in x]
    get_y_lbl = lambda x: [int( r[-1] ) for r in x] 

    train, test = list(np.copy(all_D[:(n-tn)]) ), list(np.copy(all_D[(n-tn):] ))
    X_train, X_test = get_x_path(train), get_x_path(test) 
    X_attr_train, X_attr_test = get_x_attr(train), get_x_attr(test) 
    y_train, y_test = get_y_lbl(train), get_y_lbl(test) 

    if mode_transductive:
        X_train = np.hstack( [X_train, X_test] )         
        X_attr_train = np.vstack( [X_attr_train, X_attr_test] )
        
        y_train = np.hstack( [y_train, -1*np.ones_like(y_test) ] )
        # set to empties for upstream Ops 
        X_test, X_attr_test = [], []

    return (X_train, y_train, X_attr_train), (X_test, y_test, X_attr_test )


##======= 3. Sample stats ====================
def gen_pdframe(train_tuple, test_tuple, do_list_only=False):
    '''
    takes the output of fetch_test_train and puts it ina pd frame for analysis and visualization
    '''
    # A, B, C = [ list(train_tuple[0]) + list(test_tuple[0]), list(train_tuple[2]) + list(test_tuple[2]), list(train_tuple[1]) + list(test_tuple[1]) ]
    # _d = [ (a, b[0], b[1], c) for a, b, c in zip(A, B, C)]
    _d1 = [ (a, c[0], c[1], b, 'train')  for a, b, c in zip(*train_tuple) ]
    _d2 = [ (a, c[0], c[1], b, 'test')  for a, b, c in zip(*test_tuple) ]
    if not do_list_only:
        df = pd.DataFrame(_d1+_d2, columns=['fpath', 'disease', 'node_lbl', 'y_lbl', 'tgroup']) 
        df["y_lbl"] = pd.to_numeric(df["y_lbl"])
        return df
    else: ## silly hack!! clean this up!!
        return [ x[:-1] for x in _d1+_d2]


def plot_sample_dist(df):
    plt.figure(figsize=(9,3))
    g = sns.countplot(x='node_lbl', hue='y_lbl', data=df.sort_values(['y_lbl'],ascending=False))#, ax=ax);
    # g = sns.catplot(x='node_lbl', hue='y_lbl', col='tgroup', data=df.sort_values(['y_lbl'],ascending=False), 
    #                 kind="count", height=4, aspect=1.1);
    plt.title('sample distribution - by training group and y_lbl')
    plt.show();

