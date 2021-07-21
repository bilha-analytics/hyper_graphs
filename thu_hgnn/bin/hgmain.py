'''
author: bg
goal: 
type: runner/app
how: 
ref: 
refactors: 
'''


from tqdm import tqdm 

import os 

import numpy as np
import pandas as pd 
from sklearn.pipeline import Pipeline


import matplotlib.pyplot as plt 
import seaborn as sns 


import hyperg.generation as hgen 
import hyperg.learning as hlearn 

import data_loading as dio 
import hgmodel as hgm 


##======= 1. Fmap Channels ====================

# a. order = original RGB, red , blue, yellow, darks, green, lbp
_Origi, _R, _Y, _B, _D, _G, _L = (0,1,2), 3, 4, 5, 6, 7, 8 

# c. build data permutation options  <--- defining channelz tuples b/c n_channel inputs to CNN
_channelz_ls = [
    ('origi', _Origi, ),
    ('green', (_G,) ), 
    ('darks', (_D,) ) ,
    ('lbp', (_L,) ) ,
    ('yellow', (_Y,) ) ,
    # ('lbp_green', (_L,_G) ) , 
    ('lbp_yellow', (_L,_Y) ) ,
    # ('lbp_darks', (_L,_D) ) ,
    # ('yellow_darks', (_Y,_D) ) ,
    # ('lbp_yellow_green',  (_L, _Y, _G) ) ,
    ('lbp_yellow_darks', (_L, _Y, _D) ) ,
    # ('lbp_red_yellow', (_L, _R, _Y) ) ,
    ('red_yellow_darks', (_R, _Y, _D) ) ,
    # ('red_yellow_green', (_R, _Y, _G) ) ,
    ('red_yellow_blue', (_R, _Y, _B) ) ,
    # ('yellow_darks_blue', (_Y, _D, _B) ) ,
    # ('lbp_red_yellow_green', (_L,_R,_Y,_G) ) ,
    # ('lbp_red_yellow_darks', (_L,_R,_Y,_D) ) ,
    # ('lbp_red_yellow_blue', (_L,_R,_Y,_B) ) ,
    # ('lbp_red_yellow_darks_green', (_L,_R,_Y,_D, _G) ) ,
    # ('lbp_red_yellow_darks_blue', (_L,_R,_Y,_D, _B) ) ,
    ('lbp_red_yellow_darks_blue_green', (_L,_R,_Y,_D, _B, _G) ),
]

#### ============ 2. RUN =====================

_TEST_PERCENT = .3  
_hg_learning_permutationz = [
        #. transductive kNN l2-relational single-g
        ('trans_kNN', True,{
            'gen_method': hgen.gen_knn_hg,  
            'gen_argz' : {'n_neighbors' : 4, 'with_feature': False }, 
            'learn_method' : hlearn.trans_infer, 
            'learn_argz' : {'lbd': 100, } }),
    
        #. transductive kmeans clustering single-g
        ('trans_kmeans', True,{
            'gen_method': hgen.gen_clustering_hg,  
            'gen_argz' : {'n_clusters' : 4, 'method':"kmeans", 'with_feature': False, 'random_state':987 }, 
            'learn_method' : hlearn.trans_infer, 
            'learn_argz' : {'lbd': 100, } }),
         
        #. transductive L1-relational single-g
        ('trans_L1', True,{
            'gen_method': hgen.gen_norm_hg,  
            'gen_argz' : {'gamma':.5, 'n_neighbors' : 4, 'norm_type':1, 'with_feature': False, }, 
            'learn_method' : hlearn.trans_infer, 
            'learn_argz' : {'lbd': 100, } }),
         
        #. transductive L2-relational single-g
        ('trans_L2', True,{
            'gen_method': hgen.gen_norm_hg,  
            'gen_argz' : {'gamma':.5, 'n_neighbors' : 4, 'norm_type':2, 'with_feature': False, }, 
            'learn_method' : hlearn.trans_infer, 
            'learn_argz' : {'lbd': 100, } }),
         
    
        #. transductive e-ball single-g
        ('trans_eball', True,{
            'gen_method': hgen.gen_epsilon_ball_hg,  
            'gen_argz' : {'ratio':.5, 'with_feature': False }, 
            'learn_method' : hlearn.trans_infer, 
            'learn_argz' : {'lbd': 100, } }),
                  
        #. inductive  kNN l2-relation-based single-g
        ('induct_kNN', False,{
            'gen_method': hgen.gen_knn_hg,  
            'gen_argz' : {'n_neighbors' : 4, 'with_feature': True }, 
            'learn_method' : hlearn.inductive_fit, 
            'learn_argz' : {'lbd': 100, 'mu': .5, 'eta': .5, 'max_iter': 10 , 'log':False} }),
      
        #. inductive  L1-relation-based single-g
        ('induct_L1', False,{
            'gen_method': hgen.gen_norm_hg,  
            'gen_argz' : {'gamma':.5, 'n_neighbors' : 4, 'norm_type':1, 'with_feature': True }, 
            'learn_method' : hlearn.inductive_fit, 
            'learn_argz' : {'lbd': 100, 'mu': .5, 'eta': .5, 'max_iter': 10 , 'log':False} }),
        
        #. inductive  L2-relation-based single-g
        ('induct_L2', False,{
            'gen_method': hgen.gen_norm_hg,  
            'gen_argz' : {'gamma':.5, 'n_neighbors' : 4,  'norm_type':2, 'with_feature': True }, 
            'learn_method' : hlearn.inductive_fit, 
            'learn_argz' : {'lbd': 100, 'mu': .5, 'eta': .5, 'max_iter': 10 , 'log':False} }),
    
        #. inductive  attribute-based single-g
        ('induct_attribute', False,{
            'gen_method': hgen.gen_attribute_hg,  
            'gen_argz' : { }, 
            'learn_method' : hlearn.inductive_fit, 
            'learn_argz' : {'lbd': 100, 'mu': .5, 'eta': .5, 'max_iter': 10  , 'log':False} }),
        ]
_ACCURACY_TEXT_FILE = "../data/100_runs_results__{}.txt".format
_FMAP_SIZE = (32,32)

def run_hg_learning_in_dset(fmap_channelz_ls, is_transductive, hg_kwargz, n_data=10, plot_log=True):
    '''
    an iteration for a give data pipeline ; can do multiple permutationz of channelz and feature selection 
    '''
    # each of train and test is a tuple of (X_fmap_paths, y_lbls, X_node_attributes)
    x_fmap, y_lbl, x_attr = 0, 1, 2
    train, test = dio.fetch_test_train(10, mode_transductive=is_transductive, rseed=2932) 
    if plot_log: ### aaaarrrgggghhhhhh
        dio.plot_sample_dist(dio.gen_pdframe(train, test) ) 

    ## TODO: Fix this @ HGModel 
    data = dio.gen_pdframe(train, test, do_list_only=True) 
    hg = hgm.TransductiveHyperG(data, **hg_kwargz) if is_transductive else hgm.InductiveHyperG(data, **hg_kwargz)
    
    results = {} 
    for p, channz in enumerate(fmap_channelz_ls,1):
        pname = channz[0] 
        if plot_log:
            print(f"========== @{p}. Running Pipeline <<{pname}>> ===========")

        if is_transductive:
            _ = hg.fit( dio.fetch_fmap_images(train[x_fmap], channz_ls=channz,resize_tuple=_FMAP_SIZE ), train[y_lbl], test[y_lbl] ) 
            yhat = hg.predict()
            ac = hg.score() ##TODO DRY
        else:  
            _ = hg.fit( dio.fetch_fmap_images(train[x_fmap], channz_ls=channz,resize_tuple=_FMAP_SIZE ), train[y_lbl] ) 
            yhat = hg.predict( dio.fetch_fmap_images(test[x_fmap], channz_ls=channz ,resize_tuple=_FMAP_SIZE), test[y_lbl] )    
            ac = hg.score(test[y_lbl], yhat)
        
        if plot_log:
            print(f"Prediction Accuracy: {ac}")
            hg.plot_HG()
            print(f"\n========== END of Pipeline <<{pname}>>===========\n\n")
        
        results[pname] = ac 
        
    return results

def run_e_iterations(RUNID, n_data, fmap_channelz_ls, permz=_hg_learning_permutationz, runz=3, plot_log=True):
    '''
    run several iterations for each training method 
    @input
        - RUNID     a tracker @ saving checkpoints 
        - permz:    a dict of hg training permutations @ mix of transductive Vs inductive AND different hg generation methods 
        - e:        number of iterations/epochs to learn --> paranoia @ concistency 
    @output
        - dframe    pdframe of training accuracy results per permutation 
    @actions:   autosave check points b/c training can take ages in some cases 
    '''
    acc_resultz = []
    for e in tqdm( range(runz) ):
        for tname, is_trans, hg_kwargz in permz:   
            plt.clf(); 
            rez = run_hg_learning_in_dset(fmap_channelz_ls, is_trans, hg_kwargz, n_data=n_data, plot_log=plot_log)

            for p, ac in rez.items(): # save 'checkpoint' and append listing
                r = [ tname.split('_')[0], tname, p, ac, e] 
                if not plot_log: ### aaaarrrgggghhhhhh
                    with open(_ACCURACY_TEXT_FILE(RUNID), 'a') as fd:
                        f = "\t".join([str(i) for i in r]) 
                        _ = fd.write(f"{f}\n")
                acc_resultz.append( r )

    # gen pdframe for further analysis             
    df = pd.DataFrame(acc_resultz, columns=['tm', 'tmethod', 'fmap', 'acc','e'])
    df = df.sort_values(['tm','acc'],ascending=False)
    
    return df 

def plot_accuracy_stats(acc_df):
    plt.figure(figsize=(17,10))
    sns.violinplot(x='tmethod', y='acc', data=acc_df, hue='tm',legend=False)
    sns.swarmplot(x='tmethod', y='acc', data=acc_df, hue='fmap')
    plt.legend(loc='upper right')
    plt.show();
    plt.clf()

    plt.figure(figsize=(10,6))
    sns.violinplot(x='tmethod', y='acc', data=acc_df, hue='tm')
    sns.swarmplot(x='tmethod', y='acc', data=acc_df, hue='tm')
    plt.show();
    plt.clf()

    plt.figure(figsize=(15,8))
    sns.swarmplot(x='tmethod', y='acc', data=acc_df, hue='tm')
    plt.show();


#### ============ 3. Here we go!! =====================
RUNID = '004' ## to fummed to tstamp this!!

def do_it_all(R=RUNID, n_data=10, fmap_channelz_ls=_channelz_ls, runz=3):    
    # 1. show sample network graphs 
    df1 = run_e_iterations(R, permz=[_hg_learning_permutationz[i] for i in (0,1, 5, -1)], 
            n_data=n_data, fmap_channelz_ls=[fmap_channelz_ls[i] for i in (1, -1)], 
            runz=1, plot_log=True)

    # 2. run everything for stats 
    acc_file = _ACCURACY_TEXT_FILE(R) 
    if os.path.isfile(acc_file): 
        print(f"loading old file {acc_file}")
        acc_resultz = [] 
        with open(acc_file, 'r') as fd:
            for r in fd.readlines(): 
                acc_resultz.append( r.strip().split("\t") ) 
        df2 = pd.DataFrame(acc_resultz, columns=['tm', 'tmethod', 'fmap', 'acc','e'])
        df2["acc"] = pd.to_numeric(df2["acc"])
        df2 = df2.sort_values(['tm','acc'],ascending=False)
    else:
        print(f"running new round {R}") 
        df2 = run_e_iterations(R, permz=_hg_learning_permutationz, 
            n_data=n_data, fmap_channelz_ls=fmap_channelz_ls, runz=runz, plot_log=False) 

    plot_accuracy_stats(df2) 

    return df2 
