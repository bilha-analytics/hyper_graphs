'''
author: bg
goal:   1. wrap THU_HGNN models @ sklearn use for pipelines, gridsearch, 
        2. add graph visualization functionality 
type:  
how: 
ref: 
refactors: 
'''

import numpy as np 

from sklearn.base import BaseEstimator, ClassifierMixin
import sklearn.metrics as skmetrics 

import hyperg.generation as hgen 
import hyperg.learning as hlearn 

import networkx as nx
from networkx import NetworkXException 
import hypernetx as hnx
import matplotlib.pyplot as plt 
import matplotlib.patches as mpatches
import seaborn as sns 

from collections import defaultdict

import data_loading as dio 

#### ============ 1. CLASSES =====================
class HGModel(BaseEstimator):
    def __init__(self, gdata, 
                 gen_method, gen_argz,
                 learn_method, learn_argz ): 
        super().__init__()

        self.gdata = gdata ## gdata = content listing, TODO: = list(x_attr:(disease, node lbl))
        self.gen_method =  gen_method 
        self.gen_argz = gen_argz 
        self.learn_method =  learn_method 
        self.learn_argz = learn_argz 

        self.hg = None 
        self.learnt_nodes = {'X_idx': set(), 'y_pred': [], 'y_truth': []}  
        self.acc = 0 

    def fit(self, X, y=None, y_truth=None):
        return self
    
    def predict(self, X, y=None):
        pred = [] 
        return pred 

    def score(self, yhat=None, y=None) :
        return self.acc 
    
    def get_H(self):
        return self.hg._H if self.hg is not None else None 
    
    def plot_HG(self):
        '''
        Simply put, this is just a mess !!!!! 
        '''
        H = self.hg._H 
        nodes = list(range(H.shape[0]) ) # id
        #[tuple(np.where(H[j] > 0)[0] ) for j in range(H.shape[1])]
        hyper_edges = [tuple(np.where(H.toarray()[j] > 0)[0] ) for j in range(H.shape[1])] # col span?
        #print("\n>>>>> HE \n" ,hyper_edges )       
        #node_lblz = {n: f"{n}_{self.gdata[n][2]}" for n in nodes } #id_disease     
        
        print("====== 1. PAIRWISE EDGEZ =======")
        pair_edgez = [ e for e in H.todok().keys() if e[0] != e[1]] 
        HGX = hnx.Hypergraph(  {f"he-{i}":h for i,h in enumerate(pair_edgez) } )
        hnx.draw(HGX)
        plt.show();
        plt.clf()
        
        print("====== 2. HYPER EDGEZ =======")
        kwargs = {'layout_kwargs': {'seed': 39}}
        HGX = hnx.Hypergraph(  {f"he-{i}":h for i,h in enumerate(hyper_edges) if len(h) > 1} )
        hnx.draw(HGX.collapse_nodes(), **kwargs) #hnx.drawing.draw(H.collapse_nodes(), **kwargs)
        plt.show();
        plt.clf()
        
        print("====== 3. HYPER EDGEZ + LEARNT UPDATES =======")
        color_legend = ['normal', 'not normal', 'predicted', 'hyper_edge']
        colrz = ['#77f700', '#d62828', '#2828df', '#dddddd']
                
        edgez = defaultdict(list)
        for he in hyper_edges:
            l = len(he)
            if l > 2: ## only order 3++
                edgez[l].append( he )                 
        n_he = len( edgez)
        print("# HE len groups =  ", n_he)
        
        if n_he == 0:
            print("No Hyper Edges only edges")
            edgez[l].append( [ e for e in H.todok().keys() if e[0] != e[1]]  )                 
            n_he = len( edgez)

        
        nc = min(2, n_he) 
        nr = n_he//nc
        nr += 1 if n_he % nc != 0 else 0 
        fig, axs = plt.subplots(nr, nc, figsize=(8*nc, 8*nr) ) 
        #print(axs)
        axs = [axs,] if n_he == 1 else axs.reshape(-1)
        for ax, e in zip(axs, edgez):
            ax.axis('off')
            ax.set_title(f'hyperedge degree {e}')
        fig.patch.set_facecolor('#eeeeee')
        
        
        ## Do star expansion for each hyperedge 
        H = H.toarray()
        for i, nhe in enumerate(sorted(edgez)):
            hedgez = edgez[nhe]            
            tmp_n_lblz = {} 
            he_nodes = []
            
            g = nx.DiGraph()
            #g.add_nodes_from( nodes )                         
            ## 1. group known labels into one node  AND only show per record for predicted nodes             
            ok_nodes = set([n for n in range(H.shape[0]) if int(self.gdata[n][-1]) == 0]) 
            sick_nodes = set([n for n in range(H.shape[0]) if int(self.gdata[n][-1]) == 1]) 
            
            _node_ok = 'ok'
            _node_not_ok = 'ill'
            g.add_nodes_from( [_node_ok,_node_not_ok] + list(self.learnt_nodes['X_idx']) )
            
            node_lblz = {n: f"{n}_{self.gdata[n][2]}" for n in self.learnt_nodes['X_idx'] }             
            node_lblz[_node_ok] = _node_ok
            node_lblz[_node_not_ok] = _node_not_ok
            
            ## 2. hyper edge nodes 
            if nhe == 2:
                g.add_edges_from(hedgez)
            else:
                for j, he in enumerate(hedgez,1):
                    he_node = f"he_{j}" #tuple(he) #
                    g.add_node( he_node ) 
                    tmp_n_lblz[he_node] = f"{he_node}"
                    he_nodes.append(he_node)
                    for node in he:
                        #g.add_edge( node, he_node)        
                        if node in self.learnt_nodes['X_idx']:
                            g.add_edge( node, he_node)        
                        elif node in ok_nodes:
                            g.add_edge( _node_ok, he_node)      
                        else: #if node in sick_nodes:
                            g.add_edge( _node_not_ok, he_node)   
                        
            try:
                pos = nx.planar_layout(g)
            except NetworkXException:
                #pos = nx.spring_layout(g, k=0.15, iterations=20)      
                # hack @ star expansion bipart like display
                l = set( he_nodes)
                r = g.nodes - l
                pos = {}
                pos.update((node, (1, 6*index)) for index, node in enumerate(l))
                pos.update((node, (2, 3*index)) for index, node in enumerate(r))
            
            #he_nodes = set(g.nodes) - set(nodes) 
            #print(he_nodes)
            pred_ok = []
            pred_sick = []
            for q, s, t in zip(self.learnt_nodes['X_idx'], self.learnt_nodes['y_pred'], self.learnt_nodes['y_truth']):
                if int(s) == 0:
                    pred_ok.append( q)
                else:
                    pred_sick.append( q)
                    
            nx.draw_networkx_nodes(g, pos, node_color='#55a722', node_size=200, nodelist=set([_node_ok,]), ax=axs[i] )
            nx.draw_networkx_nodes(g, pos, node_color='#a62828', node_size=200, nodelist=set([_node_not_ok,]), ax=axs[i] )
            
            nx.draw_networkx_nodes(g, pos, node_color='#77d722', node_size=100, nodelist=pred_ok, ax=axs[i] )
            nx.draw_networkx_nodes(g, pos, node_color='#d62828', node_size=100, nodelist=pred_sick, ax=axs[i] )
            
            nx.draw_networkx_nodes(g, pos, node_color='#f77f00', node_size=400, nodelist=he_nodes, ax=axs[i])
        
            nx.draw_networkx_edges(g, pos, edge_color='#666666', connectionstyle='arc3,rad=0.05' , ax=axs[i])
                    
            nx.draw_networkx_labels(g, pos, {**node_lblz,**tmp_n_lblz} , ax=axs[i])
        
        
#         plt_legendz = []
#         for c, l in zip(colrz, color_legend):
#             plt_legendz.append( mpatches.Patch(color=c, label=l))
#         plt.legend(handles=plt_legendz)
        plt.show();   
     

class TransductiveHyperG(HGModel):
    def __init__(self, gdata, 
                 gen_method=hgen.gen_knn_hg,  
                 gen_argz={'n_neighbors' : 4, 'with_feature': False }, 
                 learn_method=hlearn.trans_infer, 
                 learn_argz={'lbd': 100, }):
        super().__init__(gdata, gen_method, gen_argz, learn_method, learn_argz)         
        self.pred = None 
    
    def fit(self, X, y=None, y_truth=None ):
        ## 1. generate graph      
        self.hg = self.gen_method(X, **self.gen_argz )

        ## 2. transductive learn @ H, predicted 
        self.pred = self.learn_method(self.hg, y, **self.learn_argz)
        ac = skmetrics.accuracy_score(y_truth, self.pred, normalize=True)
        if ac >= 0:
            self.acc = ac
        #print("Truth: ", self.y_test, "\nPred: ", self.pred, "\nAcc: ", self.acc) 
        
        ## for plotting predicted items
        self.learnt_nodes['X_idx'] = set(np.where(y == -1)[0])
        self.learnt_nodes['y_pred'] = self.pred
        self.learnt_nodes['y_truth'] = y_truth  
        
        return self 

    def predict(self, X=None, y=None): 
        return self.pred 


class InductiveHyperG(HGModel):
    def __init__(self, gdata, 
                 gen_method=hgen.gen_knn_hg,  
                 gen_argz={'n_neighbors' : 4, 'with_feature': False }, 
                 learn_method=hlearn.inductive_fit, 
                 learn_argz={'lbd': 100, 'mu': .5, 'eta': .5, 'max_iter': 10 }, 
                predict_method = hlearn.inductive_predict):
        super().__init__(gdata, gen_method, gen_argz, learn_method, learn_argz) 
        self.predict_method = predict_method 
        self.hg_paramz = None 
    
    def fit(self, X, y=None, y_truth=None):  
        if X is not None:
            X = np.array(X)
        if y is not None:
            y = np.array(y)

        ## 1. generate the hyper graph        
        if self.gen_method == hgen.gen_attribute_hg:
            #print("Inductive @ gen attribute based hg ")
            n_nodes = len( X )            
            ## TODO: param pass once per dtype 
            attr_dict = {}
            for d in dio._CLASSEZ: #_CLASSEZ =  ['Undef', 'Normal', 'DR', 'Other']
                attr_dict[d] = list( np.where(np.array([ i[-2] for i in self.gdata[:n_nodes]]) == d  )[0] )
            
            #print( n_nodes , attr_dict, len(attr_dict))
            self.hg = hgen.gen_attribute_hg(n_nodes, attr_dict, X=X)
        else:
            #print("Inductive @ gen other hg ", self.gen_method )
            self.hg = self.gen_method(X, **self.gen_argz )
        
        ## 2. inductivelearning  @ returns IMHL a data class for M and omega weights
        #print("Inductive @ learn method ", self.learn_method )
        self.hg_paramz = self.learn_method(self.hg, y, **self.learn_argz)
        return self 
        
    def predict(self, X, y=None): 
        assert self.hg_paramz is not None 
        if X is not None:
            X = np.array(X)
        if y is not None:
            y = np.array(y)

        ## predict using previously trained model weights 
        pred = self.predict_method(X, self.hg_paramz)  

        ## for plotting predicted items : increment X_idx TODO: what does the udpated H look like 
        self.learnt_nodes['X_idx'] = self.hg._H.shape[0] + np.array(list(range(len(X))))
        self.learnt_nodes['y_pred'] = pred
        self.learnt_nodes['y_truth'] = y 

        return pred 
    
    def score(self, yhat, y) :
        self.acc = skmetrics.accuracy_score(yhat, y, normalize=True) 
        return self.acc 


