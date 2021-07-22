# (Hypergraph Learning: Methods and Practices Paper)[https://ieeexplore.ieee.org/document/9264674] Exploration

```
Gao, Yue, et al. “Hypergraph Learning: Methods and Practices.” IEEE Transactions on Pattern Analysis and Machine Intelligence, 2020, pp. 1–1. IEEE Xplore, doi:10.1109/TPAMI.2020.3039374.
```

Exploring this paper 


## Try it out
**1. Grab it**
You may grab the project folders

```   
    thu_gnn.zip 
    thu_gnn_extras.zip  
```

Use `requirements.txt` to setup your venv 

* Install THU-HGNN kit
Get it directly from their git page or use provided zipped folder `thu_gnn.zip`. Zipped folder has very minimal changes; nothing lost if working directly from the git folder  
	- Updated a couple of import calls to match new names due to version conflict. `sklearn` and `scipy` 
	- Updated graph generation method calls to allow for different pairwise_distances, where needed.  
	
PS: If reading in v7.3 `matlab` file formats you may need to ensure compatibility 

```   
    git clone https://github.com/iMoonLab/THU-HyperG.git
    cd THU-HyperG
    pip install .
```

* Intall other required packages
These are mainly for graphing 
```bash
pip install requirements.txt
```

**2. Use it**
* View notebook
You should have `jupyter notebook` installed. Unzip `thu_gnn_extras` and navigate to `notebooks` folder to view as is.

OR

* Run from terminal 
```bash
cd thu_gnn_extras\bin
python hgmain.py -i 'data_file.txt` 
```
`data_file.txt` is a listing of file paths and other meta on your input 


## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[MIT](https://choosealicense.com/licenses/mit/) 
