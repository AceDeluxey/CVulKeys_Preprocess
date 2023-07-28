# CVulKeys_Preprocess
The Preprocessing script of CVulKeys

***
find_root_line.py contains basic function of slice and find_root_line.
slice.py is used to slice the .dot file in a folder.
build_pickle.py is used to build the data pickle from all the dot files.
***

## 1. Run **slice.py** to cut the original dot file into subdot file

folder = "../NewVul_Model\data\\process_C_pdg" , edit it as your own folder
Attention: The first-level directories in the folder must be category names. For example, in the image below, the folder is named "process_C_pdg," and the first-level directories are the categories.

    process_C_pdg - No_Vul    ----

                  - Vul_CWE1  ----

                  - Vul_CWE2 ----

                   ...
              
                   - Vul_CWE6  -----


## 2. Run **build_pickle.py**
Edit folder_path too.

***
*keyword_list.txt is keyword list file* 

*Put doc2vec model in /model* 
