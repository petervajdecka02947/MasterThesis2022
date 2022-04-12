# Automatic Summarization
Final Master thesis 

- we first scrape politifact data from [**politifact.com**](https://www.politifact.com/), script saved in [**here**](https://github.com/petervajdecka02947/MasterThesis2022/blob/main/P0.politifact_scraping.ipynb) with its functions [**here**](https://github.com/petervajdecka02947/MasterThesis2022/blob/main/utils/scraping.py)
- preprocessing of data is saved in [**here**](https://github.com/petervajdecka02947/MasterThesis2022/blob/main/P1.data_preprocess.ipynb) with its functions [**here**](https://github.com/petervajdecka02947/MasterThesis2022/blob/main/utils/preprocess.py)
- Bert was fine-tunned in three different ways, the script of the best results are saved in [**here**](https://github.com/petervajdecka02947/MasterThesis2022/blob/main/P2B.Bert_fine-tunning.ipynb) with its functions [**here**](https://github.com/petervajdecka02947/MasterThesis2022/blob/main/utils/bert.py)
- tf-idf and doc2vec vectorizations were applied too, script is directed [**here**](https://github.com/petervajdecka02947/MasterThesis2022/blob/main/P3B.tf-idf_doc2vec.ipynb) with its functions [**here**](https://github.com/petervajdecka02947/MasterThesis2022/blob/main/utils/tf_idf_doc2vec.py)
