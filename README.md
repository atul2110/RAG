To run this code :-
## setup
Create new conda env :- `conda create -n url_rag python=3.10` <br/>
Entering the conda env :- `conda activate url_rag`<br/>
Run this command:- `pip install requests beautifulsoup4 networkx sentence-transformers pymilvus scikit-learn rank_bm25 numpy openai`

## Running
Change the base url in web_crawler.py in main function according to the url needed to be scrapped<br/>
In the terminal run :- `python web_crawler.py`<br/>
This generated scraped_data.json storing a dictionary with url and content<br/>
This process takes time according to level of scrapping currently it is 5.<br/>
<br/>
To run milvus, start docker<br/>
then run in terminal :- `docker compose up --build`<br/>
This will start the milvus in a docker container.<br/>
<br/>
Now in the another terminal run :- `python vector_and_chunking.py`<br/>
This will create a vector db and sores data in milvus<br/>
<br/>
Now in retrieval_reranking_and_qa.py in main function change query about what you need to ask and put your openai api key.<br/>
Now in the terminal run :- `python retrieval_reranking_and_qa.py`<br/>
The results are printed in the terminal.<br/>
