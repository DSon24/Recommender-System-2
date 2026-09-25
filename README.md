# Two-Tower Product Search Pre-Ranking System

A personal project that turns a search query into a short list of candidate products for a later ranking stage. The notebook trains separate query and product text encoders on the [Amazon Shopping Queries (ESCI) dataset](https://github.com/amazon-science/esci-data), then uses FAISS to retrieve products with similar embeddings.

## What the notebook does

1. Downloads the ESCI examples and product metadata, joins queries to product titles, and treats **Exact (E)** matches as positive pairs and **Substitute (S)**, **Complement (C)**, and **Irrelevant (I)** as negative pairs.
2. Tokenizes queries and titles separately. Each Keras tower uses a 64-dimensional word embedding, average pooling, a dense layer, and L2 normalization to produce a **16-dimensional vector**.
3. Trains the towers with cosine similarity mapped to a 0–1 match score for binary cross-entropy. The notebook uses the dataset's train/test designation for its training and validation split and saves the best weights by validation loss.
4. Encodes product titles and adds their normalized vectors to a **FAISS `IndexFlatIP`** index. At search time, the query tower embeds new text and FAISS returns the top-*k* products by inner product, equivalent to cosine similarity for normalized vectors.

The FAISS step narrows a large catalog to a candidate set. The notebook **does not implement the downstream ranker**.

## Current notebook run

The saved run in [`notebooks/two_tower_product_search.ipynb`](notebooks/two_tower_product_search.ipynb) reports **2,679,017 joined query–product rows** and **1,814,675 indexed product vectors**. It includes training history, a nearest-product inspection, and example text searches. The search examples are qualitative checks; the notebook does not yet report retrieval Recall@*k*, NDCG, or a comparison with a ranking baseline.

The notebook currently joins the examples and product table on `product_id` alone. The ESCI source demonstrates a join on both `product_locale` and `product_id`; the current row counts and evaluation should therefore be treated as exploratory until that join is corrected and the notebook is rerun. The tokenizer is also fitted on text from both the training and validation rows, so validation numbers should not be presented as a final generalization estimate.

## Run it

Open [`notebooks/two_tower_product_search.ipynb`](notebooks/two_tower_product_search.ipynb) in Google Colab and select **Runtime → Run all**. The notebook installs `faiss-cpu` and downloads the ESCI Parquet files from the dataset's GitHub repository. The product file is about 1.1 GB, so allow time and disk space for the download. A Google Drive mount cell is included but is not required for the model or FAISS search.

**Stack:** Python, TensorFlow/Keras, pandas, NumPy, FAISS.

**Data credit:** [Amazon Science, Shopping Queries Dataset: A Large-Scale ESCI Benchmark for Improving Product Search](https://github.com/amazon-science/esci-data). Data files and trained weights are not stored in this repository.
