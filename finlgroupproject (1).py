#!/usr/bin/env python
# coding: utf-8

# In[5]:


#Import libraries and set seed
import re
import math
import random
import numpy as np
import pandas as pd

from tqdm import tqdm
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from rank_bm25 import BM25Okapi

import torch
from torch.utils.data import DataLoader

from sentence_transformers import CrossEncoder
from sentence_transformers.readers import InputExample


# In[11]:


#Without a seed: 1. random sampling may change every run 2. train/test split may change 3.selected negative examples may change

#With a seed: we get the same random results each time

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

print("Libraries imported.")


# In[60]:


#Load MS MARCO

dataset = load_dataset("microsoft/ms_marco", "v1.1")
print(dataset)


# In[63]:


print(dataset["train"][0].keys())
print(dataset["train"][0]["query"])
print(dataset["train"][0]["passages"].keys())


# In[64]:


# Example of the dataset Data.
import pandas as pd

rows = []
count = 0

for sample in dataset["train"]:
    query = sample["query"]

    for p, l in zip(sample["passages"]["passage_text"],
                    sample["passages"]["is_selected"]):

        rows.append({
            "query": query,
            "passage": p,
            "label": l
        })

        count += 1
        if count == 10:
            break

    if count == 10:
        break

df = pd.DataFrame(rows)
df


# In[14]:


#Convert MS MARCO into positive query-passage pairs
#Each query has multiple passages. We will extract the passages where is_selected == 1 as positives.

def extract_positive_pairs(hf_split, max_queries=None):
    rows = []
    count = 0

    for example in tqdm(hf_split, desc="Extracting positives"):
        query_id = example["query_id"]
        query = example["query"]
        passages = example["passages"]["passage_text"]
        selected_flags = example["passages"]["is_selected"]

        for passage, selected in zip(passages, selected_flags):
            if selected == 1:
                rows.append({
                    "query_id": query_id,
                    "query": query,
                    "passage": passage,
                    "label": 1
                })

        count += 1
        if max_queries is not None and count >= max_queries:
            break

    return pd.DataFrame(rows)


# In[15]:


#For a laptop, start with a manageable subset first.
train_pos_df = extract_positive_pairs(dataset["train"], max_queries=10001)
val_pos_df   = extract_positive_pairs(dataset["validation"], max_queries=2001)

print(train_pos_df.shape)
print(val_pos_df.shape)
train_pos_df.head()


# In[16]:


#Keep one positive passage per query
#Some queries may have multiple selected passages. To keep the workflow simple and clean, use one positive per query.
train_pos_df = train_pos_df.drop_duplicates(subset=["query_id"]).reset_index(drop=True)
val_pos_df   = val_pos_df.drop_duplicates(subset=["query_id"]).reset_index(drop=True)

print("Train queries:", train_pos_df["query_id"].nunique())
print("Val queries:", val_pos_df["query_id"].nunique())


# In[17]:


#Create one combined dataset and split by query
#Even though MS MARCO already has train and validation splits, it is useful in coursework to show how to do query-level splitting clearly. 
#We will combine and then split by query ID.
#This is the correct way to split a ranking dataset for a reranker because it avoids query leakage across splits.

all_pos_df = pd.concat([train_pos_df, val_pos_df], ignore_index=True)
all_query_ids = all_pos_df["query_id"].unique()

train_qids, temp_qids = train_test_split(
    all_query_ids, test_size=0.30, random_state=SEED
)

val_qids, test_qids = train_test_split(
    temp_qids, test_size=0.50, random_state=SEED
)

train_df = all_pos_df[all_pos_df["query_id"].isin(train_qids)].reset_index(drop=True)
val_df   = all_pos_df[all_pos_df["query_id"].isin(val_qids)].reset_index(drop=True)
test_df  = all_pos_df[all_pos_df["query_id"].isin(test_qids)].reset_index(drop=True)

print("Train:", train_df.shape)
print("Validation:", val_df.shape)
print("Test:", test_df.shape)


# In[18]:


#Build the BM25 corpus
def extract_passage_corpus(hf_split, max_queries=None):
    rows = []
    count = 0

    for example in tqdm(hf_split, desc="Building passage corpus"):
        passages = example["passages"]["passage_text"]
        for p in passages:
            rows.append({"passage": p})

        count += 1
        if max_queries is not None and count >= max_queries:
            break

    corpus_df = pd.DataFrame(rows).drop_duplicates().reset_index(drop=True)
    corpus_df["passage_id"] = corpus_df.index
    return corpus_df


# In[19]:


corpus_df_train = extract_passage_corpus(dataset["train"], max_queries=10001)
corpus_df_val   = extract_passage_corpus(dataset["validation"], max_queries=2001)

corpus_df = pd.concat([corpus_df_train, corpus_df_val], ignore_index=True)
corpus_df = corpus_df.drop_duplicates(subset=["passage"]).reset_index(drop=True)
corpus_df["passage_id"] = corpus_df.index

print("Corpus size:", len(corpus_df))
corpus_df.head()


# In[20]:


#Preprocess text and build BM25

def simple_tokenize(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return text.split()


# In[21]:


corpus_texts = corpus_df["passage"].tolist()
tokenized_corpus = [simple_tokenize(doc) for doc in tqdm(corpus_texts, desc="Tokenizing corpus")]
bm25 = BM25Okapi(tokenized_corpus)

print("BM25 index built.")


# In[22]:


#Create helper mappings:
passage_to_id = dict(zip(corpus_df["passage"], corpus_df["passage_id"]))
id_to_passage = dict(zip(corpus_df["passage_id"], corpus_df["passage"]))
corpus_ids = corpus_df["passage_id"].tolist()


# In[23]:


#BM25 retrieval function
#This code is showing us how BM25 retrieves results for one query

def bm25_retrieve(query, top_k=10):
    tokenized_query = simple_tokenize(query)
    scores = bm25.get_scores(tokenized_query)
    top_idx = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_idx:
        results.append({
            "passage_id": corpus_ids[idx],
            "passage": corpus_texts[idx],
            "bm25_score": float(scores[idx])
        })
    return results


# In[25]:


#to see the output of the above function
sample_query = train_df.iloc[0]["query"]
results = bm25_retrieve(sample_query, top_k=10)

results_df = pd.DataFrame(results)
results_df


# In[44]:


#Test BM25 with a query

query = "what is rba"

results = bm25_retrieve(query, top_k=10)

import pandas as pd

bm25_df = pd.DataFrame(results)

# Add rank column
bm25_df["rank"] = range(1, len(bm25_df) + 1)

# Reorder columns
bm25_df = bm25_df[["rank", "passage_id", "bm25_score", "passage"]]

bm25_df.head()


# In[26]:


#Create BM25 hard negatives
#he important part: for each query, we will use the known positive passage and then retrieve top BM25 candidates.
#Any top BM25 passage that is not the positive passage becomes a hard negative.

def build_training_pairs(pos_df, top_k=50, negatives_per_query=2)#---???????????????????????????????? befor per query=4 
    rows = []

    for _, row in tqdm(pos_df.iterrows(), total=len(pos_df), desc="Building training pairs"):
        query_id = row["query_id"]
        query = row["query"]
        positive_passage = row["passage"]

# Add positive pair
        rows.append({
            "query_id": query_id,
            "query": query,
            "passage": positive_passage,
            "label": 1
        })

 # Retrieve BM25 candidates
        candidates = bm25_retrieve(query, top_k=top_k)

        negatives_added = 0
        for cand in candidates:
            if cand["passage"] != positive_passage: #Avoid adding the same positive passage again
                rows.append({
                    "query_id": query_id,
                    "query": query,
                    "passage": cand["passage"],
                    "label": 0
                })
                negatives_added += 1

            if negatives_added >= negatives_per_query:
                break

    return pd.DataFrame(rows)


# In[27]:


#This is exactly the “BM25 baseline hard negative” setup: BM25 provides the candidate list,
#and the non-positive high-ranking candidates become challenging negatives.

train_pairs_df = build_training_pairs(train_df, top_k=50, negatives_per_query=2)
val_pairs_df   = build_training_pairs(val_df, top_k=50, negatives_per_query=2)
test_pairs_df  = build_training_pairs(test_df, top_k=50, negatives_per_query=2)

print("Train pairs:", train_pairs_df.shape)
print("Val pairs:", val_pairs_df.shape)
print("Test pairs:", test_pairs_df.shape)
train_pairs_df.head(6)


# In[ ]:





# In[28]:


 #Prepare CrossEncoder training examples
#Sentence Transformers documents CrossEncoder as a model that takes a pair of texts and outputs a single score, which is the right setup for reranking. 
#For reranking, num_labels=1 is the standard choice.
train_samples = [
    InputExample(texts=[row["query"], row["passage"]], label=float(row["label"]))
    for _, row in train_pairs_df.iterrows()
]

val_samples = [
    InputExample(texts=[row["query"], row["passage"]], label=float(row["label"]))
    for _, row in val_pairs_df.iterrows()
]

print("Train samples:", len(train_samples))
print("Validation samples:", len(val_samples))


# In[30]:


#Load the CrossEncoder
#CrossEncoders are specifically meant for scoring query-passage pairs and are commonly used as second-stage rerankers.

rerank_model = "cross-encoder/ms-marco-MiniLM-L6-v2"
model = CrossEncoder(rerank_model, num_labels=1, max_length=512)

print("Loaded model:", rerank_model)


# In[31]:


#Create the training DataLoader(dataloader Load data → in batches → during training)

train_dataloader = DataLoader(
    train_samples,      #train_samples = list of training pairs
    shuffle=True,       #If not shuffled:  Model sees: all positives first → all negatives later.  With shuffle: Mixed data → better learning.
    batch_size=16       #Train on 16 samples at a time.
)

print("Train DataLoader ready.")


# In[34]:


#Train the CrossEncoder
model.fit(
    train_dataloader=train_dataloader,
    epochs=2,         # Number of times model sees the entire dataset
    warmup_steps=100,  #First 100 steps → small learning rate → slowly increases (Gradually increases learning rate at the start)
    output_path="C:/Users/Mahelet/Desktop/IR/our_reranker_model", #Saves the train
    show_progress_bar=True   #Shows training progress
)


# In[37]:


model.save("our_reranker_model")


# In[38]:


trained_model = CrossEncoder("our_reranker_model")
print("Saved model reloaded.")


# In[39]:


#Neural reranking function
#This function takes the BM25 results and reorders them with the CrossEncoder.

def neural_rerank(query, bm25_results, reranker_model):
    pairs = [[query, item["passage"]] for item in bm25_results]
    scores = reranker_model.predict(pairs)

    reranked = []
    for item, score in zip(bm25_results, scores):
        reranked.append({
            "passage_id": item["passage_id"],
            "passage": item["passage"],
            "bm25_score": item["bm25_score"],
            "neural_score": float(score)
        })

    reranked = sorted(reranked, key=lambda x: x["neural_score"], reverse=True)
    return reranked


# In[49]:


#Example to try the the retrivalof reranking function and BM25 results.

# 1. Get a sample query from raw 1
query = test_df.iloc[1]["query"]

# 2. Retrieve BM25 results
bm25_results = bm25_retrieve(query, top_k=10)

# 3. Apply neural reranking
reranked_results = neural_rerank(query, bm25_results, model)

# 4. Convert BM25 results to DataFrame
bm25_df = pd.DataFrame(bm25_results)
bm25_df["bm25_rank"] = range(1, len(bm25_df) + 1)

# 5. Convert reranked results to DataFrame
rerank_df = pd.DataFrame(reranked_results)
rerank_df["rerank"] = range(1, len(rerank_df) + 1)

# 6. Merge BM25 rank with reranked results
final_df = rerank_df.merge(
    bm25_df[["passage_id", "bm25_rank"]],
    on="passage_id",
    how="left"
)

# 7. Reorder columns for clarity
final_df = final_df[
    ["passage_id","rerank", "bm25_rank",  "bm25_score", "neural_score", "passage"]
]

# 8. (Optional) shorten passage text
final_df["passage"] = final_df["passage"].str[:250]

# 9. Show top 10 results
final_df.head(10)


# In[50]:


#Build test relevance dictionary, We need the relevant passage for each test query.

test_relevance = (
    test_df.groupby("query")["passage"]
    .apply(set)
    .to_dict()
)

test_queries = list(test_relevance.keys())
print("Number of test queries:", len(test_queries))


# In[51]:


#Passage text → convert → passage IDs for evaluation.
test_relevance_ids = {}

for query, relevant_passages in test_relevance.items():
    relevant_ids = set()
    for p in relevant_passages:
        if p in passage_to_id:
            relevant_ids.add(passage_to_id[p])
    test_relevance_ids[query] = relevant_ids


# In[52]:


#Define evaluation metrics.

def precision_at_k(ranked_ids, relevant_ids, k=10):
    ranked_ids = ranked_ids[:k]
    hits = sum(1 for pid in ranked_ids if pid in relevant_ids)
    return hits / k

def recall_at_k(ranked_ids, relevant_ids, k=10):
    ranked_ids = ranked_ids[:k]
    if len(relevant_ids) == 0:
        return 0.0
    hits = sum(1 for pid in ranked_ids if pid in relevant_ids)
    return hits / len(relevant_ids)

def mrr_at_k(ranked_ids, relevant_ids, k=10):
    ranked_ids = ranked_ids[:k]
    for i, pid in enumerate(ranked_ids, start=1):
        if pid in relevant_ids:
            return 1.0 / i
    return 0.0

def dcg_at_k(ranked_ids, relevant_ids, k=10):
    dcg = 0.0
    for i, pid in enumerate(ranked_ids[:k], start=1):
        rel = 1 if pid in relevant_ids else 0
        if rel > 0:
            dcg += rel / math.log2(i + 1)
    return dcg

def ndcg_at_k(ranked_ids, relevant_ids, k=10):
    dcg = dcg_at_k(ranked_ids, relevant_ids, k)
    ideal_hits = min(len(relevant_ids), k)
    idcg = sum(1 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


# In[54]:


#Evaluate BM25 baseline
def evaluate_bm25(test_queries, relevance_dict, top_k=10):
    precision_scores = []
    recall_scores = []
    mrr_scores = []
    ndcg_scores = []

    for query in tqdm(test_queries, desc="Evaluating BM25"):
        relevant_ids = relevance_dict[query]
        results = bm25_retrieve(query, top_k=top_k)
        ranked_ids = [r["passage_id"] for r in results]

        precision_scores.append(precision_at_k(ranked_ids, relevant_ids, k=top_k))
        recall_scores.append(recall_at_k(ranked_ids, relevant_ids, k=top_k))
        mrr_scores.append(mrr_at_k(ranked_ids, relevant_ids, k=top_k))
        ndcg_scores.append(ndcg_at_k(ranked_ids, relevant_ids, k=top_k))

    return {
        "Precision@10": np.mean(precision_scores),
        "Recall@10": np.mean(recall_scores),
        "MRR@10": np.mean(mrr_scores),
        "NDCG@10": np.mean(ndcg_scores)
    }

 # Evaluating BM25.
bm25_metrics = evaluate_bm25(test_queries, test_relevance_ids, top_k=10)
bm25_metrics


# In[55]:


#Evaluate BM25 + CrossEncoder reranker.

def evaluate_reranker(test_queries, relevance_dict, reranker_model, top_k=10, retrieve_k=50):
    precision_scores = []
    recall_scores = []
    mrr_scores = []
    ndcg_scores = []

    for query in tqdm(test_queries, desc="Evaluating reranker"):
        relevant_ids = relevance_dict[query]

        bm25_results = bm25_retrieve(query, top_k=retrieve_k)
        reranked_results = neural_rerank(query, bm25_results, reranker_model)
        ranked_ids = [r["passage_id"] for r in reranked_results[:top_k]]

        precision_scores.append(precision_at_k(ranked_ids, relevant_ids, k=top_k))
        recall_scores.append(recall_at_k(ranked_ids, relevant_ids, k=top_k))
        mrr_scores.append(mrr_at_k(ranked_ids, relevant_ids, k=top_k))
        ndcg_scores.append(ndcg_at_k(ranked_ids, relevant_ids, k=top_k))

    return {
        "Precision@10": np.mean(precision_scores),
        "Recall@10": np.mean(recall_scores),
        "MRR@10": np.mean(mrr_scores),
        "NDCG@10": np.mean(ndcg_scores)
    }


# In[56]:


# Evaluating the reranker
reranker_metrics = evaluate_reranker(
    test_queries,
    test_relevance_ids,
    reranker_model=model,
    top_k=10,
    retrieve_k=50
)

reranker_metrics


# In[65]:


#Comparing the two models.

comparison_df = pd.DataFrame(
    [bm25_metrics, reranker_metrics],
    index=["BM25", "BM25 + CrossEncoder"]
)

comparison_df


# In[ ]:




