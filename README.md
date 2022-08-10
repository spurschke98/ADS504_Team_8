Project Title:
Classifying Semantic Equivalency Across Patent Applications

Description:
U.S. patent applications require significant manual effort to reconcile similarities across patents and patent domains. Modern data science and natural language processing techniques provide an opportunity to automate this review process. The objective of this study was to explore semantic similarity classification through several experimental approaches, from simple to near-state-of-the-art, toward a hypothesis that patent phrase pairs may be semantically classified. The study performed secondary data analysis of approximately 36,000 sampled U.S. patent phrase pairs, including preliminary data mining and pattern review. The study progressed through three experimental approaches, from simple word counts to context classification and, ultimately, semantic similarity using transformers (neural networks, or BERT). The study’s final BERT approach demonstrated that semantic similarity classification (automation) is possible for this use case, rejecting a null hypothesis. This conclusion supports promising additional refinement and research.

Build Status, License
__version__ = '1.0'
__date__ = 'April 2022'
__license__ = 'MIT'

Tech/Frameworks Used:
Python
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter
import statistics
import matplotlib.pyplot as plt
import seaborn as sns

