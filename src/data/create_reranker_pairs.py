"""
This script takes a rephrased question, then it does the following:
1. It performs semantic serach using the question and the embedded paragraph corpus
2. It then takes the top 5 paragraphs and creates a pair of (question, paragraph), making sure that at most 2 paragraphs are from the file belonging to the question
3 It then takes a random paragraph from the corpus and creates a pair of (question, paragraph)
4. It does this for 2000 questions
5. It then saves the pairs to a file
"""

import os
import pickle
from tqdm import tqdm


data_folder = 'data'


