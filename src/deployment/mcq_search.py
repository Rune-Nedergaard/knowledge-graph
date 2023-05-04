import os
import sys
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.features.qa_search import find_question_answer_pairs, initialize_qa_search
from src.deployment.divide_mcq import divide_mcq
from src.deployment.process_subquestions import relevant_qa_pairs

def find_relevant_qa_pairs(mcq, df, sentence_model, question_embeddings, questions, k=10):
    subquestions = divide_mcq(mcq)
    relevant_indices = set()

    for subquestion in subquestions:
        subquestion_indices = relevant_qa_pairs(subquestion, df, sentence_model, question_embeddings, questions, k=k)
        relevant_indices.update(subquestion_indices)

    return relevant_indices

if __name__ == "__main__":
    mcq_list = [
        "Hvilken skatteprocent er højere i Danmark: indkomstskat eller selskabsskat?",
    ]

    df, sentence_model, question_embeddings, questions = initialize_qa_search()

    for mcq in mcq_list:
        print(f"MC Question: {mcq}\n")
        relevant_indices = find_relevant_qa_pairs(mcq, df, sentence_model, question_embeddings, questions)
        relevant_qa_pairs = [(df.loc[idx, 'Spørgsmål'], df.loc[idx, 'Svar']) for idx in relevant_indices]

        for i, (question, answer) in enumerate(relevant_qa_pairs):
            print(f"{i + 1}. {question}\nAnswer: {answer}\n")
