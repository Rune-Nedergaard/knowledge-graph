import argparse
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.models.bert_embed import BertEmbed

from src.models.semantic_search_function import get_similar_paragraphs
from src.models.bert_rerank import BertRerank
from src.models.reranker_function import rerank_paragraphs

def main(question):

    # Load the fine-tuned re-ranker model
    fine_tuned_model_path = 'models/fine_tuned_model/check7_model.pth'
    # Load the fine-tuned re-ranker model
    reranker = BertRerank(model_path=fine_tuned_model_path)

    # Perform semantic search to get the top 1000 paragraphs
    top_k_paragraphs = get_similar_paragraphs(question, k=100)

    # Rerank the paragraphs using the rerank_paragraphs function
    reranked_paragraphs = rerank_paragraphs(question, top_k_paragraphs, reranker)

    # Print the top 10 reranked paragraphs
    for idx, (paragraph, score) in enumerate(reranked_paragraphs[:10]):
        print(f"Paragraph {idx + 1}: {paragraph}\nRelevance Score: {score}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rerank top 1000 paragraphs for a given question.")
    parser.add_argument("question", type=str, help="Input question for semantic search and reranking")
    args = parser.parse_args()

    main(args.question)
