import argparse
import sys
import os
import openai
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.models.bert_embed import BertEmbed
from api_secrets import API_KEY
from src.models.semantic_search_function import get_similar_paragraphs
from src.models.bert_rerank import BertRerank
from src.models.reranker_function import rerank_paragraphs

openai.api_key = API_KEY

"""
CONSIDER USING THE GET_CONTEXT FUNCTION FROM THE SEMANTIC SEARCH SCRIPT HERE
"""

def process_question_with_gpt4(question, list_of_paragraphs):
    formatted_paragraphs = "\n\n".join([f"Tekststykke {i+1}: {p}" for i, p in enumerate(list_of_paragraphs)])



    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "Du er en chatbot der så vidt muligt besvarer et bruger spørgsmål baseret på information i en række givne tekststykker. Baser alene dit svar på informationen i disse refer til dem i din argumentation. Giv et så fyldestgørende svar som informationen giver mulighed for. Såfremt du ikke har nok information til at besvare spørgsmålet, så skal du forklare hvorfor du ikke kan dette, og forklare hvilke yderligere oplysninger der ville være nødvendige for at kunne besvare spørgsmålet."},
            {"role": "user", "content": f"Bruger spørgsmål: {question}\n\n{formatted_paragraphs}"}
        ],
        temperature=0.5,
        max_tokens=200,
    )

    answer = response['choices'][0]['message']['content']
    return answer

def main(question):

    # Load the fine-tuned re-ranker model
    fine_tuned_model_path = 'models/fine_tuned_model/check_highbatch2_model.pth'
    # Load the fine-tuned re-ranker model
    reranker = BertRerank(model_path=fine_tuned_model_path)

    # Perform semantic search to get the top 1000 paragraphs
    top_k_paragraphs = get_similar_paragraphs(question, k=100)

    # Rerank the paragraphs using the rerank_paragraphs function
    reranked_paragraphs = rerank_paragraphs(question, top_k_paragraphs, reranker)
    """PROBABLY GET CONTEXT FOR THE PARAGRAPHS HERE"""

    final_paragraphs = [paragraph for paragraph, score in reranked_paragraphs[:8]]
    answer = process_question_with_gpt4(question, final_paragraphs)

    print(f"Spørgsmål: {question}\n")
    print(f"Svar: {answer}\n")

# Print the top 10 reranked paragraphs
    for idx, (paragraph, score) in enumerate(reranked_paragraphs[:10]):
        print(f"Paragraph {idx + 1}: {paragraph}\nRelevance Score: {score}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rerank top 1000 paragraphs for a given question.")
    parser.add_argument("question", type=str, help="Input question for semantic search and reranking")
    args = parser.parse_args()

    main(args.question)


