import sys
import os
import openai
import re

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.features.qa_search import find_question_answer_pairs
from src.deployment.divide_mcq import divide_mcq
from api_secrets import API_KEY

openai.api_key = API_KEY


def process_question(question, qa_pairs):
    template = '''Input spørgsmål: {question}\n\n'''

    for i, pair in enumerate(qa_pairs):
        template += f"Par {i + 1}: {pair[0]} - {pair[1]}\n\n"

    return template.format(question=question)



def relevant_qa_pairs(MCQ_question, subquestion):
    results = find_question_answer_pairs(subquestion, k=30)
    
    formatted_qa_pairs = process_question(MCQ_question, results)

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": '''Du vurderer hvor relevante 30 forskellige spørgsmål-svar par er i forhold til et givent input spørgsmål. Rangér de 10 mest relevante par ved at angive indeksene (fra 1 til 30) for disse par. Svar således:

                            Indeks 1, Indeks 2, ..., Indeks 10'''},
            {"role": "user", "content": formatted_qa_pairs}
        ],
        temperature=0,
        max_tokens=100,
    )

    response_text = response['choices'][0]['message']['content']
    top_indices = [int(index) for index in re.findall(r'\b(\d+)\b', response_text)]

    return [results[i - 1] for i in top_indices]


if __name__ == "__main__":
    MCQ_question = "Hvad er et ofte fremført argument for administrativ centralisering (frem for decentralisering)? a) Centralization speeds decision making by reducing the overload of information which otherwise clogs the upper reaches of a decentralized hierarchy b) Centralization encourages innovation c) Centralization improves staff motivation and identification d) Centralization makes the line of accountability clearer and more easily understood by citizens"
    subquestions = divide_mcq(MCQ_question)
    top_10_relevant_pairs = relevant_qa_pairs(MCQ_question, subquestions[0])
    print(top_10_relevant_pairs)
