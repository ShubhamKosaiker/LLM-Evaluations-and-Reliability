import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))


def generate_answer(query, contexts):
    context_text = "\n\n".join(contexts)

    prompt = f"""
You are a medical information assistant.

Answer the question using ONLY the provided context.

If the context does not contain enough information, say:
"I don't have enough information in the provided medical sources."

Do not invent medical facts.

Context:
{context_text}

Question:
{query}
"""

    response = client.chat.completions.create(
        model="openai/gpt-oss-20b",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    test_context = [
        "The most common causes of chronic cough include upper airway cough syndrome, asthma, GERD, COPD, and ACE inhibitors."
    ]

    answer = generate_answer(
        "What are common causes of chronic cough?",
        test_context
    )

    print(answer)