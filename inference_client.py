from transformers import AutoTokenizer
import requests

MODEL_REPO = "pshashid/llama3.1B_8B_SQL_Finetuned_model"

tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO)

def build_prompt(schema, query):
    messages = [
        {"role": "system", "content": f"You are an expert SQL engineer. Return only valid SQL. Schema context: {schema}"},
        {"role": "user", "content": query},
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    return prompt


def main():

    prompt = build_prompt(
        "Table: orders(order_id, customer_id, amount, created_at)",
        "Get total sales per customer for the last 30 days."
    )

    payload = {
        "model": "sql-genie",   # must match --served-model-name
        "prompt": prompt,
        "max_tokens": 200,
        "temperature": 0.0
    }

    pod_url = "http://armed_ivory_swan-migration.runpod.io:8000/v1/completions"

    resp = requests.post(pod_url, json=payload)
    resp.raise_for_status()

    data = resp.json()

    generated_text = data["choices"][0]["text"]

    print("\nGenerated SQL:\n", generated_text)


if __name__ == "__main__":
    main()