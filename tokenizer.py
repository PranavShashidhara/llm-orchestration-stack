from transformers import AutoTokenizer
import requests

MODEL_REPO = "pshashid/llama3.1B_8B_SQL_Finetuned_model"

# ── Tokenizer is local
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
    
    tokens = tokenizer(prompt, return_tensors="pt")
    return tokens["input_ids"][0].tolist()


def main():
    # ── Step 1: Build prompt & tokenize
    input_ids = build_prompt(
        "Table: orders(order_id, customer_id, amount, created_at)",
        "Get total sales per customer for the last 30 days."
    )

    # ── Step 2: Prepare payload
    payload = {
        "served_model_name": "sql-genie",   # important if your pod serves multiple models
        "input_ids": [input_ids],
        "max_new_tokens": 200
    }
    print("Payload prepared with input token IDs:", payload["input_ids"][0][:10], "...")
    # ── Step 3: Send request to pod
    pod_url = "http://armed_ivory_swan-migration.runpod.io:8000/v1/completions"  # replace with your pod's actual endpoint
    resp = requests.post(pod_url, json=payload)
    resp.raise_for_status()  # crash on HTTP errors
    data = resp.json()

    # ── Step 4: Decode output tokens
    output_ids = data["output_ids"][0]  # shape: [max_new_tokens]
    generated_text = tokenizer.decode(output_ids, skip_special_tokens=True)

    print("Generated SQL:\n", generated_text)
    

if __name__ == "__main__":
    main()