import requests


def generate_answer(prompt):
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "mistral",
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.05,
                    "top_p": 0.9,
                    "num_predict": 25,
                },
            },
            timeout=300,
        )

        if response.status_code != 200:
            return f"Ollama error {response.status_code}: {response.text}"

        return response.json()["response"].strip()

    except requests.exceptions.RequestException as e:
        return f"Ollama connection error: {e}"