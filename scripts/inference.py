"""Standalone inference script for Hugging Face Qwen 2.5 7B."""

import os
import sys

def main():
    # You can set your token via env var: export HF_TOKEN="hf_..."
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("LLM_API_KEY")
    if not hf_token:
        print("Error: Please set the HF_TOKEN environment variable.")
        sys.exit(1)

    try:
        from huggingface_hub import InferenceClient
    except ImportError:
        print("Error: huggingface_hub is not installed. Run `uv pip install huggingface_hub`")
        sys.exit(1)

    model_id = "Qwen/Qwen2.5-7B-Instruct"
    print(f"Initializing InferenceClient for {model_id}...")
    
    client = InferenceClient(model=model_id, token=hf_token)

    prompt = """
OBSERVATION:
The alien universe has 4 sensors: pressure, turbulence, thermal_radiation, magnetic_flux.
t=0: pressure=0.5, turbulence=0.2

Please provide a JSON DSL theory that explains this data.
"""

    messages = [
        {"role": "system", "content": "You are a scientific discovery agent. Output ONLY valid JSON containing the Theory DSL."},
        {"role": "user", "content": prompt}
    ]

    print("Querying model... (This may take a moment on Serverless API)")
    try:
        response = client.chat_completion(
            messages=messages,
            max_tokens=512,
            temperature=0.2
        )
        content = response.choices[0].message.content
        print("\n--- Model Output ---")
        print(content)
        print("--------------------")
    except Exception as e:
        print(f"\nInference failed: {e}")
        print("Note: The model might be loading into memory on HF servers, try again in 1 minute.")

if __name__ == "__main__":
    main()
