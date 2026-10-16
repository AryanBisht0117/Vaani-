def main():
    from llama_cpp import Llama
    import time

    MODEL_PATH = "models/Qwen3-4B-Q4_K_M.gguf"

    print("Loading model...")

    llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=1024,
        n_threads=8,
        verbose=False
    )

    print("Model loaded.")

    messages = [
        {
            "role": "system",
            "content": (
                "You are a classification engine. "
                "Your ONLY valid outputs are ALERT, SOFT_ALERT, or IGNORE. "
                "Never explain your answer."
            )
        },
        {
            "role": "user",
            "content": (
                'Speech: "Aryan, can you come here?"\n'
                "Loudness: high\n"
                "Speaking rate: fast`n/no_think\n"
                "Repetition count: 1"
            )
        }
    ]

    print("\nGenerating...")

    start = time.perf_counter()

    output = llm.create_chat_completion(
        messages=messages,
        max_tokens=5,
        temperature=0
    )

    elapsed = time.perf_counter() - start

    result = output["choices"][0]["message"]["content"].strip()

    print("\nAI result:")
    print(result)

    print(f"\nTime: {elapsed:.2f} seconds")
    print(f"Completion tokens: {output['usage']['completion_tokens']}")

    if elapsed > 0:
        print(
            f"Speed: "
            f"{output['usage']['completion_tokens'] / elapsed:.2f} tokens/sec"
        )


if __name__ == "__main__":
    main()
