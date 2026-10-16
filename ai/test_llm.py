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

    prompt = "Reply with exactly one word: ALERT"

    print("\nGenerating...")

    start = time.perf_counter()

    output = llm(
        prompt,
        max_tokens=5,
        temperature=0
    )

    elapsed = time.perf_counter() - start

    text = output["choices"][0]["text"].strip()

    print("\nResponse:")
    print(text)

    print(f"\nTime: {elapsed:.2f} seconds")
    print(f"Prompt tokens: {output['usage']['prompt_tokens']}")
    print(f"Completion tokens: {output['usage']['completion_tokens']}")

    if elapsed > 0:
        print(
            f"Generation speed: "
            f"{output['usage']['completion_tokens'] / elapsed:.2f} tokens/sec"
        )


if __name__ == "__main__":
    main()
