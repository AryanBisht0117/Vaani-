def main():
    import urllib.request
    import json

    data = {
        "model": "models/Qwen3-4B-Q4_K_M.gguf",
        "messages": [
            {
                "role": "system",
                "content": (
                    "You classify environmental speech for a headphone user named Aryan. "
                    "Decide whether the speaker is directly addressing Aryan or trying to get "
                    "Aryan's attention. Do not assume every nearby conversation is directed at "
                    "Aryan. If clearly unrelated, return IGNORE. "
                    "If clearly directed at Aryan, return FULL_ALERT. "
                    "Reply with exactly one word: IGNORE, SOFT_ALERT, or FULL_ALERT."
                )
            },
            {
                "role": "user",
                "content": (
                    "Transcript: Excuse me, can you hear me? "
                    "I have been trying to get your attention. "
                    "Loudness: 0.07. Speaking rate: 140 WPM. Repetition count: 2."
                )
            }
        ],
        "max_tokens": 5,
        "temperature": 0
    }

    request = urllib.request.Request(
        "http://localhost:8000/v1/chat/completions",
        data=json.dumps(data).encode(),
        headers={"Content-Type": "application/json"}
    )

    response = urllib.request.urlopen(request)
    result = json.loads(response.read().decode())

    print(result["choices"][0]["message"]["content"])


if __name__ == "__main__":
    main()
