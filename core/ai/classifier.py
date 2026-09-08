import urllib.request
import json


API_URL = "http://localhost:8000/v1/chat/completions"
MODEL = "models/Qwen3-4B-Q4_K_M.gguf"


def classify_speech(transcript, loudness, speaking_rate, repetition_count):

    prompt = f"""
You classify environmental speech for a headphone user named Aryan.

IMPORTANT:
Determine whether the speaker is addressing Aryan.

Rules:
1. If the transcript is clearly a conversation between other people, return IGNORE.
2. If Aryan is explicitly addressed or someone is clearly trying to get Aryan's attention, return FULL_ALERT.
3. Use SOFT_ALERT only when there is some evidence that Aryan may need attention, but it is not clear.
4. Do not create an alert merely because speech is nearby.
5. Do not assume that every question is directed at Aryan.

Transcript: {transcript}
Loudness: {loudness}
Speaking rate: {speaking_rate} WPM
Repetition count: {repetition_count}

Return exactly one word:
IGNORE
SOFT_ALERT
FULL_ALERT
"""

    data = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": "You are a strict environmental speech classifier. Return only IGNORE, SOFT_ALERT, or FULL_ALERT."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_tokens": 5,
        "temperature": 0
    }

    request = urllib.request.Request(
        API_URL,
        data=json.dumps(data).encode(),
        headers={"Content-Type": "application/json"}
    )

    response = urllib.request.urlopen(request)
    result = json.loads(response.read().decode())

    return result["choices"][0]["message"]["content"].strip()