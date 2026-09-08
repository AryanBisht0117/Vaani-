transcript = "Hello, Aryan, Aryan, Aryan, Aryan, Aryan."

loudness = 0.0034

speaking_rate = 251.4

repetition_count = 5


features = {
    "transcript": transcript,
    "loudness": loudness,
    "speaking_rate": speaking_rate,
    "repetition_count": repetition_count
}


print("Extracted Features:")
print("-------------------")

for key, value in features.items():
    print(f"{key}: {value}")