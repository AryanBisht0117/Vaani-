
import collections
import queue

import numpy as np
import sounddevice as sd
from silero_vad import load_silero_vad, VADIterator
from core.config import MICROPHONE_DEVICE


SAMPLE_RATE = 16000
CHUNK_SIZE = 512
PRE_SPEECH_SECONDS = 1.5
MAX_RECORDING_SECONDS = 10

def enqueue_latest(audio_queue, audio):
    """Keep current speech when processing falls behind."""
    try:
        audio_queue.put_nowait(audio)
    except queue.Full:
        try:
            audio_queue.get_nowait()
        except queue.Empty:
            pass
        else:
            audio_queue.task_done()
        audio_queue.put_nowait(audio)
        print("Processing queue full. Oldest waiting recording dropped.")


def listen_continuously(audio_queue, stop_event, on_ready=None):
    model = load_silero_vad()

    vad = VADIterator(
        model,
        sampling_rate=SAMPLE_RATE,
        threshold=0.5,
        min_silence_duration_ms=800,
    )

    pre_buffer = collections.deque(
        maxlen=int(PRE_SPEECH_SECONDS * SAMPLE_RATE / CHUNK_SIZE)
    )

    recorded_chunks = []
    speech_started = False

    def submit_recording():
        if not recorded_chunks:
            return

        audio = np.concatenate(recorded_chunks)
        enqueue_latest(audio_queue, audio)
        print(f"\nSpeech queued: {len(audio) / SAMPLE_RATE:.2f}s")

    print("Vaani microphone active. Listening continuously...")

    with sd.InputStream(
        device=MICROPHONE_DEVICE,
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        blocksize=CHUNK_SIZE,
    ) as stream:
        if on_ready is not None:
            on_ready()

        while not stop_event.is_set():
            chunk, overflowed = stream.read(CHUNK_SIZE)

            if overflowed:
                print("Warning: microphone input overflow.")

            mono = chunk[:, 0].copy()
            event = vad(mono)

            if not speech_started:
                pre_buffer.append(mono)

                if event and "start" in event:
                    speech_started = True
                    recorded_chunks = list(pre_buffer)
                    print("\nSpeech detected.")

            else:
                recorded_chunks.append(mono)

                duration = (
                    len(recorded_chunks) * CHUNK_SIZE / SAMPLE_RATE
                )

                speech_ended = event and "end" in event
                maximum_reached = duration >= MAX_RECORDING_SECONDS

                if speech_ended or maximum_reached:
                    submit_recording()

                    if speech_ended:
                        recorded_chunks = []
                        speech_started = False
                        pre_buffer.clear()
                        vad.reset_states()
                    else:
                        # Preserve VAD state and context across long speech.
                        recorded_chunks = recorded_chunks[-4:]

    print("Microphone stopped.")
