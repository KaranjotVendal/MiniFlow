import time

from TTS.api import TTS
import sounddevice as sd

from src.utils import get_test_sample, get_device


if __name__ == "__main__":
    sample = get_test_sample()
    sample_text = sample.transcript
    # for lang in TTS().list_models():
    #     print(lang)
    #

    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(get_device())
    # for sp in tts.speakers:
    #     print(sp)

    start = time.time()
    # generate speech by cloning a voice using default settings
    waveform = tts.tts(
        text=sample_text,
        # speaker_wav="/path/to/target/speaker.wav",
        speaker="Gracie Wise",
        language="en",
    )
    latency = time.time() - start

    # Play the waveform directly (streams to speakers)
    print(f"Generated from text: {sample_text}")
    print(f"Latency: {latency:.3f}s")
    print("Playing audio now...")
    sd.play(waveform, tts.synthesizer.output_sample_rate)  # Usually 24000 Hz for XTTS
    sd.wait()  # Block until playback finishes
    print("Playback complete.")
