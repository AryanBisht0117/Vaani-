"""Resolve a laptop's built-in microphone without using the system default."""
import sounddevice as sd

from core.config import BUILTIN_MICROPHONE_NAME


_BUILTIN_NAME_MARKERS = (
    "microphone array",
    "internal microphone",
    "internal mic",
    "built-in microphone",
    "built in microphone",
    "built-in mic",
    "built in mic",
    "laptop microphone",
    "laptop mic",
    "digital microphone",
    "digital mic",
    "dmic",
)
_EXTERNAL_NAME_MARKERS = (
    "airpod",
    "bluetooth",
    "bthhfenum",
    "earbud",
    "hands-free",
    "handsfree",
    "headphone",
    "headset",
    "stereo mix",
    "usb",
    "webcam",
    "wireless",
)
_HOST_API_PRIORITY = (
    "mme",
    "wasapi",
    "windows directsound",
    "core audio",
    "alsa",
    "pulseaudio",
    "pipewire",
    "wdm-ks",
)


def _is_external_name(name):
    normalized = name.casefold()
    return any(marker in normalized for marker in _EXTERNAL_NAME_MARKERS)


def _is_builtin_name(name):
    normalized = name.casefold()
    return not _is_external_name(normalized) and any(
        marker in normalized for marker in _BUILTIN_NAME_MARKERS
    )


def _host_api_rank(device, host_apis):
    name = host_apis[device["hostapi"]]["name"].casefold()
    for rank, preferred_name in enumerate(_HOST_API_PRIORITY):
        if preferred_name in name:
            return rank
    return len(_HOST_API_PRIORITY)


def _device_description(device, host_apis):
    host_api = host_apis[device["hostapi"]]["name"]
    return f"{device['name']!r} ({host_api})"


def _available_input_description(devices, host_apis):
    descriptions = [
        _device_description(device, host_apis)
        for device in devices
        if device["max_input_channels"] > 0
    ]
    return ", ".join(descriptions) if descriptions else "none"


def resolve_builtin_microphone(sample_rate):
    """Return the current PortAudio index for the built-in microphone.

    Device indices are resolved at startup and are never persisted, as Windows
    may renumber them when audio devices are connected or disconnected.
    """
    devices = list(sd.query_devices())
    host_apis = list(sd.query_hostapis())
    inputs = [device for device in devices if device["max_input_channels"] > 0]
    selector = BUILTIN_MICROPHONE_NAME.casefold()

    if selector:
        matches = [
            device for device in inputs
            if device["name"].casefold() == selector and not _is_external_name(device["name"])
        ]
        if not matches:
            raise RuntimeError(
                f"Built-in microphone {BUILTIN_MICROPHONE_NAME!r} was not found. "
                "Set VAANI_BUILTIN_MICROPHONE to an exact built-in input device "
                f"name. Available inputs: {_available_input_description(devices, host_apis)}"
            )
    else:
        matches = [device for device in inputs if _is_builtin_name(device["name"])]
        if not matches:
            raise RuntimeError(
                "Could not identify the laptop's built-in microphone. "
                "Set VAANI_BUILTIN_MICROPHONE to its exact input device name. "
                f"Available inputs: {_available_input_description(devices, host_apis)}"
            )

    compatible_matches = []
    for device in matches:
        try:
            sd.check_input_settings(
                device=device["index"],
                channels=1,
                samplerate=sample_rate,
                dtype="float32",
            )
        except sd.PortAudioError:
            continue
        compatible_matches.append(device)
    if not compatible_matches:
        raise RuntimeError(
            f"The selected built-in microphone is unavailable or does not support "
            f"Vaani's required {sample_rate} Hz mono audio format. "
            f"Available inputs: {_available_input_description(devices, host_apis)}"
        )
    matches = compatible_matches

    best_rank = min(_host_api_rank(device, host_apis) for device in matches)
    preferred_matches = [
        device for device in matches
        if _host_api_rank(device, host_apis) == best_rank
    ]
    if len(preferred_matches) != 1:
        choices = ", ".join(
            _device_description(device, host_apis) for device in preferred_matches
        )
        raise RuntimeError(
            "More than one built-in microphone matches the selection. "
            "Set VAANI_BUILTIN_MICROPHONE to the exact built-in device name "
            f"to disambiguate. Matches: {choices}"
        )

    return int(preferred_matches[0]["index"])
