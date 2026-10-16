import unittest
from unittest.mock import patch

import sounddevice as sd
from core.audio.microphone_selection import resolve_builtin_microphone


class MicrophoneSelectionTests(unittest.TestCase):
    def setUp(self):
        self.host_apis = [
            {"name": "MME"},
            {"name": "Windows WASAPI"},
            {"name": "Windows WDM-KS"},
        ]

    def query_devices(self, devices, selector=""):
        with patch(
            "core.audio.microphone_selection.sd.query_devices", return_value=devices
        ), patch(
            "core.audio.microphone_selection.sd.query_hostapis",
            return_value=self.host_apis,
        ), patch(
            "core.audio.microphone_selection.sd.check_input_settings",
        ), patch(
            "core.audio.microphone_selection.BUILTIN_MICROPHONE_NAME", selector
        ):
            return resolve_builtin_microphone(16000)

    def test_skips_internal_device_api_without_required_format(self):
        devices = [
            {
                "name": "Microphone Array (Realtek Audio)",
                "index": 8,
                "hostapi": 1,
                "max_input_channels": 2,
            },
            {
                "name": "Microphone Array (Realtek Audio)",
                "index": 2,
                "hostapi": 0,
                "max_input_channels": 2,
            },
        ]

        checked_formats = []

        def check_input_settings(device, channels, samplerate, dtype):
            checked_formats.append((device, channels, samplerate, dtype))
            if device == 8:
                raise sd.PortAudioError("Invalid sample rate", -9997)

        with patch(
            "core.audio.microphone_selection.sd.query_devices", return_value=devices
        ), patch(
            "core.audio.microphone_selection.sd.query_hostapis",
            return_value=self.host_apis,
        ), patch(
            "core.audio.microphone_selection.sd.check_input_settings",
            side_effect=check_input_settings,
        ), patch(
            "core.audio.microphone_selection.BUILTIN_MICROPHONE_NAME", ""
        ):
            self.assertEqual(resolve_builtin_microphone(16000), 2)
        self.assertEqual(
            checked_formats,
            [(8, 1, 16000, "float32"), (2, 1, 16000, "float32")],
        )

    def test_selects_named_internal_microphone_not_bluetooth_default(self):
        devices = [
            {
                "name": "Headset (Bluetooth Hands-Free)",
                "index": 4,
                "hostapi": 2,
                "max_input_channels": 1,
            },
            {
                "name": "Microphone Array (Realtek Audio)",
                "index": 17,
                "hostapi": 2,
                "max_input_channels": 2,
            },
            {
                "name": "Microphone Array (Realtek Audio)",
                "index": 8,
                "hostapi": 1,
                "max_input_channels": 2,
            },
            {
                "name": "Microphone Array (Realtek Audio)",
                "index": 2,
                "hostapi": 0,
                "max_input_channels": 2,
            },
        ]
        self.assertEqual(self.query_devices(devices), 8)

    def test_external_microphone_changes_do_not_change_device_identity(self):
        without_bluetooth = [
            {
                "name": "Microphone Array (Realtek Audio)",
                "index": 8,
                "hostapi": 1,
                "max_input_channels": 2,
            }
        ]
        with_bluetooth = [
            {
                "name": "Headset (Bluetooth Hands-Free)",
                "index": 4,
                "hostapi": 1,
                "max_input_channels": 1,
            },
            {
                "name": "Microphone Array (Realtek Audio)",
                "index": 11,
                "hostapi": 1,
                "max_input_channels": 2,
            },
        ]
        first = self.query_devices(without_bluetooth)
        second = self.query_devices(with_bluetooth)
        self.assertNotEqual(first, second)
        first_name = next(device["name"] for device in without_bluetooth if device["index"] == first)
        second_name = next(device["name"] for device in with_bluetooth if device["index"] == second)
        self.assertEqual(first_name, second_name)

    def test_exact_selector_resolves_one_of_multiple_internal_mics(self):
        devices = [
            {
                "name": "Microphone Array (Realtek Audio)",
                "index": 5,
                "hostapi": 1,
                "max_input_channels": 2,
            },
            {
                "name": "Internal Microphone (Intel Audio)",
                "index": 7,
                "hostapi": 1,
                "max_input_channels": 1,
            },
        ]
        self.assertEqual(
            self.query_devices(devices, "Internal Microphone (Intel Audio)"), 7
        )

    def test_fails_when_no_builtin_microphone_is_available(self):
        devices = [
            {
                "name": "Headset (Bluetooth Hands-Free)",
                "index": 4,
                "hostapi": 1,
                "max_input_channels": 1,
            }
        ]
        with self.assertRaisesRegex(RuntimeError, "Could not identify"):
            self.query_devices(devices)

    def test_fails_when_builtin_selection_is_ambiguous(self):
        devices = [
            {
                "name": "Microphone Array A",
                "index": 5,
                "hostapi": 1,
                "max_input_channels": 2,
            },
            {
                "name": "Microphone Array B",
                "index": 7,
                "hostapi": 1,
                "max_input_channels": 1,
            },
        ]
        with self.assertRaisesRegex(RuntimeError, "VAANI_BUILTIN_MICROPHONE"):
            self.query_devices(devices)


if __name__ == "__main__":
    unittest.main()
