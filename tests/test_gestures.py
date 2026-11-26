import os
import unittest
from unittest.mock import Mock, call, patch

import gesture_runner
import valera_gestures as gestures
import voice_assistant as voice


class GestureTests(unittest.TestCase):
    def test_handshake_is_an_offer(self):
        for phrase in ("Давайте пожмем руки!", "Пожмём руки?", "Пожму вам руку.",
                       "Давайте обменяемся рукопожатием.", "Можно пожать вам руку?"):
            with self.subTest(phrase=phrase):
                self.assertEqual(gestures.choose_gesture(phrase, first_phrase=False), "shake_hand")
        for phrase in ("Слово «рукопожатие» означает приветствие.", "Он сказал: «Пожмём руки?»",
                       "Мы не пожмём руки.", "Пожмём руки или просто помашем — решите позже."):
            self.assertIsNone(gestures.choose_gesture(phrase, first_phrase=True))

    def test_timed_to_second_sentence_and_not_repeated_on_fallback(self):
        for parallel in (True, False):
            bot = object.__new__(voice.VoiceAssistant)
            bot._tts_speaker = "test"
            bot._interrupt_tts = False
            bot._synthesize_pcm_bytes = lambda text, *args: text.encode()
            bot.runtime = Mock()
            bot.runtime.feature.side_effect = lambda name, **kwargs: name == "tts"
            bot._speak_whole_fallback = Mock()
            events = []

            def play(pcm, **kwargs):
                events.append(pcm.decode())
                return pcm.decode() != "Пожмём руки?"

            with self.subTest(parallel=parallel), patch.object(voice, "play_audio", side_effect=play), \
                    patch.object(voice, "resolve_output_device", return_value="default"), \
                    patch.object(voice, "TTS_PARALLEL_SENTENCES", parallel), \
                    patch.dict(os.environ, VALERA_ROBOT_GESTURES="1"), \
                    patch.object(gestures, "start_gesture", side_effect=events.append):
                bot._speak("Рад знакомству! Пожмём руки?")
                self.assertEqual(events[:3], ["Рад знакомству!", "shake_hand", "Пожмём руки?"])
                self.assertEqual(events.count("shake_hand"), 1)

    def test_command_on_each_side(self):
        with patch.object(gestures.subprocess, "run", return_value=Mock(returncode=0)) as run:
            self.assertTrue(gestures.send_gesture("shake_hand"))
            self.assertEqual(run.call_args.args[0][-1], "python3 ~/gesture_runner.py shake_hand")
            self.assertFalse(gestures.send_gesture("other"))
            self.assertEqual(run.call_count, 1)
            for name in ("shake_hand", "wave_hand"):
                self.assertEqual(gesture_runner.main([name]), 0)
                client = gesture_runner.Path.home() / "unitree_sdk2/build/bin/h1_loco_client"
                self.assertEqual(run.call_args, call([str(client), "--network_interface=eth0", "--" + name]))
