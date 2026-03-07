"""Tests for Zwift AI Coach — uses mock data, no hardware or API keys needed."""

import json
import sys
import time
import queue
from unittest.mock import patch, MagicMock
import pytest

# Mock pynput before importing coach — it requires an X display which
# isn't available in headless CI environments.
mock_pynput = MagicMock()
mock_key = MagicMock()
mock_key.tab = "tab"
mock_key.page_up = "page_up"
mock_key.page_down = "page_down"
mock_key.space = "space"
mock_key.f10 = "f10"
mock_key.f3 = "f3"
mock_key.f2 = "f2"
mock_key.down = "down"
mock_pynput.keyboard.Key = mock_key
mock_pynput.keyboard.Controller = MagicMock
sys.modules["pynput"] = mock_pynput
sys.modules["pynput.keyboard"] = mock_pynput.keyboard

import coach


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def make_ws_message(power=200, heartrate=140, cadence=85, speed=35,
                    elapsed=600, distance=5.0, ftp=250,
                    workout=None, nearby=None):
    """Build a fake Sauce WebSocket JSON message."""
    data = {
        "event": "athlete/watching",
        "data": {
            "athlete": {"ftp": ftp},
            "stats": {
                "power": power,
                "heartrate": heartrate,
                "cadence": cadence,
                "speed": speed,
                "elapsed": elapsed,
                "distance": distance,
                "draftingWatts": 0,
            },
            "workout": workout or {},
            "nearby": nearby or [],
        },
    }
    return json.dumps(data)


def setup_state_for_trigger(**kwargs):
    """Set coach module globals so detect_trigger() can fire."""
    coach.state.reset()
    for k, v in kwargs.items():
        setattr(coach.state, k, v)
    coach.state.last_updated = time.time()
    coach.last_spoken_at = 0
    coach.last_trigger = ""


# ─────────────────────────────────────────────
# match_command tests
# ─────────────────────────────────────────────

class TestMatchCommand:
    def test_skip_interval(self):
        key, resp = coach.match_command("zwift skip interval")
        assert key is not None
        assert resp == "Skipping interval!"

    def test_harder(self):
        key, resp = coach.match_command("zwift harder")
        assert resp == "Turning it up!"

    def test_easier(self):
        key, resp = coach.match_command("zwift back off")
        assert resp == "Dialling it back."

    def test_power_up(self):
        key, resp = coach.match_command("hey zwift power up")
        assert resp == "Power up deployed!"

    def test_screenshot(self):
        key, resp = coach.match_command("zwift screenshot")
        assert resp == "Cheese!"

    def test_ride_on(self):
        key, resp = coach.match_command("zwift ride on")
        assert resp == "Ride on sent!"

    def test_u_turn(self):
        key, resp = coach.match_command("zwift u turn")
        assert resp == "Turning around."

    def test_status_report(self):
        key, resp = coach.match_command("zwift how am i doing")
        assert key is None
        assert resp == "status_report"

    def test_personality_switch(self):
        key, resp = coach.match_command("zwift be mean")
        assert key is None
        assert resp == "personality:drill_sergeant"

    def test_no_wake_word(self):
        key, resp = coach.match_command("skip interval")
        assert key is None
        assert resp is None

    def test_freeform_question(self):
        key, resp = coach.match_command("zwift should I attack now")
        assert key is None
        assert resp.startswith("freeform:")
        assert "should i attack now" in resp

    def test_case_insensitive(self):
        key, resp = coach.match_command("ZWIFT HARDER")
        assert resp == "Turning it up!"

    def test_wake_word_mid_sentence(self):
        key, resp = coach.match_command("hey zwift wave")
        assert resp == "Waving!"

    def test_empty_string(self):
        key, resp = coach.match_command("")
        assert key is None
        assert resp is None


# ─────────────────────────────────────────────
# on_message / WebSocket parsing tests
# ─────────────────────────────────────────────

class TestOnMessage:
    def setup_method(self):
        coach.state.reset()

    def test_basic_stats(self):
        msg = make_ws_message(power=250, heartrate=155, cadence=90, ftp=300)
        coach.on_message(None, msg)
        assert coach.state.power == 250
        assert coach.state.hr == 155
        assert coach.state.cadence == 90
        assert coach.state.ftp == 300
        assert coach.state.last_updated > 0

    def test_workout_mode(self):
        msg = make_ws_message(
            workout={"power": 200, "name": "Threshold", "remaining": 120}
        )
        coach.on_message(None, msg)
        assert coach.state.mode == "workout"
        assert coach.state.power_target == 200
        assert coach.state.workout_block == "Threshold"
        assert coach.state.workout_block_remaining == 120

    def test_freeride_mode(self):
        msg = make_ws_message(workout={}, nearby=[])
        coach.on_message(None, msg)
        assert coach.state.mode == "freeride"

    def test_race_mode(self):
        nearby = [
            {"position": 5, "gap": 1.2},
            {"position": 6, "gap": 2.0},
        ]
        msg = make_ws_message(nearby=nearby)
        coach.on_message(None, msg)
        assert coach.state.mode == "race"
        assert coach.state.position == 5
        assert coach.state.riders_nearby == 2
        assert coach.state.gap_to_front == 1.2

    def test_ignores_non_athlete_events(self):
        msg = json.dumps({"event": "other/event", "data": {}})
        coach.on_message(None, msg)
        assert coach.state.last_updated == 0

    def test_handles_malformed_json(self):
        coach.on_message(None, "not json at all")
        assert coach.state.last_updated == 0

    def test_preserves_previous_values_on_missing_keys(self):
        coach.state.power = 999
        msg = json.dumps({
            "event": "athlete/watching",
            "data": {"athlete": {}, "stats": {}, "workout": {}, "nearby": []},
        })
        coach.on_message(None, msg)
        assert coach.state.power == 999  # not overwritten


# ─────────────────────────────────────────────
# detect_trigger tests
# ─────────────────────────────────────────────

class TestDetectTrigger:
    def setup_method(self):
        coach.state.reset()
        coach.last_spoken_at = 0
        coach.last_trigger = ""

    def test_no_data_yet(self):
        trigger, ctx = coach.detect_trigger()
        assert trigger is None

    def test_cooldown_suppresses(self):
        setup_state_for_trigger(mode="workout", power=100, power_target=200)
        coach.last_spoken_at = time.time()  # just spoke
        trigger, ctx = coach.detect_trigger()
        assert trigger is None

    def test_sandbagger(self):
        setup_state_for_trigger(
            mode="workout", power=150, power_target=200, ftp=250
        )
        trigger, ctx = coach.detect_trigger()
        assert trigger == "sandbagger"
        assert ctx["power"] == 150
        assert ctx["target"] == 200
        assert ctx["deficit"] == 50

    def test_overcooking(self):
        setup_state_for_trigger(
            mode="workout", power=280, power_target=200, ftp=250
        )
        trigger, ctx = coach.detect_trigger()
        assert trigger == "overcooking"
        assert ctx["excess"] == 80

    def test_countdown(self):
        setup_state_for_trigger(
            mode="workout", power=200, power_target=200,
            workout_block="Threshold", workout_block_remaining=8, ftp=250
        )
        trigger, ctx = coach.detect_trigger()
        assert trigger == "countdown"
        assert ctx["seconds"] == 8

    def test_low_cadence(self):
        setup_state_for_trigger(
            mode="workout", power=200, power_target=200,
            cadence=55, ftp=250
        )
        trigger, ctx = coach.detect_trigger()
        assert trigger == "low_cadence"
        assert ctx["cadence"] == 55

    def test_hr_red(self):
        setup_state_for_trigger(
            mode="workout", power=200, power_target=200,
            hr=185, ftp=250
        )
        trigger, ctx = coach.detect_trigger()
        assert trigger == "hr_red"
        assert ctx["hr"] == 185

    def test_on_target(self):
        setup_state_for_trigger(
            mode="workout", power=200, power_target=200,
            workout_block="Sweet Spot", workout_block_remaining=60, ftp=250
        )
        coach.last_spoken_at = time.time() - 50  # past cooldown + 45s threshold
        trigger, ctx = coach.detect_trigger()
        assert trigger == "on_target"

    def test_dropped_in_race(self):
        setup_state_for_trigger(
            mode="race", power=200, gap_to_front=5.0,
            riders_nearby=10, ftp=250
        )
        trigger, ctx = coach.detect_trigger()
        assert trigger == "dropped"
        assert ctx["gap"] == 5.0

    def test_good_draft(self):
        setup_state_for_trigger(
            mode="race", power=200, draft_watts=50,
            position=8, ftp=250
        )
        trigger, ctx = coach.detect_trigger()
        assert trigger == "good_draft"
        assert ctx["savings"] == 50

    def test_podium_position(self):
        setup_state_for_trigger(
            mode="race", power=280, position=2, ftp=250,
            gap_to_front=0, draft_watts=0
        )
        trigger, ctx = coach.detect_trigger()
        assert trigger == "podium_position"
        assert ctx["position"] == 2

    def test_last_trigger_prevents_repeat(self):
        setup_state_for_trigger(
            mode="workout", power=150, power_target=200, ftp=250
        )
        coach.last_trigger = "sandbagger"
        trigger, ctx = coach.detect_trigger()
        # sandbagger suppressed; check if a different trigger fires or none
        assert trigger != "sandbagger"

    def test_freeride_no_triggers(self):
        setup_state_for_trigger(mode="freeride", power=200, ftp=250)
        trigger, ctx = coach.detect_trigger()
        assert trigger is None


# ─────────────────────────────────────────────
# _call_claude tests (mocked HTTP)
# ─────────────────────────────────────────────

class TestCallClaude:
    @patch("coach.requests.post")
    def test_strips_markdown_emphasis(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "content": [{"text": "You're doing *great* today!"}]
        }
        mock_post.return_value = mock_resp

        result = coach._call_claude("system", "prompt")
        assert result == "You're doing great today!"

    @patch("coach.requests.post")
    def test_returns_none_on_error(self, mock_post):
        mock_post.side_effect = Exception("network error")
        result = coach._call_claude("system", "prompt")
        assert result is None

    @patch("coach.requests.post")
    def test_plain_text_unchanged(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "content": [{"text": "Keep pushing, you got this!"}]
        }
        mock_post.return_value = mock_resp

        result = coach._call_claude("system", "prompt")
        assert result == "Keep pushing, you got this!"


# ─────────────────────────────────────────────
# generate_commentary tests
# ─────────────────────────────────────────────

class TestGenerateCommentary:
    @patch("coach._call_claude")
    def test_formats_template(self, mock_claude):
        mock_claude.return_value = "Nice work!"
        ctx = {"power": 200, "target": 200, "remaining": 30, "block": "Tempo"}
        result = coach.generate_commentary("on_target", ctx)

        # Verify the prompt was formatted with context values
        call_args = mock_claude.call_args[0]
        assert "200W" in call_args[1]
        assert "Tempo" in call_args[1]
        assert result == "Nice work!"

    @patch("coach._call_claude")
    def test_unknown_trigger_uses_default(self, mock_claude):
        mock_claude.return_value = "Something."
        result = coach.generate_commentary("unknown_trigger", {})
        call_args = mock_claude.call_args[0]
        assert "Comment on current ride state" in call_args[1]


# ─────────────────────────────────────────────
# handle_voice_action tests
# ─────────────────────────────────────────────

class TestHandleVoiceAction:
    def setup_method(self):
        # Drain the speech queue
        while not coach.speech_queue.empty():
            try:
                coach.speech_queue.get_nowait()
            except queue.Empty:
                break

    @patch.object(coach.keyboard, "press")
    @patch.object(coach.keyboard, "release")
    def test_keypress_action(self, mock_release, mock_press):
        coach.handle_voice_action(mock_key.tab, "Skipping interval!")
        mock_press.assert_called_once_with(mock_key.tab)
        mock_release.assert_called_once_with(mock_key.tab)
        assert coach.speech_queue.get_nowait() == "Skipping interval!"

    def test_personality_switch(self):
        coach.handle_voice_action(None, "personality:british")
        assert coach.ACTIVE_PERSONALITY == "british"
        msg = coach.speech_queue.get_nowait()
        assert "endeavour" in msg.lower()
        # Reset
        coach.ACTIVE_PERSONALITY = "hype"

    def test_status_report_no_data(self):
        coach.state.last_updated = 0
        coach.handle_voice_action(None, "status_report")
        msg = coach.speech_queue.get_nowait()
        assert "don't have any ride data" in msg.lower()

    @patch("coach.generate_commentary_raw")
    def test_freeform_calls_claude(self, mock_gen):
        mock_gen.return_value = "You should attack!"
        coach.state.last_updated = time.time()
        coach.handle_voice_action(None, "freeform:should I attack")
        mock_gen.assert_called_once()
        assert "should I attack" in mock_gen.call_args[0][0]
        msg = coach.speech_queue.get_nowait()
        assert msg == "You should attack!"


# ─────────────────────────────────────────────
# RideState tests
# ─────────────────────────────────────────────

class TestRideState:
    def test_initial_values(self):
        s = coach.RideState()
        assert s.power == 0
        assert s.mode == "unknown"
        assert s.last_updated == 0

    def test_reset(self):
        s = coach.RideState()
        s.power = 300
        s.mode = "race"
        s.reset()
        assert s.power == 0
        assert s.mode == "unknown"


# ─────────────────────────────────────────────
# speak() tests (mocked, no real audio)
# ─────────────────────────────────────────────

class TestSpeak:
    def test_no_api_key_prints(self, capsys):
        original = coach.ELEVENLABS_KEY
        coach.ELEVENLABS_KEY = ""
        try:
            coach.speak("Test message")
            captured = capsys.readouterr()
            assert "[COACH] Test message" in captured.out
        finally:
            coach.ELEVENLABS_KEY = original

    @patch("coach.requests.post")
    def test_api_error_falls_back_to_print(self, mock_post, capsys):
        original = coach.ELEVENLABS_KEY
        coach.ELEVENLABS_KEY = "fake-key"
        mock_post.side_effect = Exception("API down")
        try:
            coach.speak("Fallback test")
            captured = capsys.readouterr()
            assert "[COACH] Fallback test" in captured.out
        finally:
            coach.ELEVENLABS_KEY = original


# ─────────────────────────────────────────────
# Audio player detection tests
# ─────────────────────────────────────────────

class TestAudioPlayerDetection:
    @patch("coach.shutil.which")
    def test_finds_afplay(self, mock_which):
        mock_which.side_effect = lambda name: "/usr/bin/afplay" if name == "afplay" else None
        name, args = coach._detect_audio_player()
        assert name == "afplay"
        assert args == ["afplay"]

    @patch("coach.shutil.which")
    def test_falls_back_to_mpg123(self, mock_which):
        mock_which.side_effect = lambda name: "/usr/bin/mpg123" if name == "mpg123" else None
        name, args = coach._detect_audio_player()
        assert name == "mpg123"
        assert args == ["mpg123", "-q"]

    @patch("coach.shutil.which")
    def test_no_player_found(self, mock_which):
        mock_which.return_value = None
        name, args = coach._detect_audio_player()
        assert name is None
        assert args is None
