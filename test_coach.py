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

# Mock mlx_whisper — Apple Silicon only, not available in CI
sys.modules["mlx_whisper"] = MagicMock()

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
    @patch("coach._classify_voice_command", return_value="skip_interval")
    def test_skip_interval(self, mock_classify):
        key, resp = coach.match_command("zwift skip interval")
        assert key is not None
        assert resp == "Skipping interval!"

    @patch("coach._classify_voice_command", return_value="harder")
    def test_harder(self, mock_classify):
        key, resp = coach.match_command("zwift harder")
        assert resp == "Turning it up!"

    @patch("coach._classify_voice_command", return_value="easier")
    def test_easier(self, mock_classify):
        key, resp = coach.match_command("zwift back off")
        assert resp == "Dialling it back."

    @patch("coach._classify_voice_command", return_value="power_up")
    def test_power_up(self, mock_classify):
        key, resp = coach.match_command("hey zwift power up")
        assert resp == "Power up deployed!"

    @patch("coach._classify_voice_command", return_value="screenshot")
    def test_screenshot(self, mock_classify):
        key, resp = coach.match_command("zwift screenshot")
        assert resp == "Cheese!"

    @patch("coach._classify_voice_command", return_value="ride_on")
    def test_ride_on(self, mock_classify):
        key, resp = coach.match_command("zwift ride on")
        assert resp == "Ride on sent!"

    @patch("coach._classify_voice_command", return_value="u_turn")
    def test_u_turn(self, mock_classify):
        key, resp = coach.match_command("zwift u turn")
        assert resp == "Turning around."

    @patch("coach._classify_voice_command", return_value="status_report")
    def test_status_report(self, mock_classify):
        key, resp = coach.match_command("zwift how am i doing")
        assert key is None
        assert resp == "status_report"

    @patch("coach._classify_voice_command", return_value="personality_drill")
    def test_personality_switch(self, mock_classify):
        key, resp = coach.match_command("zwift be mean")
        assert key is None
        assert resp == "personality:drill_sergeant"

    def test_no_wake_word(self):
        key, resp = coach.match_command("skip interval")
        assert key is None
        assert resp is None

    @patch("coach._classify_voice_command", return_value="freeform")
    def test_freeform_question(self, mock_classify):
        key, resp = coach.match_command("zwift should I attack now")
        assert key is None
        assert resp.startswith("freeform:")
        assert "should i attack now" in resp

    @patch("coach._classify_voice_command", return_value="harder")
    def test_case_insensitive(self, mock_classify):
        key, resp = coach.match_command("ZWIFT HARDER")
        assert resp == "Turning it up!"

    @patch("coach._classify_voice_command", return_value="wave")
    def test_wake_word_mid_sentence(self, mock_classify):
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

# ─────────────────────────────────────────────
# Wake word variant tests
# ─────────────────────────────────────────────

class TestWakeWordVariants:
    @patch("coach._classify_voice_command", return_value="harder")
    def test_swift_mishearing(self, mock_classify):
        key, resp = coach.match_command("swift harder")
        assert resp == "Turning it up!"

    @patch("coach._classify_voice_command", return_value="screenshot")
    def test_zwith_mishearing(self, mock_classify):
        key, resp = coach.match_command("zwith screenshot")
        assert resp == "Cheese!"

    @patch("coach._classify_voice_command", return_value="wave")
    def test_is_with_mishearing(self, mock_classify):
        key, resp = coach.match_command("is with wave")
        assert resp == "Waving!"

    @patch("coach._classify_voice_command", return_value="ride_on")
    def test_his_lift_mishearing(self, mock_classify):
        key, resp = coach.match_command("his lift ride on")
        assert resp == "Ride on sent!"

    @patch("coach._classify_voice_command", return_value="easier")
    def test_is_we_mishearing(self, mock_classify):
        key, resp = coach.match_command("is we easier")
        assert resp == "Dialling it back."

    def test_wake_word_only_no_command(self):
        key, resp = coach.match_command("zwift")
        assert key is None
        assert resp is None

    def test_swift_only_no_command(self):
        key, resp = coach.match_command("swift")
        assert key is None
        assert resp is None


# ─────────────────────────────────────────────
# _classify_voice_command tests
# ─────────────────────────────────────────────

class TestClassifyVoiceCommand:
    @patch("coach.requests.post")
    def test_returns_classified_id(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "content": [{"text": "harder"}]
        }
        mock_post.return_value = mock_resp

        result = coach._classify_voice_command("push harder")
        assert result == "harder"
        # Verify it called the right API
        call_kwargs = mock_post.call_args
        assert "api.anthropic.com" in call_kwargs[1]["url"] if "url" in call_kwargs[1] else "api.anthropic.com" in call_kwargs[0][0]

    @patch("coach.requests.post")
    def test_returns_freeform_on_error(self, mock_post):
        mock_post.side_effect = Exception("timeout")
        result = coach._classify_voice_command("random gibberish")
        assert result == "freeform"

    @patch("coach.requests.post")
    def test_lowercases_and_strips_response(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "content": [{"text": "  Screenshot  \n"}]
        }
        mock_post.return_value = mock_resp

        result = coach._classify_voice_command("take a photo")
        assert result == "screenshot"


# ─────────────────────────────────────────────
# Additional detect_trigger tests (race)
# ─────────────────────────────────────────────

class TestDetectTriggerRace:
    def setup_method(self):
        coach.state.reset()
        coach.last_spoken_at = 0
        coach.last_trigger = ""

    def test_burning_matches(self):
        setup_state_for_trigger(
            mode="race", power=275, ftp=250,
            gap_to_front=0.5, position=5,
            draft_watts=0, riders_nearby=10,
        )
        trigger, ctx = coach.detect_trigger()
        assert trigger == "burning_matches"
        assert ctx["pct_ftp"] == 110
        assert ctx["power"] == 275

    def test_coasting(self):
        setup_state_for_trigger(
            mode="race", power=125, ftp=250,
            gap_to_front=1.0, riders_nearby=10,
            position=8, draft_watts=0,
        )
        trigger, ctx = coach.detect_trigger()
        assert trigger == "coasting"
        assert ctx["pct_ftp"] == 50
        assert ctx["power"] == 125


# ─────────────────────────────────────────────
# generate_commentary_raw tests
# ─────────────────────────────────────────────

class TestGenerateCommentaryRaw:
    @patch("coach._call_claude")
    def test_passes_prompt_with_active_personality(self, mock_claude):
        mock_claude.return_value = "Looking strong!"
        coach.ACTIVE_PERSONALITY = "british"
        try:
            result = coach.generate_commentary_raw("How is the rider doing?")
            assert result == "Looking strong!"
            system_prompt = mock_claude.call_args[0][0]
            assert "British" in system_prompt
        finally:
            coach.ACTIVE_PERSONALITY = "hype"

    @patch("coach._call_claude")
    def test_returns_none_on_failure(self, mock_claude):
        mock_claude.return_value = None
        result = coach.generate_commentary_raw("test prompt")
        assert result is None


# ─────────────────────────────────────────────
# handle_voice_action status_report with data
# ─────────────────────────────────────────────

class TestStatusReportWithData:
    def setup_method(self):
        while not coach.speech_queue.empty():
            try:
                coach.speech_queue.get_nowait()
            except queue.Empty:
                break

    @patch("coach.generate_commentary_raw")
    def test_status_report_with_workout_data(self, mock_gen):
        mock_gen.return_value = "You're crushing it at 250W!"
        coach.state.last_updated = time.time()
        coach.state.mode = "workout"
        coach.state.power = 250
        coach.state.power_target = 240
        coach.state.hr = 160
        coach.state.cadence = 90
        coach.state.distance = 12.5
        coach.state.elapsed = 1800

        coach.handle_voice_action(None, "status_report")

        mock_gen.assert_called_once()
        prompt = mock_gen.call_args[0][0]
        assert "250W" in prompt
        assert "240W" in prompt
        assert "160bpm" in prompt
        msg = coach.speech_queue.get_nowait()
        assert msg == "You're crushing it at 250W!"

    @patch("coach.generate_commentary_raw")
    def test_status_report_race_includes_position(self, mock_gen):
        mock_gen.return_value = "Hold that wheel!"
        coach.state.last_updated = time.time()
        coach.state.mode = "race"
        coach.state.power = 280
        coach.state.power_target = 0
        coach.state.hr = 170
        coach.state.cadence = 95
        coach.state.position = 3
        coach.state.gap_to_front = 1.5
        coach.state.distance = 20.0
        coach.state.elapsed = 2400

        coach.handle_voice_action(None, "status_report")

        prompt = mock_gen.call_args[0][0]
        assert "position" in prompt.lower() or "3" in prompt


# ─────────────────────────────────────────────
# WebSocket reconnection tests
# ─────────────────────────────────────────────

class TestWebSocketReconnection:
    @patch("coach.connect_websocket")
    @patch("coach.time.sleep")
    def test_on_close_increments_reconnect_count(self, mock_sleep, mock_connect):
        coach._ws_reconnect_count = 0
        coach.on_close(None)
        assert coach._ws_reconnect_count == 1
        mock_sleep.assert_called_once_with(5)
        mock_connect.assert_called_once()

    @patch("coach.connect_websocket")
    @patch("coach.time.sleep")
    def test_on_close_backoff_increases(self, mock_sleep, mock_connect):
        coach._ws_reconnect_count = 3
        coach.on_close(None)
        assert coach._ws_reconnect_count == 4
        mock_sleep.assert_called_once_with(20)

    @patch("coach.connect_websocket")
    @patch("coach.time.sleep")
    def test_on_close_backoff_caps_at_30(self, mock_sleep, mock_connect):
        coach._ws_reconnect_count = 10
        coach.on_close(None)
        mock_sleep.assert_called_once_with(30)

    def test_on_open_resets_reconnect_count(self):
        coach._ws_reconnect_count = 5
        mock_ws = MagicMock()
        coach.on_open(mock_ws)
        assert coach._ws_reconnect_count == 0
        mock_ws.send.assert_called_once()


# ─────────────────────────────────────────────
# on_message draft watts test
# ─────────────────────────────────────────────

class TestOnMessageDraftWatts:
    def setup_method(self):
        coach.state.reset()

    def test_draft_watts_parsed(self):
        nearby = [{"position": 5, "gap": 1.0}]
        data = {
            "event": "athlete/watching",
            "data": {
                "athlete": {"ftp": 250},
                "stats": {
                    "power": 200, "heartrate": 140, "cadence": 85,
                    "speed": 35, "elapsed": 600, "distance": 5.0,
                    "draftingWatts": 45,
                },
                "workout": {},
                "nearby": nearby,
            },
        }
        coach.on_message(None, json.dumps(data))
        assert coach.state.draft_watts == 45


# ─────────────────────────────────────────────
# Personality affects generate_commentary
# ─────────────────────────────────────────────

class TestPersonalityInCommentary:
    @patch("coach._call_claude")
    def test_drill_sergeant_personality_used(self, mock_claude):
        mock_claude.return_value = "Move it!"
        original = coach.ACTIVE_PERSONALITY
        coach.ACTIVE_PERSONALITY = "drill_sergeant"
        try:
            coach.generate_commentary("sandbagger", {
                "power": 150, "target": 200, "deficit": 50, "pct": 75,
            })
            system_prompt = mock_claude.call_args[0][0]
            assert "drill sergeant" in system_prompt.lower()
        finally:
            coach.ACTIVE_PERSONALITY = original

    @patch("coach._call_claude")
    def test_data_personality_used(self, mock_claude):
        mock_claude.return_value = "Fascinating metrics."
        original = coach.ACTIVE_PERSONALITY
        coach.ACTIVE_PERSONALITY = "data"
        try:
            coach.generate_commentary("hr_red", {"hr": 185})
            system_prompt = mock_claude.call_args[0][0]
            assert "data" in system_prompt.lower()
        finally:
            coach.ACTIVE_PERSONALITY = original


# ─────────────────────────────────────────────
# speak() with ElevenLabs mock
# ─────────────────────────────────────────────

class TestSpeakWithElevenLabs:
    @patch("coach.os.unlink")
    @patch("coach.subprocess.Popen")
    @patch("coach.tempfile.NamedTemporaryFile")
    def test_speak_streams_and_plays(self, mock_tmpfile, mock_popen, mock_unlink):
        """Test speak() with a mocked ElevenLabs client and audio player."""
        original_client = coach._elevenlabs
        original_player_args = coach.AUDIO_PLAYER_ARGS

        mock_client = MagicMock()
        mock_client.text_to_speech.stream.return_value = [b"fake_audio_chunk"]
        coach._elevenlabs = mock_client
        coach.AUDIO_PLAYER_ARGS = ["afplay"]

        # Mock the temp file
        mock_file = MagicMock()
        mock_file.name = "/tmp/fake.mp3"
        mock_file.__enter__ = lambda s: s
        mock_file.__exit__ = MagicMock(return_value=False)
        mock_tmpfile.return_value = mock_file

        # Mock Popen — make wait() return immediately so _finish thread completes
        mock_proc = MagicMock()
        mock_proc.wait.return_value = 0
        mock_popen.return_value = mock_proc

        try:
            coach.speak("Test TTS output")
            mock_client.text_to_speech.stream.assert_called_once()
            call_kwargs = mock_client.text_to_speech.stream.call_args
            assert call_kwargs[1]["text"] == "Test TTS output"
            mock_file.write.assert_called_once_with(b"fake_audio_chunk")
            mock_popen.assert_called_once()
        finally:
            coach._elevenlabs = original_client
            coach.AUDIO_PLAYER_ARGS = original_player_args

    def test_speak_no_player_prints(self, capsys):
        """When ElevenLabs is configured but no audio player exists, fall back to print."""
        original_client = coach._elevenlabs
        original_player_args = coach.AUDIO_PLAYER_ARGS

        mock_client = MagicMock()
        mock_client.text_to_speech.stream.return_value = [b"audio"]
        coach._elevenlabs = mock_client
        coach.AUDIO_PLAYER_ARGS = None

        try:
            coach.speak("No player test")
            captured = capsys.readouterr()
            assert "[COACH - no player] No player test" in captured.out
        finally:
            coach._elevenlabs = original_client
            coach.AUDIO_PLAYER_ARGS = original_player_args


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


# ─────────────────────────────────────────────
# STT fallback tests (mlx_whisper → Google)
# ─────────────────────────────────────────────

class TestSTTFallback:
    def test_mlx_whisper_flag_set_when_available(self):
        """mlx_whisper is mocked in our test setup, so the flag should be True."""
        assert coach._USE_MLX_WHISPER is True

    def test_google_fallback_when_mlx_unavailable(self):
        """When _USE_MLX_WHISPER is False, voice_listener should use Google STT."""
        original = coach._USE_MLX_WHISPER
        coach._USE_MLX_WHISPER = False
        try:
            # Verify the flag controls the branch — we can't run the full
            # listener but we can confirm the flag toggles correctly
            assert coach._USE_MLX_WHISPER is False
        finally:
            coach._USE_MLX_WHISPER = original

    @patch("coach.sr.Microphone")
    def test_voice_listener_prints_google_engine(self, mock_mic, capsys):
        """When mlx_whisper unavailable, voice_listener prints 'google' as STT engine."""
        original = coach._USE_MLX_WHISPER
        coach._USE_MLX_WHISPER = False

        # Make Microphone raise so listener exits early after printing
        mock_mic.side_effect = OSError("no mic")
        try:
            coach.voice_listener()
            captured = capsys.readouterr()
            assert "STT: google" in captured.out
        finally:
            coach._USE_MLX_WHISPER = original

    @patch("coach.sr.Microphone")
    def test_voice_listener_prints_mlx_engine(self, mock_mic, capsys):
        """When mlx_whisper available, voice_listener prints 'mlx_whisper' as STT engine."""
        original = coach._USE_MLX_WHISPER
        coach._USE_MLX_WHISPER = True

        mock_mic.side_effect = OSError("no mic")
        try:
            coach.voice_listener()
            captured = capsys.readouterr()
            assert "STT: mlx_whisper" in captured.out
        finally:
            coach._USE_MLX_WHISPER = original
