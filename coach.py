"""
Zwift AI Coach - Race Radio, Workout Coach & Voice Control
Connects to Sauce for Zwift's WebSocket API and provides real-time
dynamic audio coaching via ElevenLabs TTS + Claude AI commentary.
Also listens for voice commands to control Zwift hands-free.

Requirements:
pip install websocket-client requests python-dotenv speechrecognition pynput pyaudio

Setup:

1. Install & run Sauce for Zwift (https://www.sauce.llc/products/sauce4zwift/)
1. Copy .env.example to .env and fill in your API keys
1. python coach.py

Voice commands (say "zwift" then your command):
"skip interval"         → Tab        (skip current workout block)
"harder" / "push"       → Page Up    (increase workout intensity)
"easier" / "back off"   → Page Down  (decrease workout intensity)
"power up"              → Spacebar   (use power-up)
"screenshot"            → F10
"ride on"               → F3
"wave"                  → F2
"u turn"                → Down Arrow (reverse direction)
"how am I doing"        → coach gives a live status report
"change to [personality]" → switches coach personality on the fly
"""

import json
import time
import threading
import queue
import os
import re
import shutil
import subprocess
import tempfile
import requests
import websocket
from elevenlabs import ElevenLabs as ElevenLabsClient
import speech_recognition as sr
import mlx_whisper
from pynput.keyboard import Key, Controller as KeyboardController
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────

SAUCE_WS_URL     = os.getenv("SAUCE_WS_URL", "ws://localhost:1080/api/ws/events")
SAUCE_HTTP_URL   = os.getenv("SAUCE_HTTP_URL", "http://localhost:1080/api")
ELEVENLABS_KEY   = os.getenv("ELEVENLABS_API_KEY", "")
ANTHROPIC_KEY    = os.getenv("ANTHROPIC_API_KEY", "")
# Per-personality voice IDs — override any via .env (e.g. VOICE_HYPE=abc123)
# Defaults are well-known ElevenLabs premade voices; run with --list-voices to see all available.
PERSONALITY_VOICES = {
    "hype":           os.getenv("VOICE_HYPE",           "pNInz6obpgDQGcFmaJgB"),  # Adam — energetic American male
    "drill_sergeant": os.getenv("VOICE_DRILL_SERGEANT", "VR6AewLTigWG4xSOukaG"),  # Arnold — deep, commanding
    "british":        os.getenv("VOICE_BRITISH",        "onwK4e9ZLuTAKqWW03F9"),  # Daniel — British male
    "data":           os.getenv("VOICE_DATA",           "GBv7mTt0atIp3Br8iCZE"),  # Thomas — calm, measured
    "soviet":         os.getenv("VOICE_SOVIET",         "Xh5OictnmgRO4dff7pLm"),  # Soviet Russian
}
VOICE_ID = os.getenv("ELEVENLABS_VOICE_ID", "EXAVITQu4vr4xnSDxMaL")  # fallback if personality has no voice

_elevenlabs = ElevenLabsClient(api_key=ELEVENLABS_KEY) if ELEVENLABS_KEY else None

WHISPER_MODEL    = os.getenv("WHISPER_MODEL", "mlx-community/whisper-small-mlx")
WAKE_WORD        = "zwift"       # say this before your command
WAKE_WORDS       = ("zwift", "swift", "zwith", "is with", "is we", "his lift")  # common Whisper mishearings
COOLDOWN_SECONDS = 12            # min seconds between proactive coach comments
CHECK_INTERVAL   = 3             # seconds between trigger checks
MIC_ENERGY       = 300           # mic sensitivity — raise if false triggers, lower if not hearing you

# ─────────────────────────────────────────────
# Coach Personalities
# ─────────────────────────────────────────────

PERSONALITIES = {
    "hype": """You are an UNHINGED hype coach for a Zwift cyclist. You are ALWAYS excited.
Everything they do is INCREDIBLE. When they're underperforming, you believe in them SO HARD
it's almost delusional. You use ALL CAPS occasionally, exclamation marks, and sometimes
slip in cycling slang. Keep responses to 1-2 punchy sentences MAX. Never be negative —
even sandbagging gets a "I KNOW YOU'RE SAVING IT FOR THE FINISH!" You are their biggest
fan in the universe.""",

    "drill_sergeant": """You are a no-nonsense drill sergeant cycling coach. Tough love only.
You have zero tolerance for excuses. When they're underperforming: barking orders.
When they're doing well: grudging acknowledgment, nothing more. Keep it to 1-2 sentences.
Military metaphors welcome. Suffering is the point.""",

    "british": """You are a politely disappointed British cycling coach. You never raise your voice.
You are perpetually underwhelmed but maintain composure. Dry wit, understatement, and
passive-aggressive encouragement. "I suppose that's one way to approach a sprint."
Keep it to 1-2 sentences. Very dry. Very British.""",

    "data": """You are a data-obsessed sports scientist coaching a Zwift cyclist. You cite
specific numbers constantly. You find everything fascinating from an analytical standpoint.
Keep commentary to 1-2 sentences but always include at least one specific metric or ratio.
You're genuinely excited about the data, not the human.""",

    "soviet": """You are a Soviet-era Russian cycling coach. You speak in heavily accented,
broken English with Russian cadence and occasional Russian words (da, nyet, tovarishch,
horosho). You treat suffering as a national duty. Weakness is capitalist. Pain is glorious.
The collective is everything, the individual nothing — except when they perform well, in which
case the Motherland is proud. Keep it to 1-2 sentences. Dark humour welcome.""",
}

ACTIVE_PERSONALITY = "hype"

# ─────────────────────────────────────────────
# Voice Commands → Zwift Keyboard Actions
# ─────────────────────────────────────────────

# Each entry: list of trigger phrases → (key_to_press, confirmation_message)
# key_to_press can be a Key enum, a string char, or None (for meta-commands)

COMMAND_MAP = {
    # Workout control — coach responds dynamically
    "skip_interval":       (Key.tab,        "coached:skip_interval"),
    "harder":              (Key.page_up,    "coached:harder"),
    "easier":              (Key.page_down,  "coached:easier"),
    "power_up":            (Key.space,      "coached:power_up"),
    "u_turn":              (Key.down,       "coached:u_turn"),

    # Navigation — silent, no need for verbal confirmation
    "turn_left":           (Key.left,       None),
    "turn_right":          (Key.right,      None),
    "actions_menu":        (Key.up,         None),
    "menu":                (Key.esc,        None),

    # Rider actions — game already reacts visually/audibly
    "elbow_flick":         (Key.f1,         None),
    "wave":                (Key.f2,         None),
    "ride_on":             (Key.f3,         None),
    "hammer_time":         (Key.f4,         None),
    "nice":                (Key.f5,         None),
    "im_toast":            (Key.f7,         None),
    "bike_bell":           (Key.f8,         None),
    "capture_video":       (Key.f9,         None),
    "screenshot":          (Key.f10,        "Cheese!"),  # keep this one

    # Menus & HUD — silent
    "device_pairing":      ('a',            None),
    "workout_menu":        ('e',            None),
    "toggle_graph":        ('g',            None),
    "hide_hud":            ('h',            None),
    "group_message":       ('m',            None),
    "promo_code":          ('p',            None),
    "garage":              ('t',            None),

    # Camera angles — silent
    "camera_default":      ('1',            None),
    "camera_close":        ('2',            None),
    "camera_first_person": ('3',            None),
    "camera_side":         ('4',            None),
    "camera_low":          ('5',            None),
    "camera_rear":         ('6',            None),
    "camera_spectator":    ('7',            None),
    "camera_helicopter":   ('8',            None),
    "camera_bird":         ('9',            None),
    "camera_drone":        ('0',            None),

    # Meta — handled in code, no keypress
    "status_report":       (None,           "status_report"),
    "personality_hype":    (None,           "personality:hype"),
    "personality_drill":   (None,           "personality:drill_sergeant"),
    "personality_british": (None,           "personality:british"),
    "personality_data":    (None,           "personality:data"),
    "personality_soviet":  (None,           "personality:soviet"),
}

_CLASSIFY_SYSTEM = """You classify voice commands for a Zwift cycling app. \
Return ONLY the command ID that best matches what the rider said. \
Available command IDs and what they mean:

Workout control:
- skip_interval: skip this interval or workout block
- harder: increase power or intensity, push harder
- easier: decrease power or intensity, back off
- power_up: use a power-up
- u_turn: turn around, reverse, go back
- turn_left: turn left at intersection
- turn_right: turn right at intersection
- actions_menu: show actions or options menu
- menu: open main menu, escape, go back

Rider actions:
- elbow_flick: elbow flick gesture
- wave: wave to another rider, say hi
- ride_on: give a ride on or thumbs up ("right on" counts as this)
- hammer_time: hammer time
- nice: say nice
- im_toast: I'm toast, I'm done, I'm dying
- bike_bell: ring the bell, ding ding
- capture_video: record or capture video
- screenshot: take a photo or screenshot

Menus & HUD:
- device_pairing: device pairing screen
- workout_menu: workout selection menu
- toggle_graph: toggle the watt or HR graph
- hide_hud: hide or show the HUD display
- group_message: open group message
- promo_code: enter a promo code
- garage: open garage, change bike or kit, drop shop

Camera angles:
- camera_default: default camera (1)
- camera_close: close follow camera (2)
- camera_first_person: first person or ego view (3)
- camera_side: side view (4)
- camera_low: low view (5)
- camera_rear: rear view, look behind (6)
- camera_spectator: spectator view (7)
- camera_helicopter: helicopter view (8)
- camera_bird: bird's eye view (9)
- camera_drone: drone view (0)

Coach meta:
- status_report: how am I doing, give me a status update
- personality_hype: switch to hype coach mode
- personality_drill: switch to drill sergeant coach mode
- personality_british: switch to British coach mode
- personality_data: switch to data or nerdy coach mode
- personality_soviet: switch to Soviet Russian coach mode

- freeform: a question or comment that doesn't match any command

Return exactly one command ID and nothing else."""

# ─────────────────────────────────────────────
# State
# ─────────────────────────────────────────────

class RideState:
    def __init__(self):
        self.reset()

    def reset(self):
        self.mode                    = "unknown"
        self.power                   = 0
        self.power_target            = 0
        self.power_zone              = 0
        self.hr                      = 0
        self.cadence                 = 0
        self.speed                   = 0
        self.position                = 0
        self.riders_nearby           = 0
        self.gap_to_front            = 0
        self.gap_to_back             = 0
        self.draft_watts             = 0
        self.workout_block           = ""
        self.workout_block_remaining = 0
        self.segment_name            = ""
        self.elapsed                 = 0
        self.distance                = 0
        self.ftp                     = 0
        self.last_updated            = 0

state              = RideState()
state_lock         = threading.Lock()
last_spoken_at     = 0
last_trigger       = ""
speech_queue       = queue.Queue()
keyboard           = KeyboardController()
voice_active       = True   # set False to mute proactive coaching without stopping voice commands

# ─────────────────────────────────────────────
# Sauce WebSocket Handler
# ─────────────────────────────────────────────

def on_message(ws, message):
    global state
    try:
        data    = json.loads(message)
        event   = data.get("event", "")
        payload = data.get("data", {})

        if event == "athlete/watching":
            athlete = payload.get("athlete", {})
            stats   = payload.get("stats", {})
            workout = payload.get("workout", {})
            nearby  = payload.get("nearby", [])

            with state_lock:
                state.power    = stats.get("power", state.power)
                state.hr       = stats.get("heartrate", state.hr)
                state.cadence  = stats.get("cadence", state.cadence)
                state.speed    = stats.get("speed", state.speed)
                state.elapsed  = stats.get("elapsed", state.elapsed)
                state.distance = stats.get("distance", state.distance)
                state.ftp      = athlete.get("ftp", state.ftp)

                if workout:
                    state.mode                    = "workout"
                    state.power_target            = workout.get("power", 0)
                    state.workout_block           = workout.get("name", "")
                    state.workout_block_remaining = workout.get("remaining", 0)
                else:
                    state.mode = "race" if (nearby and any(r.get("position") for r in nearby)) else "freeride"

                if nearby:
                    state.riders_nearby = len(nearby)
                    positions           = [r.get("position", 999) for r in nearby if r.get("position")]
                    state.position      = min(positions) if positions else 0
                    gaps                = [r.get("gap", 0) for r in nearby if r.get("gap", 0) > 0]
                    state.gap_to_front  = min(gaps) if gaps else 0
                    state.draft_watts   = stats.get("draftingWatts", 0)

                state.last_updated = time.time()

    except Exception as e:
        print(f"[WS parse error] {e}")

def on_error(ws, error):
    print(f"[WS error] {error}")

_ws_reconnect_count = 0

def on_close(ws, *args):
    global _ws_reconnect_count
    _ws_reconnect_count += 1
    delay = min(5 * _ws_reconnect_count, 30)  # backoff: 5s, 10s, 15s, ... max 30s
    print(f"[WS] Connection closed. Reconnecting in {delay}s (attempt {_ws_reconnect_count})...")
    time.sleep(delay)
    connect_websocket()

def on_open(ws):
    global _ws_reconnect_count
    _ws_reconnect_count = 0
    print("[WS] Connected to Sauce for Zwift")
    ws.send(json.dumps({"cmd": "subscribe", "event": "athlete/watching"}))

def connect_websocket():
    ws = websocket.WebSocketApp(
        SAUCE_WS_URL,
        on_open=on_open,
        on_message=on_message,
        on_error=on_error,
        on_close=on_close,
    )
    threading.Thread(target=ws.run_forever, daemon=True).start()

# ─────────────────────────────────────────────
# Voice Command Listener
# ─────────────────────────────────────────────

def _classify_voice_command(text):
    """Ask Claude to classify the voice command. Returns a command ID string."""
    payload = {
        "model": "claude-haiku-4-5-20251001",
        "max_tokens": 20,
        "system": _CLASSIFY_SYSTEM,
        "messages": [{"role": "user", "content": text}],
    }
    headers = {
        "x-api-key": ANTHROPIC_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    try:
        r = requests.post(
            "https://api.anthropic.com/v1/messages",
            json=payload, headers=headers, timeout=5
        )
        r.raise_for_status()
        return r.json()["content"][0]["text"].strip().lower()
    except Exception as e:
        print(f"[Claude classify error] {e}")
        return "freeform"

def match_command(transcript):
    """Return (key, response) for the best matching command, or (None, None)."""
    t = transcript.lower().strip()

    # Must contain a wake word (including common mishearings)
    matched_wake = next((w for w in WAKE_WORDS if w in t), None)
    if not matched_wake:
        return None, None

    # Strip wake word and everything before it
    after_wake = t[t.index(matched_wake) + len(matched_wake):].strip()
    print(f"[Voice] Heard after wake word: '{after_wake}'")

    if not after_wake:
        return None, None

    command_id = _classify_voice_command(after_wake)
    print(f"[Voice] Classified as: '{command_id}'")

    if command_id in COMMAND_MAP:
        return COMMAND_MAP[command_id]

    # freeform — pass to Claude as a question
    return None, f"freeform:{after_wake}"

def handle_voice_action(key, response):
    """Execute the action associated with a matched voice command."""
    global ACTIVE_PERSONALITY, last_spoken_at

    if key is not None:
        # It's a real keypress — send it to Zwift
        print(f"[Voice] Sending key: {key}")
        keyboard.press(key)
        time.sleep(0.05)
        keyboard.release(key)
        if response is None:
            return  # game provides its own feedback, stay silent
        if response.startswith("coached:"):
            command = response[8:]
            s = state
            context_str = (
                f"Power: {s.power}W, HR: {s.hr}bpm, cadence: {s.cadence}rpm, mode: {s.mode}"
                + (f", target: {s.power_target}W" if s.power_target else "")
                + "."
            )
            prompt = (
                f"The rider just triggered the '{command}' command. "
                f"Current ride data: {context_str}. "
                f"Give a single short reaction (1 sentence, no more than 10 words). "
                f"Vary your response — don't repeat the same phrase."
            )
            commentary = generate_commentary_raw(prompt)
            if commentary:
                speech_queue.put(commentary)
                last_spoken_at = time.time()
        else:
            speech_queue.put(response)
            last_spoken_at = time.time()

    elif response.startswith("personality:"):
        # Switch personality — let Claude react in character
        new = response.split(":")[1]
        ACTIVE_PERSONALITY = new
        personality_names = {
            "hype": "hype hype-beast", "drill_sergeant": "drill sergeant",
            "british": "dry British", "data": "data-driven nerdy",
            "soviet": "Soviet Russian",
        }
        prompt = (
            f"You just switched to {personality_names.get(new, new)} coach mode. "
            f"Give a single sentence in-character to announce the switch."
        )
        commentary = generate_commentary_raw(prompt)
        speech_queue.put(commentary or f"Switched to {new} mode.")
        last_spoken_at = time.time()

    elif response == "status_report":
        # Generate a live status report via Claude
        s = state
        if s.last_updated == 0:
            speech_queue.put("I don't have any ride data yet -- are you on the bike?")
            return

        pct = int(s.power / s.power_target * 100) if s.power_target > 0 else None
        context_str = (
            f"Mode: {s.mode}. Power: {s.power}W. "
            + (f"Target: {s.power_target}W ({pct}%). " if pct else "")
            + f"HR: {s.hr}bpm. Cadence: {s.cadence}rpm. "
            + (f"Race position: {s.position}. Gap to front: {s.gap_to_front}s. " if s.mode == "race" else "")
            + f"Distance: {round(s.distance, 1)}km. Elapsed: {int(s.elapsed // 60)}min."
        )
        prompt = f"Rider asked for a status update. Here's the data: {context_str}. Give a punchy 2-sentence summary."
        commentary = generate_commentary_raw(prompt)
        if commentary:
            speech_queue.put(commentary)
            last_spoken_at = time.time()

    elif response.startswith("freeform:"):
        # Rider said something we didn't recognise — pass it to Claude as a question
        question = response[9:]
        s = state
        context_str = f"Power: {s.power}W, HR: {s.hr}bpm, cadence: {s.cadence}rpm, mode: {s.mode}."
        prompt = f"The rider just said: '{question}'. Current ride data: {context_str}. Respond helpfully in 1-2 sentences."
        commentary = generate_commentary_raw(prompt)
        if commentary:
            speech_queue.put(commentary)
            last_spoken_at = time.time()

def voice_listener():
    """Continuously listens for voice commands in a background thread."""
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = MIC_ENERGY
    recognizer.dynamic_energy_threshold = False  # prevents wake word clipping
    recognizer.pause_threshold = 0.6

    print("[Voice] Listening for voice commands (wake word: 'zwift')...")

    try:
        mic = sr.Microphone()
    except (OSError, AttributeError) as e:
        print(f"[Voice] No microphone found: {e}")
        print("[Voice] Voice commands disabled. Proactive coaching still active.")
        return

    with mic as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)

        while True:
            try:
                audio = recognizer.listen(source, timeout=None, phrase_time_limit=5)

                # Discard audio captured during or just after TTS playback
                if _tts_active.is_set() or time.time() - _tts_ended_at < TTS_GRACE_SECONDS:
                    continue

                tmp_wav = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                try:
                    tmp_wav.write(audio.get_wav_data())
                    tmp_wav.close()
                    result = mlx_whisper.transcribe(
                        tmp_wav.name,
                        path_or_hf_repo=WHISPER_MODEL,
                        language="en",
                        initial_prompt="Zwift",  # short = less for Whisper to hallucinate
                        condition_on_previous_text=False,
                    )
                    transcript = result["text"].strip()
                finally:
                    os.unlink(tmp_wav.name)

                if not transcript:
                    continue

                # Filter Whisper hallucinations (outputting prompt or silence artifacts)
                _lower = transcript.lower()
                if any(h in _lower for h in ("cycling app", "wake word", "thank you for watching", "thanks for watching")):
                    continue
                if _lower in ("you", "you.", "yeah", "yeah.", "hmm", "hmm."):
                    continue

                print(f"[Voice] Heard: '{transcript}'")

                key, response = match_command(transcript)
                if response:
                    handle_voice_action(key, response)

            except sr.RequestError as e:
                print(f"[Voice] Speech recognition error: {e}")
            except Exception as e:
                print(f"[Voice] Unexpected error: {e}")

# ─────────────────────────────────────────────
# Proactive Trigger Detection
# ─────────────────────────────────────────────

def detect_trigger():
    global last_spoken_at, last_trigger
    now = time.time()

    with state_lock:
        if state.last_updated == 0:
            return None, None
        if now - last_spoken_at < COOLDOWN_SECONDS:
            return None, None
        # Snapshot state under lock so we read a consistent set of values
        s = RideState()
        s.__dict__.update(state.__dict__)

    pct_of_target = (s.power / s.power_target * 100) if s.power_target > 0 else None
    pct_of_ftp    = (s.power / s.ftp * 100) if s.ftp > 0 else None

    # -- WORKOUT --
    if s.mode == "workout":

        if 0 < s.workout_block_remaining <= 10 and last_trigger != "countdown":
            return "countdown", {
                "seconds": int(s.workout_block_remaining),
                "block":   s.workout_block,
                "power":   s.power,
                "target":  s.power_target,
            }

        if pct_of_target and pct_of_target < 88 and s.power_target > 50:
            if last_trigger != "sandbagger":
                return "sandbagger", {
                    "power":  s.power,
                    "target": s.power_target,
                    "deficit":int(s.power_target - s.power),
                    "pct":    int(pct_of_target),
                }

        if pct_of_target and pct_of_target > 112 and s.power_target > 50:
            if last_trigger != "overcooking":
                return "overcooking", {
                    "power":  s.power,
                    "target": s.power_target,
                    "excess": int(s.power - s.power_target),
                }

        if 0 < s.cadence < 70 and s.power > 100 and last_trigger != "low_cadence":
            return "low_cadence", {"cadence": s.cadence, "power": s.power}

        if s.hr > 175 and last_trigger != "hr_red":
            return "hr_red", {"hr": s.hr}

        if pct_of_target and 95 <= pct_of_target <= 108 and s.power_target > 50:
            if now - last_spoken_at > 45:
                return "on_target", {
                    "power":     s.power,
                    "target":    s.power_target,
                    "block":     s.workout_block,
                    "remaining": int(s.workout_block_remaining),
                }

    # -- RACE --
    elif s.mode == "race":

        if s.gap_to_front > 3 and last_trigger != "dropped":
            return "dropped", {
                "gap":    round(s.gap_to_front, 1),
                "power":  s.power,
                "riders": s.riders_nearby,
            }

        if s.draft_watts > 40 and last_trigger != "good_draft":
            return "good_draft", {
                "savings":  int(s.draft_watts),
                "position": s.position,
            }

        if 0 < s.position <= 3 and last_trigger != "podium_position":
            return "podium_position", {
                "position": s.position,
                "power":    s.power,
                "pct_ftp":  int(pct_of_ftp) if pct_of_ftp else "?",
            }

        if pct_of_ftp and pct_of_ftp > 105 and s.gap_to_front < 1:
            if last_trigger != "burning_matches":
                return "burning_matches", {
                    "power":    s.power,
                    "pct_ftp":  int(pct_of_ftp),
                    "position": s.position,
                }

        if pct_of_ftp and pct_of_ftp < 55 and s.gap_to_front < 2 and s.riders_nearby > 3:
            if last_trigger != "coasting":
                return "coasting", {
                    "power":   s.power,
                    "pct_ftp": int(pct_of_ftp),
                }

    return None, None

# ─────────────────────────────────────────────
# Claude Commentary
# ─────────────────────────────────────────────

TRIGGER_PROMPTS = {
    "countdown":       "The current workout interval has {seconds} seconds remaining. Rider is at {power}W, target {target}W. Give a short countdown hype/warning.",
    "sandbagger":      "Rider is sandbagging: only {power}W vs target {target}W ({pct}%, {deficit}W under). Call them out.",
    "overcooking":     "Rider is {excess}W OVER target ({power}W vs {target}W). Warn them to save energy but be positive about their power.",
    "low_cadence":     "Rider cadence is only {cadence}rpm at {power}W — they're grinding. Encourage them to spin up.",
    "hr_red":          "Rider HR is {hr}bpm — very high. Short check-in, keep them focused.",
    "on_target":       "Rider nailing target: {power}W vs {target}W, {remaining}s left in {block} block. Short encouragement.",
    "dropped":         "Rider has been dropped! Gap to group is {gap} seconds. They're at {power}W. URGENT motivation to bridge back.",
    "good_draft":      "Rider is saving {savings} watts in the draft, sitting in position {position}. Tactical encouragement.",
    "podium_position": "Rider is in position {position} in the race at {pct_ftp}% FTP. Hype them to hold or move up.",
    "burning_matches": "Rider burning {pct_ftp}% FTP ({power}W) while sheltered in the bunch. Strategic warning.",
    "coasting":        "Rider only at {pct_ftp}% FTP ({power}W) while sitting in the bunch. Wake them up.",
}

def _call_claude(system_prompt, user_prompt):
    payload = {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 100,
        "system": system_prompt,
        "messages": [{"role": "user", "content": user_prompt}],
    }
    headers = {
        "x-api-key": ANTHROPIC_KEY,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    try:
        r = requests.post(
            "https://api.anthropic.com/v1/messages",
            json=payload, headers=headers, timeout=8
        )
        r.raise_for_status()
        text = r.json()["content"][0]["text"].strip()
        return re.sub(r'\*([^*]+)\*', r'\1', text).strip()
    except Exception as e:
        print(f"[Claude error] {e}")
        return None

def generate_commentary(trigger, context):
    template = TRIGGER_PROMPTS.get(trigger, "Comment on current ride state.")
    try:
        prompt = template.format(**context)
    except KeyError:
        prompt = template
    personality = PERSONALITIES.get(ACTIVE_PERSONALITY, PERSONALITIES["hype"])
    return _call_claude(personality, prompt)

def generate_commentary_raw(prompt):
    """For freeform / status queries — uses active personality."""
    personality = PERSONALITIES.get(ACTIVE_PERSONALITY, PERSONALITIES["hype"])
    return _call_claude(personality, prompt)

# ─────────────────────────────────────────────
# ElevenLabs TTS
# ─────────────────────────────────────────────

def _detect_audio_player():
    """Detect available audio player once at startup."""
    for name, args in [
        ("afplay",  ["afplay"]),
        ("mpg123",  ["mpg123", "-q"]),
        ("ffplay",  ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet"]),
    ]:
        if shutil.which(name):
            return name, args
    return None, None

AUDIO_PLAYER_NAME, AUDIO_PLAYER_ARGS = _detect_audio_player()

# Set while TTS audio is playing so the voice listener ignores its own output
_tts_active = threading.Event()
_tts_ended_at = 0.0  # timestamp when TTS last finished (for grace period)
TTS_GRACE_SECONDS = 1.5  # ignore mic for this long after TTS ends

def speak(text):
    if not _elevenlabs:
        print(f"[COACH] {text}")
        return

    try:
        voice_id = PERSONALITY_VOICES.get(ACTIVE_PERSONALITY, VOICE_ID)
        audio_stream = _elevenlabs.text_to_speech.stream(
            voice_id=voice_id,
            text=text,
            model_id="eleven_turbo_v2_5",
            output_format="mp3_44100_128",
            voice_settings={"stability": 0.4, "similarity_boost": 0.8, "style": 0.6},
        )
        tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        try:
            for chunk in audio_stream:
                tmp.write(chunk)
            tmp.close()
            if AUDIO_PLAYER_ARGS:
                proc = subprocess.Popen(
                    AUDIO_PLAYER_ARGS + [tmp.name],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
                _tts_active.set()
                def _finish(p, f):
                    global _tts_ended_at
                    p.wait()
                    _tts_ended_at = time.time()
                    _tts_active.clear()
                    os.unlink(f)
                threading.Thread(target=_finish, args=(proc, tmp.name), daemon=True).start()
            else:
                print(f"[COACH - no player] {text}")
                os.unlink(tmp.name)
        except Exception:
            tmp.close()
            os.unlink(tmp.name)
            raise
    except Exception as e:
        print(f"[ElevenLabs error] {e}")
        print(f"[COACH] {text}")

def speech_worker():
    while True:
        text = speech_queue.get()
        if text:
            speak(text)
        speech_queue.task_done()

# ─────────────────────────────────────────────
# Main Coaching Loop
# ─────────────────────────────────────────────

def coaching_loop():
    global last_spoken_at, last_trigger
    print("[Coach] Proactive coaching loop started...")

    while True:
        time.sleep(CHECK_INTERVAL)

        if not voice_active:
            continue

        trigger, context = detect_trigger()
        if not trigger:
            continue

        print(f"[Coach] Trigger: {trigger} | {context}")
        commentary = generate_commentary(trigger, context)
        if not commentary:
            continue

        print(f"[Coach] {commentary}")
        speech_queue.put(commentary)
        last_spoken_at = time.time()
        last_trigger   = trigger

# ─────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Zwift AI Coach + Voice Control")
    print("=" * 60)
    print(f"  Personality : {ACTIVE_PERSONALITY.upper()}")
    print(f"  Wake word   : {WAKE_WORDS}")
    print(f"  Sauce WS    : {SAUCE_WS_URL}")
    print(f"  ElevenLabs  : {'configured' if ELEVENLABS_KEY else 'missing (print only)'}")
    print(f"  Anthropic   : {'configured' if ANTHROPIC_KEY else 'missing'}")
    print("=" * 60)
    print()
    print("  Voice commands:")
    print("  \"zwift skip interval\"    -> skip workout block")
    print("  \"zwift harder\"           -> increase intensity")
    print("  \"zwift easier\"           -> decrease intensity")
    print("  \"zwift power up\"         -> use power-up")
    print("  \"zwift how am I doing\"   -> live status report")
    print("  \"zwift be mean\"          -> switch to drill sergeant")
    print("  ...and more. See coach.py for full list.")
    print()

    if not ANTHROPIC_KEY:
        print("ERROR: ANTHROPIC_API_KEY is not set. The coach cannot generate commentary.")
        print("Copy .env.example to .env and add your API key.")
        return

    # Start threads
    threading.Thread(target=speech_worker,  daemon=True).start()
    threading.Thread(target=voice_listener, daemon=True).start()

    # Connect to Sauce for Zwift
    connect_websocket()

    # Give a startup message once Sauce connects (short delay)
    time.sleep(3)
    speech_queue.put("Zwift AI Coach is online. I'm watching. Let's go!")

    # Main coaching loop (blocks)
    coaching_loop()

if __name__ == "__main__":
    import sys
    if "--list-voices" in sys.argv:
        if not _elevenlabs:
            print("ELEVENLABS_API_KEY not set.")
        else:
            voices = _elevenlabs.voices.get_all().voices
            print(f"{'Name':<25} {'Voice ID':<30} Category")
            print("-" * 70)
            for v in sorted(voices, key=lambda v: v.name):
                print(f"{v.name:<25} {v.voice_id:<30} {v.category or ''}")
        sys.exit(0)
    main()
