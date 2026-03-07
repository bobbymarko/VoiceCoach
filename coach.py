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
import speech_recognition as sr
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
VOICE_ID         = os.getenv("ELEVENLABS_VOICE_ID", "EXAVITQu4vr4xnSDxMaL")

WAKE_WORD        = "zwift"       # say this before your command
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
}

ACTIVE_PERSONALITY = "hype"

# ─────────────────────────────────────────────
# Voice Commands → Zwift Keyboard Actions
# ─────────────────────────────────────────────

# Each entry: list of trigger phrases → (key_to_press, confirmation_message)
# key_to_press can be a Key enum, a string char, or None (for meta-commands)

VOICE_COMMANDS = [
    # Workout control
    (["skip interval", "skip", "next interval", "skip block"],
        Key.tab,        "Skipping interval!"),

    (["harder", "push", "more power", "increase", "pump it up"],
        Key.page_up,    "Turning it up!"),

    (["easier", "back off", "back it off", "too hard", "reduce", "less"],
        Key.page_down,  "Dialling it back."),

    # In-game actions
    (["power up", "use power up", "powerup"],
        Key.space,      "Power up deployed!"),

    (["screenshot", "take a photo", "take a picture", "snap"],
        Key.f10,        "Cheese!"),

    (["ride on", "give ride on", "thumbs up"],
        Key.f3,         "Ride on sent!"),

    (["wave", "say hi"],
        Key.f2,         "Waving!"),

    (["u turn", "turn around", "go back", "reverse"],
        Key.down,       "Turning around."),

    # Meta commands — key=None means handled in code, not keypress
    (["how am i doing", "status", "give me a status", "what's my status", "update"],
        None,           "status_report"),

    (["coach hype", "be hype", "hype mode", "change to hype"],
        None,           "personality:hype"),

    (["coach drill sergeant", "drill sergeant", "change to drill sergeant", "be mean"],
        None,           "personality:drill_sergeant"),

    (["coach british", "be british", "change to british", "disappoint me"],
        None,           "personality:british"),

    (["coach data", "data mode", "be nerdy", "change to data"],
        None,           "personality:data"),
]

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

def match_command(transcript):
    """Return (key, response) for the best matching command, or (None, None)."""
    t = transcript.lower().strip()

    # Must start with wake word (or be very close to it)
    if WAKE_WORD not in t:
        return None, None

    # Strip wake word and everything before it
    after_wake = t[t.index(WAKE_WORD) + len(WAKE_WORD):].strip()
    print(f"[Voice] Heard after wake word: '{after_wake}'")

    for phrases, key, response in VOICE_COMMANDS:
        for phrase in phrases:
            if phrase in after_wake or after_wake in phrase:
                return key, response

    # No match — ask Claude to interpret it as a freeform question
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
        speech_queue.put(response)
        last_spoken_at = time.time()

    elif response.startswith("personality:"):
        # Switch personality
        new = response.split(":")[1]
        ACTIVE_PERSONALITY = new
        confirmations = {
            "hype":            "LET'S GOOOOO! Hype mode ACTIVATED, baby!",
            "drill_sergeant":  "Switching to drill sergeant. Don't embarrass yourself.",
            "british":         "Very well. I shall endeavour to contain my disappointment.",
            "data":            "Personality switch confirmed. Optimising motivational output.",
        }
        speech_queue.put(confirmations.get(new, f"Switched to {new} mode."))
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
    recognizer.dynamic_energy_threshold = True
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
                transcript = recognizer.recognize_google(audio)
                print(f"[Voice] Heard: '{transcript}'")

                key, response = match_command(transcript)
                if response:
                    handle_voice_action(key, response)

            except sr.UnknownValueError:
                pass   # didn't catch anything intelligible
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

def speak(text):
    if not ELEVENLABS_KEY:
        print(f"[COACH] {text}")
        return

    url     = f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}/stream"
    headers = {"xi-api-key": ELEVENLABS_KEY, "Content-Type": "application/json"}
    payload = {
        "text": text,
        "model_id": "eleven_turbo_v2",
        "voice_settings": {"stability": 0.4, "similarity_boost": 0.8, "style": 0.6},
    }
    try:
        r = requests.post(url, json=payload, headers=headers, stream=True, timeout=10)
        r.raise_for_status()
        tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        try:
            for chunk in r.iter_content(chunk_size=4096):
                tmp.write(chunk)
            tmp.close()
            if AUDIO_PLAYER_ARGS:
                proc = subprocess.Popen(
                    AUDIO_PLAYER_ARGS + [tmp.name],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
                # Wait for playback to finish, then clean up the temp file
                threading.Thread(
                    target=lambda p, f: (p.wait(), os.unlink(f)),
                    args=(proc, tmp.name),
                    daemon=True,
                ).start()
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
    print(f"  Wake word   : \"{WAKE_WORD}\"")
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
    main()
