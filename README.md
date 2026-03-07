# 🚴 Zwift AI Coach + Voice Control

Real-time dynamic audio coaching AND hands-free voice control for Zwift.
Powered by Claude AI + ElevenLabs TTS + Google Speech Recognition.
Hooks into Sauce for Zwift’s WebSocket API — perfectly in sync with pauses,
skipped intervals, and everything else.

-----

## Voice Commands

Say **“zwift”** followed by your command. That’s it.

|Say…                       |Does…                               |
|---------------------------|------------------------------------|
|“zwift skip interval”      |Skips current workout block (Tab)   |
|“zwift harder” / “push”    |Increases workout intensity (PgUp)  |
|“zwift easier” / “back off”|Decreases workout intensity (PgDown)|
|“zwift power up”           |Uses your power-up (Spacebar)       |
|“zwift screenshot”         |Takes a screenshot (F10)            |
|“zwift ride on”            |Sends a Ride On (F3)                |
|“zwift wave”               |Waves at other riders (F2)          |
|“zwift u turn”             |Reverses direction (↓)              |
|“zwift how am I doing”     |Live status report from the coach   |
|“zwift be mean”            |Switches to Drill Sergeant          |
|“zwift be british”         |Switches to Disappointed British    |
|“zwift hype mode”          |Switches to Hype Coach              |
|“zwift data mode”          |Switches to Data Nerd               |
|“zwift [anything else]”    |Claude tries to answer it!          |

The last one is the fun part — “zwift should I attack here?” or “zwift how far
to the finish?” will get an AI response based on your live ride data.

-----

## What the coach proactively says

**In workouts:**

- Calls you out when you’re sandbagging
- Warns you if you’re overcooking it
- Countdown shouts in the final 10 seconds of hard intervals
- Cadence reminders if you’re grinding
- Heart rate check-ins at very high effort

**In races:**

- URGENT alerts when you get dropped
- Tactical reminders when you’re burning matches
- Hype when you’re in podium position
- Draft efficiency tips
- Coasting alerts

-----

## Setup

### 1. Prerequisites

- [Sauce for Zwift](https://www.sauce.llc/products/sauce4zwift/) installed and running
- Python 3.9+
- Microphone connected
- [Anthropic API key](https://console.anthropic.com)
- [ElevenLabs API key](https://elevenlabs.io)

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

**macOS** — pyaudio needs portaudio first:

```bash
brew install portaudio
pip install -r requirements.txt
```

**Linux:**

```bash
sudo apt install portaudio19-dev python3-pyaudio
pip install -r requirements.txt
```

**Windows:** pyaudio has a pre-built wheel:

```bash
pip install pipwin
pipwin install pyaudio
pip install -r requirements.txt
```

### 3. Configure

```bash
cp .env.example .env
# Edit .env with your API keys
```

### 4. Run

```bash
python coach.py
```

Start Zwift, then start a ride. The coach announces itself when ready.

-----

## Tuning

**Coach too chatty?** Increase `COOLDOWN_SECONDS` in coach.py (default: 12).

**Voice not hearing you?** Adjust `MIC_ENERGY` — lower = more sensitive (default: 300).
Too many false triggers? Raise it to 400-500.

**Change wake word?** Edit `WAKE_WORD = "zwift"` — pick anything short and distinct.

**Add a command?** Add a new entry to `VOICE_COMMANDS`:

```python
(["my phrase", "alternate phrase"],
    Key.f5,   "Confirmation message"),
```

-----

## How it all fits together

```
┌─────────────────────────────────────────────────────┐
│                    coach.py                         │
│                                                     │
│  Sauce WS  ──► RideState (power/HR/position/...)   │
│                    │                                │
│              detect_trigger()                       │
│                    │                                │
│              Claude AI ──► ElevenLabs TTS ──► 🔊   │
│                                                     │
│  Microphone ──► SpeechRecognition                  │
│                    │                                │
│              match_command()                        │
│                 │          │                        │
│            Keypress    Claude AI ──► ElevenLabs TTS │
│           (→ Zwift)                       ──► 🔊   │
└─────────────────────────────────────────────────────┘
```

-----

## Audio playback

- **macOS**: uses `afplay` (built-in)
- **Linux**: install `mpg123` (`apt install mpg123`) or `ffmpeg`
- **Windows**: install `ffmpeg`, ensure `ffplay` is in PATH

## Cost estimate

- Claude: ~$0.001 per coaching line
- ElevenLabs: free tier = 10,000 chars/month (~500 lines)
- Speech recognition: free (Google STT via SpeechRecognition library)
