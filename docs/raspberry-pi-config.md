# Raspberry Pi system configuration

Everything below lives **outside** the repository — in `/boot/firmware/`,
`/etc/pulse/`, and the `pi` user's groups — so nothing here is restored by a
`git clone`. This is the record of what a working PALM-9000 host actually
has set, captured from the running device rather than written from memory.

`README.md` is the step-by-step build guide, with the wiring and the
reasoning. This file is the reference: what the settings currently *are*, and
why the non-obvious ones are the way they are.

## Verified on

| | |
| --- | --- |
| Hardware | Raspberry Pi Zero 2W (aarch64, 416 MB RAM) |
| OS | Debian GNU/Linux 12 (bookworm) |
| System Python | 3.11.2 (the app runs 3.12 from its own `uv` venv) |
| PulseAudio | 16.1 |

## `/boot/firmware/config.txt`

```
dtparam=i2s=on
dtparam=spi=on
dtparam=audio=on
dtoverlay=googlevoicehat-soundcard
```

`dtparam=spi=on` is for the MAX7219 LED matrix on SPI0/CE0.
`dtparam=i2s=on` plus the `googlevoicehat-soundcard` overlay is what makes
the INMP441 I2S microphone appear as an ALSA capture device — the overlay is
reused for a plain I2S mic, which is why the card is named "Google voiceHAT"
in logs despite no such HAT being present.

The remaining entries (`camera_auto_detect`, `vc4-kms-v3d`, `arm_boost`, the
`[cm4]`/`[cm5]` blocks) are Raspberry Pi OS defaults, untouched.

## Group membership for the `pi` user

```
adm dialout cdrom sudo audio video plugdev games users
input render netdev spi i2c gpio
```

`spi` is required to open `/dev/spidev0.0` without root, `gpio` for the
ADC0834 bit-banging, `audio` for PulseAudio. The rest are distribution
defaults.

## `/etc/pulse/default.pa` — acoustic echo cancellation

Without this the microphone hears the speaker and the bot talks to itself.
Line 144 of the stock file:

```
load-module module-echo-cancel \
  source_name=echosource \
  sink_name=echosink \
  source_master=alsa_input.platform-soc_sound.stereo-fallback \
  sink_master=alsa_output.usb-GeneralPlus_USB_Audio_Device-00.analog-stereo \
  use_master_format=1 \
  aec_method=webrtc \
  aec_args="analog_gain_control=0 digital_gain_control=1 extended_filter=1 noise_suppression=1 drift_compensation=1"
```

(One line in the real file; wrapped here for readability.)

The `source_master` / `sink_master` names are device-specific — derive yours
from `pactl list short sources` and `pactl list short sinks` rather than
copying these.

### Why `drift_compensation=1`

**Added 2026-09-21.** The capture and playback devices are on **different
hardware clocks**: the microphone is I2S off the SoC
(`platform-soc_sound`), the speaker is a USB audio adapter
(`usb-GeneralPlus_USB_Audio_Device`). Echo cancellation works by subtracting
what was played from what was heard, which requires the two streams to stay
time-aligned. Two independent clocks drift apart, the alignment slips, and
the canceller progressively stops removing the echo.

This is observable. Running PulseAudio in the foreground prints:

```
module-echo-cancel.c: Doing resync
module-echo-cancel.c: Playback too far ahead (23724), drop source 9104
```

The symptom at the application layer is the bot interrupting itself:
residual echo reaches Gemini, whose server-side VAD reads it as the user
starting to speak and cuts the reply off mid-sentence.

Keep it in mind if you change either audio device. Putting the microphone
and speaker on the *same* clock domain — for example both on the USB
adapter, or both on the HAT — removes the drift at the source and is a
better fix than compensating for it.

## Applying and verifying a PulseAudio change

PulseAudio re-reads `default.pa` only on start, and **a runtime
`pactl unload-module` is not enough to test a change**: with `autospawn =
yes` (the default in `/etc/pulse/client.conf`), unloading the module can
leave the daemon with no clients, whereupon it exits and immediately
respawns from `default.pa` — silently discarding the runtime edit. Edit the
file, then restart:

```sh
sudo cp /etc/pulse/default.pa /etc/pulse/default.pa.bak-$(date +%Y%m%d)
sudo nano /etc/pulse/default.pa            # edit the aec_args line
pulseaudio --kill
pulseaudio --daemonize=yes
```

Verify the change actually took, rather than assuming:

```sh
pactl list modules | grep -o 'aec_args="[^"]*"'   # shows the live args
pactl list short sources                          # echosource must be present
```

If `echosource` is missing the app will fall back to the raw ALSA input and
the bot will hear itself. `pactl list source-outputs` shows which source the
app actually captures from once it is running — confirm it is `echosource`,
not the raw device.

To see startup errors, run it in the foreground:

```sh
pulseaudio --daemonize=no --log-target=stderr
```

A `module-alsa-card` failure for `platform-3f902000.hdmi` is expected noise
when no HDMI display is attached, and is unrelated.

## Deploying the application

The repository is public and the Pi's remote is anonymous HTTPS, so the
device pulls without credentials:

```sh
ssh raspberrypi-zero2w.local
export PATH="$HOME/.local/bin:$PATH"   # uv is not on a non-interactive PATH
cd ~/Projects/PALM-9000
git pull
uv sync --no-dev
```

`.env` is gitignored and lives only on the device, with its own
`INPUT_DEVICE` — never copy it between machines.
