# PALM-9000
PALM-9000 is a Raspberry Pi and LLM–powered talking palm tree—ever-watchful, eerily articulate, and not entirely sure it should let you prune that branch.

# Raspberry Pi Setup

![Raspberry Pi Zero 2W GPIO Pinout](images/Raspberry-Pi-Zero-2W-GPIO-Pinout.png)

## 1. Install PortAudio for Audio I/O

```sh
sudo apt update
sudo apt install -y portaudio19-dev
```

## 2. Enable SPI

Edit `/boot/firmware/config.txt` and add (if not already):
```sh
dtparam=spi=on
```

Reboot.
```sh
sudo reboot
```

Verify that the device exists.
```sh
# You should see something like /dev/spidev0.0 and /dev/spidev0.1
ls /dev/spi*
```

## 3. Connect the OTG USB Cable and USB Audio Adapter

Hook a USB OTG cable to the Pi Zero's USB port, then connect a USB audio adapter to the OTG cable. For this project, I used a [UGREEN 10396](https://www.amazon.co.jp/dp/B00LN3LQKQ) and [ALLVD B0CC519BSM](https://www.amazon.co.jp/dp/B0CC519BSM).

Find the card/device numbers for the speaker.
```sh
aplay -l
```

Do a quick test to confirm sound output. The `-D plughw:1,0` option specifies card 1, device 0; replace with your connected device numbers.
```sh
speaker-test -c 2 -t wav -l 1 -D plughw:1,0
```

## 4. Connect the INMP441 Microphone

![GPIO INMP441 Pinout Diagram](images/GPIO-INMP441-Pinout-Diagram.png)

Edit `/boot/firmware/config.txt` (add these if not present):
```
dtparam=i2s=on
dtoverlay=googlevoicehat-soundcard
```

Reboot.
```sh
sudo reboot
```

Confirm ALSA sees the mic.
```sh
# You should see something like "Google voiceHAT SoundCard HiFi"
# Mark the card number and device number for later use
arecord -l
```

Record a test sample
```sh
# Replace 1 with the card number from the previous command
arecord -D plughw:0,0 -f cd -c 1 -r 44100 -d 5 test.wav
```

Play back the test sample.
```sh
# Replace the first 0 with the connected speaker device number
aplay -D plughw:1,0 test.wav
```

## 5. Enable Acoustic Echo Cancellation (AEC)

Without Acoustic Echo Cancellation (AEC), speaker output may be picked up as microphone input, causing a feedback loop where the agent hears its own output and responds to itself. Raspberry Pi's sound server PulseAudio can be configured to use AEC to reduce echo from the speaker when using a microphone.

Install PulseAudio and its utilities.
```sh
sudo apt update
sudo apt install pulseaudio pulseaudio-utils
```

Now we need to identify the device names for the mic and speaker so that we can configure PulseAudio to use them for AEC; otherwise, PulseAudio might default to the wrong devices. Run the following commands to list the available sources (microphones) and sinks (speakers).
```sh
pactl list short sources  # E.g., alsa_input.platform-soc_sound.stereo-fallback
pactl list short sinks    # E.g., alsa_output.usb-GeneralPlus_USB_Audio_Device-00.analog-stereo
```

Edit the PulseAudio configuration file (`/etc/pulse/default.pa`) to load the echo cancel module on startup. Specify the correct source_master and sink_master based on the previous step.
```sh
load-module module-echo-cancel source_name=echosource sink_name=echosink source_master=alsa_input.platform-soc_sound.stereo-fallback sink_master=alsa_output.usb-GeneralPlus_USB_Audio_Device-00.analog-stereo use_master_format=1 aec_method=webrtc aec_args="analog_gain_control=0 digital_gain_control=1 extended_filter=1 noise_suppression=1"
set-default-source echosource
set-default-sink echosink
```

Restart PulseAudio to apply the changes.
```sh
systemctl --user restart pulseaudio
```

Confirm it's working. You should see "echosink" and "echosource" as the default sink and source.
```sh
pactl info | grep "Default Sink"
pactl info | grep "Default Source"
```

To test that AEC is working, we'll record our voice while playing audio through the speaker. If AEC is working correctly, we should only hear our voice in the recording, while the audio from the speaker is removed.

First, prepare a sample audio file to play through the speaker.
```sh
wget https://download.samplelib.com/wav/sample-3s.wav
```

Record from the AEC source via Pulse. While you're recording, play the sample audio file in a separate terminal and also speak into the microphone. Confirm card/device numbers in previous steps otherwise this may not work. Note that the `-f cd` option is shorthand for recording 16 bit little endian, 44100 Hz, stereo quality.
```sh
# 1. Start recording
arecord -D pulse -f cd -d 10 test.wav
# 2. Say: "1, 2, 3"
# 3. In another terminal window, play test audio through speakers
aplay -D pulse sample-3s.wav
# 4. Say: "4, 5, 6"
```

Now play the recording.
```sh
aplay -D pulse test.wav
```

The recording should contain your voice (mic) but little to none of the sample audio being played from the speaker.

## (Deprecated) Enable WM8960 HAT Interfaces & Driver

Edit `/boot/firmware/config.txt` (add these if not present):
```sh
dtparam=i2s=on
dtparam=i2c_arm=on
dtoverlay=wm8960-soundcard
```

Also edit /etc/modules and add (if not already):
```sh
i2c-dev
```

Reboot.
```sh
sudo reboot
```

Confirm HAT is detected.
```sh
# I2C control plane (WM8960 is usually at 0x1a)
sudo apt-get update -y && sudo apt-get install -y i2c-tools alsa-utils
sudo i2cdetect -y 1

# ALSA sees the sound card? (should see wm8960-soundcard)
aplay -l
```

Unmute outputs & route PCM.
```sh
amixer -c 0 sset 'Headphone' 80% unmute
amixer -c 0 sset 'Playback' 80% unmute
amixer -c 0 sset 'Speaker' 80% unmute
amixer -c 0 sset 'Speaker Playback ZC' on
amixer -c 0 sset 'Mono Out' 80% unmute
amixer -c 0 sset 'Left Output Mixer PCM' on
amixer -c 0 sset 'Right Output Mixer PCM' on
```

Plugin a speaker or headphones and play a quick test:
```sh
speaker-test -c 2 -t wav -l 1
```

Save your mixer state so it persists across reboots.
```sh
sudo alsactl store
```

# Run

```sh
PULSE_LATENCY_MSEC=60 uv run --no-dev main.py
```

# Future Work

- [ ] Moisture sensor for health monitoring
- [ ] Sunlight sensor for optimal placement
- [ ] YouTube video
- [ ] Deploy to the cloud for remote access
- [ ] Integrate with ChatGPT, add access to metrics via custom API
