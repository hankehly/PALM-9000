# PALM-9000
PALM-9000 is a Raspberry Pi and LLM–powered talking palm tree—ever-watchful, eerily articulate, and not entirely sure it should let you prune that branch.

# Raspberry Pi Configuration

![Raspberry Pi Zero 2W GPIO Pinout](images/Raspberry-Pi-Zero-2W-GPIO-Pinout.png)

## Connect the OTG USB Cable and USB Audio Adapter

## Connect the INMP441 Microphone

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
arecord -l
```

Record a test sample
```sh
# Replace 1 with the card number from the previous command
arecord -D plughw:1,0 -f cd -c 1 -r 44100 -d 5 test.wav
```

Play back the test sample.
```sh
# Replace the first 0 with the connected speaker device number
aplay -D plughw:0,0 test.wav
```

## Enable Acoustic Echo Cancellation (AEC)

Without Acoustic Echo Cancellation (AEC), speaker output may be picked up as microphone input, causing a feedback loop where the agent hears its own output and responds to itself.

Raspberry PI's sound server PulseAudio can be configured to use Acoustic Echo Cancellation (AEC) to reduce echo from the speaker when using a microphone. The following setup assumes you are connecting to a headless 64-bit Raspberry Pi from macOS.

On macOS, install an X server (e.g., XQuartz).
```sh
brew install --cask xquartz
```

After installing, log out and log back in (or reboot) so `$DISPLAY` is set correctly on macOS.
```sh
echo $DISPLAY
```

You should see something like this:
```sh
/private/tmp/com.apple.launchd.N7l0ZjE86u/org.xquartz:0
```

Login to the raspberry pi with X11 forwarding enabled. This will allow you to see the GUI from your Raspberry Pi on your macOS.
```sh
ssh -Y <username>@<raspberry_pi_ip>
```

Now inside the Raspberry Pi, install PulseAudio and its utilities. PulseAudio is probably already installed on the Raspberry Pi, but you may need to install pavucontrol.
```sh
sudo apt update
sudo apt install pulseaudio pulseaudio-utils pavucontrol
```

Manually enable AEC.
```sh
pactl load-module module-echo-cancel \
    source_name=echosource sink_name=echosink \
    aec_method=webrtc
```

Make sure the AEC devices exist. You should see something like `echosource` and `echosink`.
```sh
pactl list short sources
pactl list short sinks
```

Run pavucontrol on the Raspberry Pi.
```sh
pavucontrol
```

You should see a window like this appear on your macOS. Go to the Output Devices and Input Devices tabs and click the green icon next to the option showing "echo cancelled" in the title.

![Pavucontrol](images/pavucontrol.png)

Now go back to the Raspberry Pi and confirm it's working.
```sh
pactl info | grep "Default Sink"
pactl info | grep "Default Source"
```

You should see:
```sh
Default Sink: echosink
Default Source: echosource
```

### Set as default (optional)

To set AEC as the default, edit the PulseAudio configuration file.
```sh
sudo vi /etc/pulse/default.pa
```

Add at the end:
```sh
load-module module-echo-cancel source_name=echosource sink_name=echosink aec_method=webrtc
set-default-source echosource
set-default-sink echosink
```

Save, then restart PulseAudio:
```sh
pulseaudio -k
pulseaudio --start
```

### Test

To test that AEC is working, we'll record our voice while playing audio through the speaker. If AEC is working correctly, we should only hear our voice in the recording, while the audio from the speaker is removed.

First, prepare a sample audio file to play through the speaker.
```sh
wget https://download.samplelib.com/wav/sample-3s.wav
```

Record from the AEC source via Pulse. (Note that the `-f cd` option is shorthand for recording 16 bit little endian, 44100 Hz, stereo quality.)
```sh
PULSE_SOURCE=echosource arecord -D pulse -f cd -d 10 test.wav
```

While you're recording, play the sample audio file in a separate terminal and also speak into the microphone.
```sh
# Flow:
#  1. Say: "Test 1, 2, 3"
#  2. Play test audio
aplay sample-3s.wav
#  3. Say: "Test 4, 5, 6"
```

Now play the recording.
```sh
PULSE_SINK=echosink aplay -D pulse test.wav
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

# Future Work

- [ ] Moisture sensor for health monitoring
- [ ] Sunlight sensor for optimal placement
- [ ] YouTube video
- [ ] Deploy to the cloud for remote access
- [ ] Integrate with ChatGPT, add access to metrics via custom API
