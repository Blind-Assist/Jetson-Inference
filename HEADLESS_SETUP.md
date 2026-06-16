# Blind Assist - Headless Setup Guide

This guide shows how to run Blind Assist as a headless service on your Jetson Orin Nano without a monitor.

## Prerequisites
- Jetson Orin Nano with JetPack OS
- USB webcam connected
- Bluetooth headset (or USB speakers) paired and set as default audio device
- SSH access to your Jetson (optional but recommended)

## Installation Steps

### 1. Install Dependencies
```bash
sudo apt-get update
sudo apt-get install -y python3-pip python3-dev
pip install pyttsx3 torch torchvision transformers pillow opencv-python
```

### 2. Clone Your Project
```bash
cd /home/jetson/Projects
git clone <your-repo-url> Jetson-Inference
cd Jetson-Inference
```

### 3. Configure Bluetooth Audio (if using Bluetooth headset)
```bash
# Pair your Bluetooth headset
bluetoothctl
# Then in bluetoothctl prompt:
# scan on
# pair <MAC_ADDRESS>
# trust <MAC_ADDRESS>
# connect <MAC_ADDRESS>
# quit

# Set as default audio output
pactl list short sinks
pactl set-default-sink <bluetooth_device_name>

# Test audio
speaker-test -t sine -f 1000 -l 1
```

### 4. Test the Script Manually
```bash
cd /home/jetson/Projects/Jetson-Inference
python3 webcam_inference_live_with_threshold.py
```
You should hear audio guidance through your headset/speakers when obstacles are detected.

### 5. Create Systemd Service (Auto-start on Boot)

Copy the service file to systemd:
```bash
sudo cp /home/jetson/Projects/Jetson-Inference/blind-assist.service /etc/systemd/system/
```

Enable and start the service:
```bash
sudo systemctl daemon-reload
sudo systemctl enable blind-assist
sudo systemctl start blind-assist
```

### 6. Monitor Service Status
```bash
# Check if running
sudo systemctl status blind-assist

# View logs
sudo journalctl -u blind-assist -f

# Stop service
sudo systemctl stop blind-assist
```

## Configuration

Edit `webcam_inference_live_with_threshold.py` to adjust:

```python
HEADLESS_MODE = True              # Set to False to enable video display (requires monitor)
WEBCAM_INDEX = 0                  # Change to 1 if camera not detected
CHANGE_THRESHOLD = 0.08           # Lower = more sensitive, Higher = less sensitive
MIN_INFERENCE_INTERVAL = 1.5      # Seconds between inferences (GPU efficiency)
FORCE_INFERENCE_INTERVAL = 10.0   # Force check every N seconds
```

## Troubleshooting

### Audio not playing
1. Check Bluetooth connection:
   ```bash
   pactl list short sinks
   pactl get-default-sink
   ```
2. Test audio directly:
   ```bash
   speaker-test -t sine -f 1000 -l 1
   ```
3. Check pyttsx3 logs in service:
   ```bash
   sudo journalctl -u blind-assist -f
   ```

### Webcam not detected
1. List available cameras:
   ```bash
   ls -la /dev/video*
   ```
2. Change `WEBCAM_INDEX` in the script (try 0, 1, 2, etc.)

### Service won't start
1. Check permissions:
   ```bash
   sudo chown -R jetson:jetson /home/jetson/Projects/Jetson-Inference
   ```
2. Run manually to see errors:
   ```bash
   python3 /home/jetson/Projects/Jetson-Inference/webcam_inference_live_with_threshold.py
   ```

### High CPU/GPU usage
- Increase `MIN_INFERENCE_INTERVAL` (less frequent checks)
- Increase `CHANGE_THRESHOLD` (only run on significant changes)

## Running Commands Without SSH

If you want to control the service remotely:
```bash
# SSH into Jetson
ssh jetson@<jetson-ip>

# Control service
sudo systemctl stop blind-assist
sudo systemctl start blind-assist
sudo journalctl -u blind-assist -f
```

## Notes
- The service runs as the `jetson` user by default
- Logs are written to journalctl (system logs)
- The service auto-restarts if it crashes
- No monitor or display is required in HEADLESS_MODE
