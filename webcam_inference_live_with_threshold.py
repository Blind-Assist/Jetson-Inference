import time
import torch
import cv2
import numpy as np
import gc
import threading
import sys
from PIL import Image as PILImage
from transformers import AutoModel, AutoTokenizer
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
import pyttsx3

# --- Configuration ---
MERGED_MODEL = "blind-assist/internvl3-1b-merged-v1"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

WEBCAM_INDEX = 0           # change to 1 if /dev/video0 doesn't work
INPUT_SIZE = 448

# --- Change-threshold inference control ---
# This is the thresholding logic from the notebook:
# score = mean absolute grayscale frame difference / 255.
# If score >= CHANGE_THRESHOLD, the scene is considered changed enough.
CHANGE_THRESHOLD = 0.08

# Minimum time gap between two model calls.
# Lower value = more responsive but more GPU usage.
MIN_INFERENCE_INTERVAL = 1.5

# Safety refresh: run inference at least once every N seconds even if the
# change score stays below threshold. Set to 0 to disable forced refresh.
FORCE_INFERENCE_INTERVAL = 10.0

# Resize used only for cheap change detection, not for model input.
CHANGE_DETECT_SIZE = 224

PROMPT = (
    "Given the visual input from the user's forward perspective, identify the closest "
    "immediate obstacle that poses the highest collision risk (especially within approximately "
    "2 meters), and generate exactly one short sentence guiding a visually impaired user by "
    "describing its location using clock directions relative to the user (12 o'clock is straight "
    "ahead), including relevant details such as size, material, or distance, and giving one clear "
    "action to avoid it, prioritizing immediate safety and ignoring less urgent or distant objects, "
    "with no extra explanation."
)

# --- Shared state between threads ---
latest_response = "Waiting for first inference..."
inference_running = False
last_inference_time = 0.0
frame_count = 0

# --- Text-to-Speech (TTS) ---
tts_engine = None


def init_tts():
    """Initialize the TTS engine."""
    global tts_engine
    try:
        tts_engine = pyttsx3.init()
        tts_engine.setProperty('rate', 150)  # Speech rate (slower for clarity)
        tts_engine.setProperty('volume', 0.9)  # Volume (0-1)
        print("✅ TTS engine initialized")
    except Exception as e:
        print(f"⚠️  TTS initialization failed: {e}")
        tts_engine = None


def speak_response(text):
    """Speak the text in a background thread to avoid blocking video."""
    if tts_engine is None:
        return
    
    def _speak():
        try:
            tts_engine.say(text)
            tts_engine.runAndWait()
        except Exception as e:
            print(f"⚠️  TTS error: {e}")
    
    t = threading.Thread(target=_speak, daemon=True)
    t.start()


def build_transform(input_size):
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])


def load_image_for_internvl(image: PILImage.Image, input_size=448):
    transform = build_transform(input_size)
    return transform(image).unsqueeze(0)


def frame_change_score(current_rgb: np.ndarray, reference_rgb: np.ndarray) -> float:
    """
    Returns a value from 0.0 to 1.0 showing how much the current frame differs
    from the reference frame.

    This is the thresholding idea from the notebook:
    1. Convert both frames to grayscale
    2. Resize both to 224x224
    3. Compute mean absolute pixel difference
    4. Normalize by 255
    """
    current_gray = cv2.cvtColor(current_rgb, cv2.COLOR_RGB2GRAY)
    reference_gray = cv2.cvtColor(reference_rgb, cv2.COLOR_RGB2GRAY)

    current_small = cv2.resize(
        current_gray,
        (CHANGE_DETECT_SIZE, CHANGE_DETECT_SIZE),
        interpolation=cv2.INTER_AREA,
    )
    reference_small = cv2.resize(
        reference_gray,
        (CHANGE_DETECT_SIZE, CHANGE_DETECT_SIZE),
        interpolation=cv2.INTER_AREA,
    )

    mad = np.mean(
        np.abs(current_small.astype(np.float32) - reference_small.astype(np.float32))
    )
    return float(mad / 255.0)


def patch_qwen2_to_avoid_float32(model):
    from transformers.models.qwen2 import modeling_qwen2
    original_forward = modeling_qwen2.Qwen2ForCausalLM.forward

    def patched_forward(self, *args, **kwargs):
        output = original_forward(self, *args, **kwargs)
        if hasattr(output, "logits") and output.logits is not None:
            output.logits = output.logits.to(torch.bfloat16)
        return output

    modeling_qwen2.Qwen2ForCausalLM.forward = patched_forward
    print("✅ Patched Qwen2 forward to avoid float32 logit upcast")


def run_inference_thread(model, tokenizer, image: PILImage.Image):
    """Runs inference in a background thread so video stays smooth."""
    global latest_response, inference_running, last_inference_time

    pixel_values = load_image_for_internvl(image, input_size=INPUT_SIZE).to(
        model.device, dtype=torch.bfloat16
    )
    try:
        t_start = time.time()
        with torch.inference_mode():
            response = model.chat(
                tokenizer=tokenizer,
                pixel_values=pixel_values,
                question=PROMPT,
                generation_config=dict(max_new_tokens=64, do_sample=False),
                num_patches_list=[1]
            )
        elapsed = time.time() - t_start
        last_inference_time = elapsed
        latest_response = response
        print(f"\n⏱️  {elapsed:.2f}s")
        print(f"🔊 {response}")
        print("-" * 60)
        # Speak the response
        speak_response(response)
    except RuntimeError as e:
        latest_response = f"Error: {str(e)[:60]}"
        print(f"❌ Inference error: {e}")
        torch.cuda.empty_cache()
        gc.collect()
    finally:
        del pixel_values
        torch.cuda.empty_cache()
        gc.collect()
        inference_running = False


def _wrap_text(text, max_chars=55, max_lines=3):
    words = str(text).split()
    lines = []
    current = ""

    for word in words:
        if len(current) + len(word) + 1 <= max_chars:
            current += (" " if current else "") + word
        else:
            lines.append(current)
            current = word

    if current:
        lines.append(current)

    return lines[:max_lines]


def draw_overlay(frame, response_text, status_text, is_inferencing, frame_num):
    """Draw the response text and status overlay on the frame."""
    h, w = frame.shape[:2]

    # --- Dark semi-transparent bar at bottom for guidance text ---
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h - 135), (w, h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    for i, line in enumerate(_wrap_text(response_text, max_chars=55, max_lines=3)):
        cv2.putText(
            frame,
            line,
            (10, h - 112 + i * 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 180),
            1,
            cv2.LINE_AA,
        )

    # --- Top status bar ---
    overlay2 = frame.copy()
    cv2.rectangle(overlay2, (0, 0), (w, 58), (0, 0, 0), -1)
    cv2.addWeighted(overlay2, 0.5, frame, 0.5, 0, frame)

    cv2.putText(
        frame,
        f"Frame #{frame_num}",
        (10, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        "BLIND ASSIST",
        (w // 2 - 70, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    if is_inferencing:
        top_status = "Inferencing..."
        status_color = (0, 200, 255)
    else:
        top_status = status_text
        status_color = (100, 255, 100)

    cv2.putText(
        frame,
        top_status[:90],
        (10, 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        status_color,
        1,
        cv2.LINE_AA,
    )

    return frame


def main():
    global latest_response, inference_running, last_inference_time, frame_count

    patch_qwen2_to_avoid_float32(None)
    init_tts()

    print(f"🔄 Loading model: {MERGED_MODEL}")
    model = AutoModel.from_pretrained(
        MERGED_MODEL,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map={"": "cuda:0"},
        low_cpu_mem_usage=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        MERGED_MODEL, trust_remote_code=True, use_fast=False
    )
    model.eval()
    print("✅ Model loaded!")

    print(f"📷 Opening webcam at index {WEBCAM_INDEX}...")
    cap = cv2.VideoCapture(WEBCAM_INDEX)

    if not cap.isOpened():
        print(f"❌ Cannot open webcam at index {WEBCAM_INDEX}. Try WEBCAM_INDEX = 1")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    print("✅ Webcam opened!")
    print(
        "🚀 Live threshold mode enabled.\n"
        f"   CHANGE_THRESHOLD={CHANGE_THRESHOLD}\n"
        f"   MIN_INFERENCE_INTERVAL={MIN_INFERENCE_INTERVAL}s\n"
        f"   FORCE_INFERENCE_INTERVAL={FORCE_INFERENCE_INTERVAL}s\n"
        "   Press Q or ESC to quit.\n"
    )

    # Reference frame is the frame used for the last inference.
    # Comparing against this instead of the immediately previous video frame makes
    # slow cumulative scene changes still trigger a new instruction.
    reference_frame_rgb = None
    last_inference_start_ts = 0.0
    status_text = "Waiting for first frame..."

    window_name = "Blind Assist - Live Feed"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 800, 600)

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("⚠️  Frame read failed, retrying...")
            time.sleep(0.1)
            continue

        frame_count += 1
        now = time.time()

        # Convert frame to RGB once. OpenCV display still uses the original BGR frame.
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if reference_frame_rgb is None:
            score = 1.0
            changed = True
        else:
            score = frame_change_score(frame_rgb, reference_frame_rgb)
            changed = score >= float(CHANGE_THRESHOLD)

        seconds_since_last_inference = (
            float("inf") if last_inference_start_ts == 0.0 else now - last_inference_start_ts
        )
        interval_ok = seconds_since_last_inference >= float(MIN_INFERENCE_INTERVAL)
        force_due = (
            FORCE_INFERENCE_INTERVAL > 0
            and seconds_since_last_inference >= float(FORCE_INFERENCE_INTERVAL)
        )

        was_first_frame = reference_frame_rgb is None
        should_run = (
            not inference_running
            and interval_ok
            and (was_first_frame or changed or force_due)
        )

        if should_run:
            inference_running = True
            last_inference_start_ts = now
            reference_frame_rgb = frame_rgb.copy()

            pil_image = PILImage.fromarray(frame_rgb)

            if was_first_frame:
                reason = "first frame"
            elif force_due and not changed:
                reason = "safety refresh"
            else:
                reason = "change detected"

            timestamp = time.strftime("%H:%M:%S")
            print(
                f"[{timestamp}] 📸 Frame #{frame_count} — running inference "
                f"({reason}, score={score:.4f}, threshold={CHANGE_THRESHOLD:.4f})"
            )

            t = threading.Thread(
                target=run_inference_thread,
                args=(model, tokenizer, pil_image),
                daemon=True
            )
            t.start()

            status_text = (
                f"score={score:.4f} | threshold={CHANGE_THRESHOLD:.4f} | "
                f"changed={changed} | mode=inference"
            )
        else:
            if inference_running:
                status_text = (
                    f"score={score:.4f} | threshold={CHANGE_THRESHOLD:.4f} | "
                    "mode=inferencing"
                )
            elif not interval_ok:
                wait_left = max(0.0, MIN_INFERENCE_INTERVAL - seconds_since_last_inference)
                status_text = (
                    f"score={score:.4f} | threshold={CHANGE_THRESHOLD:.4f} | "
                    f"changed={changed} | mode=cooldown {wait_left:.1f}s"
                )
            elif changed:
                # This usually only appears very briefly before the thread starts.
                status_text = (
                    f"score={score:.4f} | threshold={CHANGE_THRESHOLD:.4f} | "
                    "changed=True | mode=ready"
                )
            else:
                status_text = (
                    f"score={score:.4f} | threshold={CHANGE_THRESHOLD:.4f} | "
                    "changed=False | mode=stable"
                )

        display_frame = draw_overlay(
            frame.copy(),
            latest_response,
            status_text,
            inference_running,
            frame_count
        )

        cv2.imshow(window_name, display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:  # Q or ESC
            print("\n🛑 Stopping...")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
