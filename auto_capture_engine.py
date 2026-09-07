import cv2
import numpy as np
import time
from pathlib import Path

def gray_world_normalize(img):
    img_float = img.astype(np.float32)
    mean_b, mean_g, mean_r = np.mean(img_float[:, :, 0]), np.mean(img_float[:, :, 1]), np.mean(img_float[:, :, 2])
    mean_gray = (mean_b + mean_g + mean_r) / 3.0 + 1e-6
    img_float[:, :, 0] = np.clip(img_float[:, :, 0] * (mean_gray / (mean_b + 1e-6)), 0, 255)
    img_float[:, :, 1] = np.clip(img_float[:, :, 1] * (mean_gray / (mean_g + 1e-6)), 0, 255)
    img_float[:, :, 2] = np.clip(img_float[:, :, 2] * (mean_gray / (mean_r + 1e-6)), 0, 255)
    return img_float.astype(np.uint8)

def run_autocapture_stream():
    # Initialize default video feed (webcam / USB camera)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open camera device.")
        return

    print("\n" + "="*50)
    print(" VITALIS AUTO-CAPTURE ENGINE INITIALIZED ")
    print(" Point camera at target. Auto-capture will trigger when stable.")
    print(" Press 'q' to quit manually.")
    print("="*50 + "\n")

    buffer_frames = []
    REQUIRED_BUFFER_SIZE = 5

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 1. Measure Quality Gate Signals Real-Time
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        mean_brightness = np.mean(gray)

        # 2. Evaluate Adaptive Conditions
        is_bright_enough = mean_brightness >= 30.0
        is_sharp_enough = laplacian_var >= 60.0

        # UI Overlay Feedback Logic
        if not is_bright_enough:
            status_msg = "ADJUSTING: Too Dark - Need More Light"
            color = (0, 0, 255) # Red
            buffer_frames.clear()
        elif not is_sharp_enough:
            status_msg = "HOLD STEADY: Adjusting Focus..."
            color = (0, 165, 255) # Orange
            buffer_frames.clear()
        else:
            status_msg = f"LOCKING ON... ({len(buffer_frames) + 1}/{REQUIRED_BUFFER_SIZE})"
            color = (0, 255, 0) # Green
            buffer_frames.append(frame)

        # Draw Real-Time On-Screen Display (OSD)
        cv2.putText(frame, f"Sharpness: {laplacian_var:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Brightness: {mean_brightness:.1f}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, status_msg, (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        cv2.imshow("Vitalis Auto-Capture Feed", frame)

        # 3. Trigger Auto-Capture once 5 stable frames buffered
        if len(buffer_frames) >= REQUIRED_BUFFER_SIZE:
            print("\n[AUTO-CAPTURE TRIGGERED!] Quality conditions met across 5 continuous frames.")
            
            # Median normalize buffered frames
            normalized_vectors = []
            for b_frame in buffer_frames:
                norm_img = gray_world_normalize(b_frame)
                rgb = cv2.cvtColor(norm_img, cv2.COLOR_BGR2RGB)
                lab = cv2.cvtColor(norm_img, cv2.COLOR_BGR2LAB)
                
                r_g = np.mean(rgb[:,:,0]) / (np.mean(rgb[:,:,1]) + 1e-6)
                a_star = np.mean(lab[:,:,1])
                normalized_vectors.append((r_g, a_star))

            median_r_g = np.median([v[0] for v in normalized_vectors])
            median_a_star = np.median([v[1] for v in normalized_vectors])

            print("-" * 45)
            print(f" Stabilized Median R/G Ratio : {median_r_g:.4f}")
            print(f" Stabilized Median CIELAB a* : {median_a_star:.2f}")
            print(" Result logged cleanly. Exiting stream...")
            print("-" * 45)
            
            # Play a short audio cue or delay to signify completion
            time.sleep(1.5)
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_autocapture_stream()