import os
import cv2
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, welch

class VitalisEngine:
    def __init__(self, buffer_seconds=10, fps=30):
        self.fps = fps
        self.buffer_size = int(buffer_seconds * fps)
        self.rgb_buffer = []      # Forehead/Cheek skin BGR data for HR
        self.chest_buffer = []    # Torso luminance motion data for RR
        
        local_cascade = "haarcascade_frontalface_default.xml"
        system_cascade = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        cascade_path = local_cascade if os.path.exists(local_cascade) else system_cascade
        
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

    def compute_fitzpatrick_tone(self, face_roi):
        """Calculates Individual Typology Angle (ITA°) from face skin pixels to determine Fitzpatrick scale."""
        if face_roi is None or face_roi.size == 0:
            return "Type V"

        # Convert BGR to CIELAB color space
        lab = cv2.cvtColor(face_roi, cv2.COLOR_BGR2LAB)
        
        # Normalize L and b components to standard CIELAB ranges
        L = lab[:, :, 0] * (100.0 / 255.0)
        b = lab[:, :, 2] - 128.0
        
        mean_L = np.mean(L)
        mean_b = np.mean(b) + 1e-6
        
        # Calculate ITA angle in degrees
        ita = np.arctan2((mean_L - 50.0), mean_b) * (180.0 / np.pi)
        
        # Standard ITA classification mapping to Fitzpatrick Scale
        if ita > 55:
            return "Type I-II"
        elif ita > 28:
            return "Type III"
        elif ita > 10:
            return "Type IV"
        elif ita > -30:
            return "Type V"
        else:
            return "Type VI"

    def _extract_rois_mean_rgb(self, frame, x, y, w, h):
        """Extract dermal skin channels across forehead and cheeks."""
        fh_y1, fh_y2 = int(y + 0.15 * h), int(y + 0.30 * h)
        fh_x1, fh_x2 = int(x + 0.25 * w), int(x + 0.75 * w)
        forehead = frame[fh_y1:fh_y2, fh_x1:fh_x2]

        lc_y1, lc_y2 = int(y + 0.50 * h), int(y + 0.70 * h)
        lc_x1, lc_x2 = int(x + 0.15 * w), int(x + 0.40 * w)
        left_cheek = frame[lc_y1:lc_y2, lc_x1:lc_x2]

        rc_y1, rc_y2 = int(y + 0.50 * h), int(y + 0.70 * h)
        rc_x1, rc_x2 = int(x + 0.60 * w), int(x + 0.85 * w)
        right_cheek = frame[rc_y1:rc_y2, rc_x1:rc_x2]

        pixels = [np.mean(r, axis=(0, 1)) for r in [forehead, left_cheek, right_cheek] if r.size > 0]
        if len(pixels) > 0:
            mean_bgr = np.mean(pixels, axis=0)
            return [mean_bgr[2], mean_bgr[1], mean_bgr[0]], forehead
        return None, None

    def _estimate_anemia_risk(self, frame, x, y, w, h):
        """Estimate ocular conjunctiva redness ratio."""
        eye_y1, eye_y2 = int(y + 0.25 * h), int(y + 0.45 * h)
        eye_x1, eye_x2 = int(x + 0.1 * w), int(x + 0.9 * w)
        eye_region = frame[eye_y1:eye_y2, eye_x1:eye_x2]
        
        if eye_region.size > 0:
            r_val = np.mean(eye_region[:, :, 2])
            g_val = np.mean(eye_region[:, :, 1]) + 1e-6
            redness_ratio = r_val / g_val
            
            if redness_ratio < 1.15:
                return "Elevated (Pallor Detected)"
            elif redness_ratio < 1.30:
                return "Moderate"
            return "Low / Normal"
        return "Normal"

    def _pos_algorithm(self, rgb_signals):
        """Plane-Orthogonal-to-Skin chrominance method for rPPG extraction."""
        rgb = np.array(rgb_signals)
        mean_rgb = np.mean(rgb, axis=0) + 1e-6
        rgb_norm = rgb / mean_rgb
        
        S1 = rgb_norm[:, 1] - rgb_norm[:, 2]
        S2 = rgb_norm[:, 1] + rgb_norm[:, 2] - 2 * rgb_norm[:, 0]
        
        std_S1 = np.std(S1)
        std_S2 = np.std(S2)
        
        if std_S2 == 0:
            return np.zeros(len(rgb))
            
        alpha = std_S1 / std_S2
        return S1 + (alpha * S2)

    def _bandpass_filter(self, signal, lowcut, highcut, order=3):
        nyquist = 0.5 * self.fps
        b, a = butter(order, [lowcut / nyquist, highcut / nyquist], btype='band')
        return filtfilt(b, a, signal)

    def calculate_respiratory_rate(self):
        """Dynamic Respiratory Rate estimation from upper torso displacement."""
        min_frames = int(self.fps * 6)
        if len(self.chest_buffer) < min_frames:
            return 16.0

        signal_data = np.array(self.chest_buffer)

        lowcut, highcut = 0.12, 0.45
        filtered = self._bandpass_filter(signal_data, lowcut, highcut, order=2)

        min_distance = int(self.fps * 1.8)
        peaks, _ = find_peaks(filtered, distance=min_distance, prominence=0.03)

        if len(peaks) < 2:
            return 16.0

        intervals = np.diff(peaks) / self.fps
        avg_interval = np.mean(intervals)

        if avg_interval <= 0:
            return 16.0

        rr_bpm = 60.0 / avg_interval
        return float(np.round(np.clip(rr_bpm, 8.0, 30.0), 1))

    def process_frame_advanced(self, frame, show_rois=True):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(120, 120))
        
        bpm, rr, anemia_risk, sqi = None, 16.0, "Normal", 0.0
        detected_skin_tone = "Type V"
        filtered_pulse = []
        progress = 0.0
        
        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
            frame_h, frame_w, _ = frame.shape

            # 1. FACE ROI & AUTO SKIN-TONE DETECTION
            rgb_mean, forehead_roi = self._extract_rois_mean_rgb(frame, x, y, w, h)
            if rgb_mean is not None:
                self.rgb_buffer.append(rgb_mean)
                detected_skin_tone = self.compute_fitzpatrick_tone(forehead_roi)

            # 2. CHEST / TORSO ROI (for Respiration)
            chest_y1 = int(y + 1.05 * h)
            chest_y2 = min(frame_h, int(y + 2.1 * h))
            chest_x1 = max(0, int(x - 0.25 * w))
            chest_x2 = min(frame_w, int(x + 1.25 * w))

            chest_roi = frame[chest_y1:chest_y2, chest_x1:chest_x2]
            if chest_roi.size > 0:
                chest_val = np.mean(chest_roi[:, :, 1])
                self.chest_buffer.append(chest_val)

            # Maintain buffer sizes
            if len(self.rgb_buffer) > self.buffer_size:
                self.rgb_buffer.pop(0)
            if len(self.chest_buffer) > self.buffer_size:
                self.chest_buffer.pop(0)

            progress = (len(self.rgb_buffer) / self.buffer_size) * 100

            # 3. OVERLAY RENDERING
            if show_rois:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 242, 0), 2)
                cv2.putText(frame, f"rPPG FACE ROI [{detected_skin_tone}]", (x, y - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 242, 0), 1)

                if chest_roi.size > 0:
                    cv2.rectangle(frame, (chest_x1, chest_y1), (chest_x2, chest_y2), (255, 0, 255), 2)
                    cv2.putText(frame, "CHEST MOTION ROI", (chest_x1, chest_y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 255), 1)

            # Compute Vitals
            rr = self.calculate_respiratory_rate()
            anemia_risk = self._estimate_anemia_risk(frame, x, y, w, h)

            if len(self.rgb_buffer) >= int(self.fps * 3):
                pos_pulse = self._pos_algorithm(self.rgb_buffer)
                filtered_pulse = self._bandpass_filter(pos_pulse, lowcut=0.75, highcut=3.0)
                
                freqs, psd = welch(filtered_pulse, fs=self.fps, nperseg=len(filtered_pulse))
                valid_idx = np.where((freqs >= 0.75) & (freqs <= 3.0))[0]
                
                if len(valid_idx) > 0:
                    peak_idx = np.argmax(psd[valid_idx])
                    peak_freq = freqs[valid_idx[peak_idx]]
                    bpm = float(peak_freq * 60)
                    sqi = float((psd[valid_idx[peak_idx]] / (np.sum(psd[valid_idx]) + 1e-6)) * 100)

        return frame, bpm, rr, anemia_risk, sqi, filtered_pulse, progress, detected_skin_tone
