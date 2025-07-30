import cv2, time, asyncio, shutil, subprocess, logging, os, threading, signal, sys, queue, math, numpy as np, json
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import FileResponse, HTMLResponse
from multiprocessing import Process, Queue, set_start_method

# Setup logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set multiprocessing start method untuk Linux
try:
    set_start_method('spawn', force=True)
except RuntimeError:
    pass

HLS_BASE_DIR = "stream"
os.makedirs(HLS_BASE_DIR, exist_ok=True)

CONFIG_FILE = "config/config.json"

def load_config():
    with open(CONFIG_FILE, "r") as f:
        return json.load(f)

config = load_config()

# Array untuk mengelompokkan jenis kendaraan yang di deteksi
vehicle_classid = [1, 2, 3, 5, 7]
class_vehicle_indonesian = {
    1: "Sepeda",
    2: "Mobil", 
    3: "Motor",
    5: "Bus",
    7: "Truk"
}

class StreamManager:
    def __init__(self):
        self.streams = {}
        self.lock = threading.Lock()
    
    def add_stream(self, channel_id, stream_info):
        with self.lock:
            self.streams[channel_id] = stream_info
    
    def remove_stream(self, channel_id):
        with self.lock:
            if channel_id in self.streams:
                del self.streams[channel_id]# ... (kode StreamManager dan lainnya)

class StreamManager:
    def __init__(self):
        self.active_streams = {}

    def add_stream(self, stream_id, stream_info):
        self.active_streams[stream_id] = stream_info

    def get_stream(self, stream_id):
        return self.active_streams.get(stream_id)

    def remove_stream(self, stream_id):
        if stream_id in self.active_streams:
            stream_info = self.active_streams.pop(stream_id)
            
            # Setel event stop untuk memberi tahu child processes
            if 'stop_event' in stream_info and stream_info['stop_event'] is not None:
                logger.info(f"Setting stop event for channel {stream_id}")
                stream_info['stop_event'].set() # <--- PENTING!

            # Terminate processes if they are still alive
            # Beri sedikit waktu untuk proses berhenti sendiri
            time.sleep(1) # Beri waktu 1 detik untuk proses child merespon stop_event

            if stream_info['process_video'].is_alive():
                logger.info(f"Terminating video process for channel {stream_id}")
                stream_info['process_video'].terminate()
                stream_info['process_video'].join(timeout=5)
            if stream_info['process_ffmpeg'].is_alive():
                logger.info(f"Terminating FFmpeg process for channel {stream_id}")
                stream_info['process_ffmpeg'].terminate()
                stream_info['process_ffmpeg'].join(timeout=5)
            # Clean up HLS directory
            if os.path.exists(stream_info['hls_dir']):
                shutil.rmtree(stream_info['hls_dir'], ignore_errors=True)
            logger.info(f"Cleaned up resources for channel {stream_id}")
    
    def get_all_streams(self):
        with self.lock:
            return dict(self.streams)

stream_manager = StreamManager()

class SimpleTracker:
    """Simple object tracker untuk memberikan ID yang konsisten"""
    
    def __init__(self, max_disappeared=15, max_distance=100):
        self.next_id = 1
        self.objects = {}
        self.disappeared = {}
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
    
    def register(self, centroid, class_id):
        """Register objek baru dengan ID unik"""
        self.objects[self.next_id] = {
            'centroid': centroid,
            'class_id': class_id
        }
        self.disappeared[self.next_id] = 0
        self.next_id += 1
        return self.next_id - 1
    
    def deregister(self, object_id):
        """Hapus objek dari tracking"""
        if object_id in self.objects:
            del self.objects[object_id]
        if object_id in self.disappeared:
            del self.disappeared[object_id]
    
    def calculate_distance(self, point1, point2):
        """Hitung jarak Euclidean antara dua titik"""
        return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)
    
    def update(self, detections):
        """Update tracker dengan deteksi baru"""
        if len(detections) == 0:
            # Tidak ada deteksi, mark semua sebagai disappeared
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return {}
        
        # Hitung centroid untuk setiap deteksi
        input_centroids = []
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            input_centroids.append((cx, cy))
        
        # Jika tidak ada objek yang di-track sebelumnya
        if len(self.objects) == 0:
            tracking_results = {}
            for i, centroid in enumerate(input_centroids):
                obj_id = self.register(centroid, detections[i]['class'])
                tracking_results[obj_id] = {
                    'bbox': detections[i]['bbox'],
                    'class': detections[i]['class'],
                    'centroid': centroid
                }
            return tracking_results
        
        # Hitung jarak antara objek yang ada dengan deteksi baru
        object_ids = list(self.objects.keys())
        object_centroids = [self.objects[obj_id]['centroid'] for obj_id in object_ids]
        
        # Matrix jarak
        distance_matrix = []
        for obj_centroid in object_centroids:
            row = []
            for input_centroid in input_centroids:
                distance = self.calculate_distance(obj_centroid, input_centroid)
                row.append(distance)
            distance_matrix.append(row)
        
        # Assignment sederhana - greedy approach
        rows = len(distance_matrix)
        cols = len(distance_matrix[0]) if rows > 0 else 0
        
        used_row_idx = set()
        used_col_idx = set()
        assignments = []
        
        # Cari assignment dengan jarak minimum
        for _ in range(min(rows, cols)):
            min_distance = float('inf')
            min_row = -1
            min_col = -1
            
            for row in range(rows):
                if row in used_row_idx:
                    continue
                for col in range(cols):
                    if col in used_col_idx:
                        continue
                    if distance_matrix[row][col] < min_distance:
                        min_distance = distance_matrix[row][col]
                        min_row = row
                        min_col = col
            
            if min_distance < self.max_distance:
                assignments.append((min_row, min_col))
                used_row_idx.add(min_row)
                used_col_idx.add(min_col)
        
        # Update objek yang ter-assign
        tracking_results = {}
        for row_idx, col_idx in assignments:
            object_id = object_ids[row_idx]
            self.objects[object_id]['centroid'] = input_centroids[col_idx]
            self.objects[object_id]['class_id'] = detections[col_idx]['class']
            self.disappeared[object_id] = 0
            
            tracking_results[object_id] = {
                'bbox': detections[col_idx]['bbox'],
                'class': detections[col_idx]['class'],
                'centroid': input_centroids[col_idx]
            }
        
        # Register objek baru untuk deteksi yang tidak ter-assign
        unused_col_idx = set(range(cols)) - used_col_idx
        for col_idx in unused_col_idx:
            obj_id = self.register(input_centroids[col_idx], detections[col_idx]['class'])
            tracking_results[obj_id] = {
                'bbox': detections[col_idx]['bbox'],
                'class': detections[col_idx]['class'],
                'centroid': input_centroids[col_idx]
            }
        
        # Mark objek yang tidak ter-assign sebagai disappeared
        unused_row_idx = set(range(rows)) - used_row_idx
        for row_idx in unused_row_idx:
            object_id = object_ids[row_idx]
            self.disappeared[object_id] += 1
            if self.disappeared[object_id] > self.max_disappeared:
                self.deregister(object_id)
        
        return tracking_results

class UltralyticsYOLO:
    """YOLO implementation using Ultralytics YOLO11n"""
    
    def __init__(self):
        self.model = None
        self.tracker = SimpleTracker()
        self.load_model()
    
    def load_model(self):
        try:
            # Import ultralytics
            from ultralytics import YOLO
            
            # Load YOLO11n model - gunakan model nano untuk performa
            model_path = "yolo11n.pt"  # nano model untuk speed
            
            # Download model jika belum ada
            if not os.path.exists(model_path):
                logger.info("Downloading YOLO11n model...")
            
            self.model = YOLO(model_path)
            
            # Set model untuk inferensi CPU (lebih stabil untuk streaming)
            self.model.to('cpu')
            
            logger.info("YOLO11n model loaded successfully")
            return True
                
        except ImportError:
            logger.error("Ultralytics not installed. Install with: pip install ultralytics")
            return False
        except Exception as e:
            logger.error(f"Failed to load YOLO11n: {e}")
            return False
    
    def detect(self, frame):
        """Detect objects in frame using YOLO11n"""
        detections = []
        
        if self.model is None:
            return {}
        
        try:
            # Run inference
            results = self.model(frame, conf=0.3, verbose=False)
            
            # Process results
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get class ID
                        class_id = int(box.cls[0])
                        
                        # Filter only vehicle classes
                        if class_id in vehicle_classid:
                            # Get bounding box coordinates
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                            
                            # Ensure bounding box is valid
                            height, width = frame.shape[:2]
                            x1 = max(0, min(x1, width))
                            y1 = max(0, min(y1, height))
                            x2 = max(0, min(x2, width))
                            y2 = max(0, min(y2, height))
                            
                            if x2 > x1 and y2 > y1:  # Valid bounding box
                                detections.append({
                                    'class': class_id,
                                    'bbox': (x1, y1, x2, y2)
                                })
            
            # Update tracker dengan deteksi
            tracking_results = self.tracker.update(detections)
            
        except Exception as e:
            logger.error(f"YOLO11n detection error: {e}")
            tracking_results = {}
        
        return tracking_results

def signal_handler(signum, frame):
    """Handler untuk graceful shutdown"""
    logger.info("Received shutdown signal, cleaning up...")
    cleanup_all_streams()
    sys.exit(0)

def cleanup_all_streams():
    """Cleanup semua active streams"""
    streams = stream_manager.get_all_streams()
    for channel_id, stream_info in streams.items():
        try:
            if 'process_video' in stream_info and stream_info['process_video'].is_alive():
                stream_info['process_video'].terminate()
                stream_info['process_video'].join(timeout=5)
                if stream_info['process_video'].is_alive():
                    stream_info['process_video'].kill()
            
            if 'process_ffmpeg' in stream_info and stream_info['process_ffmpeg'].is_alive():
                stream_info['process_ffmpeg'].terminate()
                stream_info['process_ffmpeg'].join(timeout=5)
                if stream_info['process_ffmpeg'].is_alive():
                    stream_info['process_ffmpeg'].kill()
            
            if 'hls_dir' in stream_info and os.path.exists(stream_info['hls_dir']):
                shutil.rmtree(stream_info['hls_dir'])
                
        except Exception as e:
            logger.error(f"Error cleaning up stream {channel_id}: {e}")
    
    stream_manager.streams.clear()

# Modifikasi pada fungsi videoProcessing untuk output 16:9
def videoProcessing(stream_id: int, frame_queue: Queue):
    """Function untuk memproses video dengan deteksi kendaraan menggunakan YOLO11n - 16:9 Version"""
    try:
        # Support both old and new config format
        stream_config = config["streams"][str(stream_id)]
        if isinstance(stream_config, dict):
            stream_url = stream_config["url"]
        else:
            # If it's just a string (old format)
            stream_url = stream_config

        # Initialize YOLO11n detector
        detector = UltralyticsYOLO()
        
        # Setup video capture
        cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            logger.error(f'CCTV channel {stream_id} sedang tidak aktif atau sedang ada masalah')
            return

        # Optimized capture settings - tetap gunakan resolusi asli untuk capture
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_FPS, 15)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Target output dimensions (16:9 aspect ratio)
        OUTPUT_WIDTH = 960   # 16:9 ratio
        OUTPUT_HEIGHT = 540  # 16:9 ratio
        
        # Original processing dimensions (untuk deteksi)
        PROCESS_WIDTH = 640
        PROCESS_HEIGHT = 480
        
        # Variable untuk waktu dan tracking
        start_time_interval = time.time()
        unique_vehicle_byid = set()
        frame_per_minutes = 0
        average_vehicles = 0
        frame_skip_count = 0
        
        # Background subtractor for motion detection fallback
        backSub = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

        # Text rendering settings - disesuaikan untuk resolusi baru
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5  # Diperbesar untuk resolusi lebih tinggi
        font_thickness = 2
        text_color = (0, 0, 0)
        background_color = (255, 255, 255)
        background_alpha = 0.7
        padding = 4  # Padding diperbesar

        logger.info(f"Started video processing for channel {stream_id} with YOLO11n and 16:9 output ({OUTPUT_WIDTH}x{OUTPUT_HEIGHT})")
        last_successful_frame = None

        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning(f'Stream channel {stream_id} telah terputus!')
                if last_successful_frame is not None:
                    frame = last_successful_frame.copy()
                    ret = True
                else:
                    break

            # Store successful frame
            if ret:
                last_successful_frame = frame.copy()

            # Skip frames untuk target FPS
            frame_skip_count += 1
            if frame_skip_count % 2 != 0:
                continue

            # Resize frame untuk processing (deteksi) - tetap gunakan 4:3 untuk konsistensi deteksi
            if frame.shape[:2] != (PROCESS_HEIGHT, PROCESS_WIDTH):
                processing_frame = cv2.resize(frame, (PROCESS_WIDTH, PROCESS_HEIGHT))
            else:
                processing_frame = frame.copy()

            annotated_frame = processing_frame.copy()
            current_vehicles = 0

            try:
                # Use YOLO11n detection pada frame processing
                tracking_results = detector.detect(processing_frame)
                current_vehicles = len(tracking_results)
                
                for obj_id, obj_info in tracking_results.items():
                    x1, y1, x2, y2 = obj_info['bbox']
                    cls_id = obj_info['class']
                    
                    # Ensure bounding box is within processing frame
                    x1 = max(0, min(x1, PROCESS_WIDTH))
                    y1 = max(0, min(y1, PROCESS_HEIGHT))
                    x2 = max(0, min(x2, PROCESS_WIDTH))
                    y2 = max(0, min(y2, PROCESS_HEIGHT))
                    
                    label_indonesian = class_vehicle_indonesian.get(cls_id, f"Kendaraan")
                    display_text = f"ID:{obj_id} {label_indonesian}"  # Format: ID:123 Mobil
                    
                    # Add to unique vehicles set
                    unique_vehicle_byid.add(obj_id)
                    
                    # Draw rectangle and text pada processing frame
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Calculate text size for background
                    (text_w, text_h), baseline = cv2.getTextSize(display_text, font, font_scale, font_thickness)
                    
                    # Ensure text background is within processing frame
                    text_bg_y1 = max(0, y1 - text_h - baseline - padding * 2)
                    text_bg_x2 = min(PROCESS_WIDTH, x1 + text_w + padding * 2)
                    text_bg_y2 = min(PROCESS_HEIGHT, y1)
                    
                    # Draw text background
                    cv2.rectangle(annotated_frame, 
                                (x1, text_bg_y1), 
                                (text_bg_x2, text_bg_y2), 
                                (0, 0, 0), -1)
                    
                    # Draw text
                    text_y = max(text_h + baseline + padding, y1 - baseline - padding)
                    cv2.putText(annotated_frame, display_text, 
                              (x1 + padding, text_y), 
                              font, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)
                
            except Exception as detect_error:
                logger.error(f"YOLO11n detection error for channel {stream_id}: {detect_error}")
                # Fallback motion detection (pada processing frame)
                try:
                    fgMask = backSub.apply(processing_frame)
                    contours, _ = cv2.findContours(fgMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    motion_objects = 0
                    for contour in contours:
                        if cv2.contourArea(contour) > 500:
                            x, y, w, h = cv2.boundingRect(contour)
                            x = max(0, min(x, PROCESS_WIDTH-w))
                            y = max(0, min(y, PROCESS_HEIGHT-h))
                            w = min(w, PROCESS_WIDTH-x)
                            h = min(h, PROCESS_HEIGHT-y)
                            
                            cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                            cv2.putText(annotated_frame, "Moving Object", (x, max(y - 10, 20)), 
                                      font, font_scale, (0, 255, 255), font_thickness)
                            motion_objects += 1
                    
                    current_vehicles = motion_objects
                    
                except Exception as motion_error:
                    logger.error(f"Motion detection error for channel {stream_id}: {motion_error}")
                    current_vehicles = 0

            # Calculate statistics
            current_time = time.time()
            elapsed_time = current_time - start_time_interval
            frame_per_minutes += 1

            if elapsed_time >= 60:
                if frame_per_minutes > 0:
                    average_vehicles = len(unique_vehicle_byid)
                    logger.info(f"Channel {stream_id} - Kendaraan unik dalam 1 menit ada: {average_vehicles}")
                else:
                    average_vehicles = 0
                    logger.info(f"Channel {stream_id} - Tidak ada kendaraan yang lewat")
            
                # Reset counters
                start_time_interval = time.time()
                unique_vehicle_byid.clear()
                frame_per_minutes = 0

            # Prepare overlay text (disesuaikan untuk processing frame) - tanpa channel
            text_lines = [
                f"Jumlah kendaraan saat ini: {current_vehicles} unit",
                f"Rata-rata kendaraan per menit: {average_vehicles} unit"
            ]

            # Calculate overlay dimensions untuk processing frame
            max_width = 0
            total_height = 0
            line_heights = []

            for line in text_lines:
                (w, h), baseline = cv2.getTextSize(line, font, font_scale, font_thickness)
                max_width = max(max_width, w)
                line_heights.append(h + baseline)
                total_height += (h + baseline + padding)

            # Position overlay pada processing frame
            bg_x1 = 10
            bg_y1 = 30
            bg_x2 = min(PROCESS_WIDTH, bg_x1 + max_width + (2 * padding))
            bg_y2 = min(PROCESS_HEIGHT, bg_y1 + total_height + (2 * padding))

            # Draw semi-transparent background
            overlay = annotated_frame.copy()
            cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), background_color, -1)
            cv2.addWeighted(overlay, background_alpha, annotated_frame, 1 - background_alpha, 0, annotated_frame)

            # Draw text lines
            current_y_offset = bg_y1 + padding + line_heights[0]
            for i, line in enumerate(text_lines):
                if i > 0:
                    current_y_offset += (line_heights[i-1] + padding)
                cv2.putText(annotated_frame, line, (bg_x1 + padding, current_y_offset), 
                          font, font_scale, text_color, font_thickness, cv2.LINE_AA)

            # RESIZE KE 16:9 UNTUK OUTPUT
            # Resize dari processing frame (640x480) ke output frame (960x540)
            output_frame = cv2.resize(annotated_frame, (OUTPUT_WIDTH, OUTPUT_HEIGHT), 
                                    interpolation=cv2.INTER_AREA)

            # Ensure frame is contiguous and correct format
            if not output_frame.flags['C_CONTIGUOUS']:
                output_frame = np.ascontiguousarray(output_frame)
            
            # Put resized frame in queue
            try:
                # Clear old frames if queue is getting full
                while frame_queue.qsize() > 3:
                    try:
                        frame_queue.get_nowait()
                    except queue.Empty:
                        break
                
                frame_queue.put(output_frame, block=False)
            except queue.Full:
                pass
            except Exception as e:
                logger.warning(f"Error putting frame in queue for channel {stream_id}: {e}")

    except Exception as e:
        logger.error(f"Error in video processing for channel {stream_id}: {e}")
    finally:
        try:
            cap.release()
        except:
            pass
        logger.info(f"Video processing for channel {stream_id} stopped")

def ffmpegConvert(channel_id: int, frame_queue: Queue, hls_dir: str):
    """Function untuk konversi frame ke HLS - 16:9 Version"""
    # Update dimensions untuk 16:9
    input_width, input_height = 960, 540

    # Clean and create HLS directory
    if os.path.exists(hls_dir):
        shutil.rmtree(hls_dir)
    os.makedirs(hls_dir, exist_ok=True)

    # Updated FFmpeg command untuk 16:9
    command = [
        'ffmpeg',
        '-y',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-pix_fmt', 'bgr24',
        '-s', f"{input_width}x{input_height}",  # Updated size
        '-i', '-',
        
        # Video encoding parameters
        '-c:v', 'libx264',
        '-preset', 'ultrafast',
        '-tune', 'zerolatency',
        '-profile:v', 'baseline',
        '-level', '3.1',  # Slightly higher level for higher resolution
        '-pix_fmt', 'yuv420p',
        
        # Rate control - adjusted for higher resolution
        '-crf', '26',  # Slightly lower CRF for better quality
        '-maxrate', '1500k',  # Higher bitrate for larger resolution
        '-bufsize', '3000k',
        '-g', '30',
        '-keyint_min', '15',
        '-sc_threshold', '0',
        
        # HLS specific parameters
        '-f', 'hls',
        '-hls_time', '3',
        '-hls_list_size', '6',
        '-hls_flags', 'delete_segments+program_date_time',
        '-hls_segment_type', 'mpegts',
        "-hls_base_url", f"/stream/{channel_id}/",
        '-hls_segment_filename', os.path.join(hls_dir, 'segment%03d.ts'),
        
        '-threads', '2',
        '-loglevel', 'warning',
        
        os.path.join(hls_dir, 'index.m3u8')
    ]

    proc = None
    frame_count = 0
    last_frame_time = time.time()
    
    try:
        proc = subprocess.Popen(command, 
                              stdin=subprocess.PIPE, 
                              stderr=subprocess.PIPE, 
                              stdout=subprocess.PIPE,
                              bufsize=0)
        
        logger.info(f"FFmpeg started for channel {channel_id} with 16:9 resolution ({input_width}x{input_height})")
        
        while True:
            try:
                frame_np = frame_queue.get(timeout=10)
                
                if proc.poll() is not None:
                    logger.warning(f"FFmpeg process for channel {channel_id} has exited")
                    break
                
                if frame_np is None or frame_np.size == 0:
                    logger.warning(f"Channel {channel_id}: Received empty frame, skipping")
                    continue
                
                # Validate frame dimensions untuk 16:9
                if frame_np.shape != (input_height, input_width, 3):
                    logger.warning(f"Channel {channel_id}: Frame shape mismatch {frame_np.shape}, expected ({input_height}, {input_width}, 3)")
                    frame_np = cv2.resize(frame_np, (input_width, input_height))
                
                if not frame_np.flags['C_CONTIGUOUS']:
                    frame_np = np.ascontiguousarray(frame_np)
                
                try:
                    proc.stdin.write(frame_np.tobytes())
                    proc.stdin.flush()
                    frame_count += 1
                    
                    if frame_count % 150 == 0:
                        logger.info(f"Channel {channel_id}: Processed {frame_count} frames (16:9 YOLOv8)")
                        
                except BrokenPipeError:
                    logger.error(f"Channel {channel_id}: Broken pipe to FFmpeg")
                    break
                except OSError as e:
                    logger.error(f"Channel {channel_id}: OS error writing to FFmpeg: {e}")
                    break
                    
            except Exception as e:
                if "Empty" in str(e) or "timeout" in str(e).lower():
                    if proc.poll() is not None:
                        logger.warning(f"FFmpeg process for channel {channel_id} died during timeout")
                        break
                    continue
                else:
                    logger.error(f"FFmpeg Worker for Channel {channel_id}: Error processing frame: {e}")
                    break

    except Exception as e:
        logger.error(f"FFmpeg Worker for Channel {channel_id}: Error running subprocess: {e}")
    finally:
        # Cleanup process (sama seperti sebelumnya)
        if proc:
            try:
                if proc.stdin:
                    proc.stdin.close()
            except:
                pass
            
            try:
                proc.wait(timeout=3)
            except subprocess.TimeoutExpired:
                proc.terminate()
                try:
                    proc.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    proc.kill()
        
        logger.info(f"FFmpeg Worker for Channel {channel_id}: Process terminated, processed {frame_count} frames (16:9 YOLOv8)")

# Initialize FastAPI app
app = FastAPI(title="HLS Stream with YOLOv8 Vehicle Detection", version="3.0.0")

# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# --- FastAPI Endpoints ---
@app.get("/stream/{channel_id}")
async def get_hls_playlist(channel_id: int):
    """Endpoint untuk mendapatkan HLS playlist dengan headers yang tepat"""
    hls_stream_dir = os.path.join(HLS_BASE_DIR, str(channel_id))
    m3u8_path = os.path.join(hls_stream_dir, 'index.m3u8')

    logger.info(f"HLS request for channel {channel_id}, checking path: {m3u8_path}")

    if not os.path.exists(m3u8_path):
        stream_info = stream_manager.get_stream(channel_id)
        if not stream_info:
            logger.info(f"Starting YOLOv8 workers for channel {channel_id}")
            
            # Create frame queue
            frame_q = Queue(maxsize=30)

            # Create and start video processing process
            p_video = Process(target=videoProcessing, args=(channel_id, frame_q))
            p_video.daemon = True
            p_video.start()
            
            # Create and start FFmpeg streaming process
            p_ffmpeg = Process(target=ffmpegConvert, args=(channel_id, frame_q, hls_stream_dir))
            p_ffmpeg.daemon = True
            p_ffmpeg.start()

            stream_info = {
                'frame_queue': frame_q,
                'process_video': p_video,
                'process_ffmpeg': p_ffmpeg,
                'hls_dir': hls_stream_dir,
                'start_time': time.time()
            }
            
            stream_manager.add_stream(channel_id, stream_info)
            
            logger.info(f"YOLOv8 workers started for channel {channel_id}. Video PID: {p_video.pid}, FFmpeg PID: {p_ffmpeg.pid}")
            
            # Wait for HLS files to be generated with progress logging
            max_wait = 30  # seconds
            wait_count = 0
            while not os.path.exists(m3u8_path) and wait_count < max_wait:
                await asyncio.sleep(1)
                wait_count += 1
                
                # Log progress every 5 seconds
                if wait_count % 5 == 0:
                    logger.info(f"Waiting for HLS generation for channel {channel_id}... ({wait_count}/{max_wait}s)")
                    
                    # Check if processes are still alive
                    if not p_video.is_alive():
                        logger.error(f"Video process for channel {channel_id} died")
                        break
                    if not p_ffmpeg.is_alive():
                        logger.error(f"FFmpeg process for channel {channel_id} died")
                        break
                
            if not os.path.exists(m3u8_path):
                # Cleanup failed stream
                stream_manager.remove_stream(channel_id)
                if p_video.is_alive():
                    p_video.terminate()
                if p_ffmpeg.is_alive():
                    p_ffmpeg.terminate()
                
                raise HTTPException(status_code=503, 
                                  detail=f"HLS stream for channel {channel_id} failed to start. Check RTSP connection.")
        else:
            # Stream exists but m3u8 not ready yet
            max_retry = 10
            retry_count = 0
            while not os.path.exists(m3u8_path) and retry_count < max_retry:
                await asyncio.sleep(2)
                retry_count += 1
                logger.info(f"Retry {retry_count}/{max_retry} waiting for m3u8 file for channel {channel_id}")
                
            if not os.path.exists(m3u8_path):
                raise HTTPException(status_code=503, 
                                  detail=f"HLS stream for channel {channel_id} is starting but not ready yet. Please wait a moment.")
    
    # Verify file exists and is readable
    try:
        with open(m3u8_path, 'r') as f:
            content = f.read()
            if len(content) < 10:  # File too small, probably empty
                raise HTTPException(status_code=503, 
                                  detail=f"HLS playlist for channel {channel_id} is empty or corrupted.")
    except Exception as e:
        logger.error(f"Error reading m3u8 file for channel {channel_id}: {e}")
        raise HTTPException(status_code=503, 
                          detail=f"Cannot read HLS playlist for channel {channel_id}.")
            
    logger.info(f"Serving HLS playlist for channel {channel_id}")
    
    # Return with proper headers for HLS streaming
    headers = {
        "Content-Type": "application/vnd.apple.mpegurl",
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Pragma": "no-cache",
        "Expires": "0",
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type"
    }
    
    return FileResponse(m3u8_path, 
                       media_type="application/vnd.apple.mpegurl",
                       headers=headers)

@app.get("/stream/{channel_id}/{segment_name}")
async def get_hls_segment(channel_id: int, segment_name: str):
    """Endpoint untuk menyajikan segmen TS dengan headers yang tepat"""
    # Validate segment name untuk security
    if not segment_name.endswith('.ts') or '..' in segment_name:
        raise HTTPException(status_code=400, detail="Invalid segment name")
    
    hls_stream_dir = os.path.join(HLS_BASE_DIR, str(channel_id))
    segment_path = os.path.join(hls_stream_dir, segment_name)

    if not os.path.exists(segment_path):
        logger.warning(f"HLS segment not found: {segment_path}")
        raise HTTPException(status_code=404, detail="HLS segment not found")
    
    # Headers untuk TS segments
    headers = {
        "Content-Type": "video/mp2t",
        "Cache-Control": "max-age=3600",
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type"
    }
    
    return FileResponse(segment_path, 
                       media_type="video/mp2t",
                       headers=headers)

# Add OPTIONS handler for CORS
@app.options("/stream/{channel_id}")
async def options_hls_playlist(channel_id: int):
    """Handle CORS preflight for HLS playlist"""
    headers = {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type"
    }
    return Response(headers=headers)

@app.options("/stream/{channel_id}/{segment_name}")
async def options_hls_segment(channel_id: int, segment_name: str):
    """Handle CORS preflight for HLS segments"""
    headers = {
        "Access-Control-Allow-Origin": "*",
        "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
        "Access-Control-Allow-Headers": "Content-Type"
    }
    return Response(headers=headers)

@app.get("/stop/{channel_id}")
async def stop_stream(channel_id: int):
    """Endpoint untuk menghentikan stream"""
    stream_info = stream_manager.get_stream(channel_id)
    if stream_info:
        try:
            # Terminate processes
            if 'process_video' in stream_info and stream_info['process_video'].is_alive():
                stream_info['process_video'].terminate()
                stream_info['process_video'].join(timeout=5)
                if stream_info['process_video'].is_alive():
                    stream_info['process_video'].kill()
                
            if 'process_ffmpeg' in stream_info and stream_info['process_ffmpeg'].is_alive():
                stream_info['process_ffmpeg'].terminate()
                stream_info['process_ffmpeg'].join(timeout=5)
                if stream_info['process_ffmpeg'].is_alive():
                    stream_info['process_ffmpeg'].kill()
            
            # Clean up HLS directory
            if 'hls_dir' in stream_info and os.path.exists(stream_info['hls_dir']):
                shutil.rmtree(stream_info['hls_dir'])
                
            # Remove from active streams
            stream_manager.remove_stream(channel_id)
            
            logger.info(f"Stream for channel {channel_id} stopped")
            return {"message": f"Stream for channel {channel_id} stopped successfully"}
        except Exception as e:
            logger.error(f"Error stopping stream {channel_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Error stopping stream: {e}")
    else:
        raise HTTPException(status_code=404, detail="Stream not found")

@app.get("/status")
async def get_status():
    """Endpoint untuk mendapatkan status semua stream"""
    status = {}
    streams = stream_manager.get_all_streams()
    for channel_id, stream_info in streams.items():
        try:
            status[channel_id] = {
                "video_process_alive": stream_info['process_video'].is_alive() if 'process_video' in stream_info else False,
                "ffmpeg_process_alive": stream_info['process_ffmpeg'].is_alive() if 'process_ffmpeg' in stream_info else False,
                "hls_dir_exists": os.path.exists(stream_info['hls_dir']) if 'hls_dir' in stream_info else False
            }
        except Exception as e:
            status[channel_id] = {"error": str(e)}
    return status

@app.get("/", response_class=HTMLResponse)
async def root():
    with open("index.html", "r") as file:
        return HTMLResponse(content=file.read(), status_code=200)

# --- Event Handler ---
@app.on_event("shutdown")
async def shutdown_event():
    """Handler untuk shutdown aplikasi"""
    logger.info("Shutting down FastAPI server...")
    cleanup_all_streams()

if __name__ == "__main__":
    import uvicorn
    
    # Ensure proper multiprocessing setup
    try:
        set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    
    # Run the FastAPI application
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")