import cv2
import numpy as np
from PIL import Image
import os
import time
import dlib

# Load the facial landmarks predictor for face shape detection
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

def overlay_image_alpha(img, img_overlay, pos, alpha_overlay=1.0):
    """Overlay img_overlay on top of img at the position specified by pos."""
    x, y = pos
    y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
    x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])

    overlay_image = img_overlay[y1-y:y2-y, x1-x:x2-x]
    img_crop = img[y1:y2, x1:x2]

    alpha = overlay_image[:, :, 3] / 255.0 * alpha_overlay  # Adjust alpha for transparency control
    alpha_inv = 1.0 - alpha

    for c in range(0, 3):
        img_crop[:, :, c] = alpha * overlay_image[:, :, c] + alpha_inv * img_crop[:, :, c]

    img[y1:y2, x1:x2] = img_crop
    return img

def load_glasses_images_from_folder(folder):
    """Load all PNG images from the specified folder."""
    glasses_images = []
    for filename in os.listdir(folder):
        if filename.endswith(".png"):
            img = Image.open(os.path.join(folder, filename)).convert("RGBA")
            glasses_images.append(cv2.cvtColor(np.array(img), cv2.COLOR_RGBA2BGRA))
    return glasses_images

def create_sidebar(glasses_images, current_glasses, frame_height):
    """Create a sidebar to display available glasses."""
    sidebar = np.zeros((frame_height, 100, 3), dtype=np.uint8)
    thumbnail_height = frame_height // len(glasses_images)
    
    for i, img in enumerate(glasses_images):
        y = i * thumbnail_height
        resized = cv2.resize(img[:,:,:3], (80, thumbnail_height - 20))
        sidebar[y+10:y+thumbnail_height-10, 10:90] = resized
        
        if i == current_glasses:
            cv2.rectangle(sidebar, (5, y+5), (95, y+thumbnail_height-5), (0, 255, 0), 2)
    
    return sidebar

def save_snapshot(frame, folder='snapshots', filename=None):
    """Save the current frame with the glasses overlay applied."""
    if not os.path.exists(folder):
        os.makedirs(folder)
    if filename is None:
        filename = time.strftime("%Y%m%d-%H%M%S") + '.png'
    filepath = os.path.join(folder, filename)
    cv2.imwrite(filepath, frame)
    print(f"Snapshot saved as {filepath}")

def draw_help_menu(frame):
    """Display a help menu showing available key bindings."""
    help_text = [
        "'n' - Next glasses",
        "'s' - Save snapshot",
        "'u' - Increase transparency",
        "'d' - Decrease transparency",
        "'+' - Increase glasses size",
        "'-' - Decrease glasses size",
        "'r' - Reset glasses",
        "'c' - Switch camera",
        "'z' - Zoom in",
        "'x' - Zoom out",
        "'h' - Help menu",
        "'f' - Switch face shape",
        "'q' - Quit"
    ]
    y0, dy = 50, 30
    for i, text in enumerate(help_text):
        y = y0 + i * dy
        cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

def detect_face_shape(landmarks):
    """Detect face shape based on facial landmarks."""
    jaw_width = landmarks.part(16).x - landmarks.part(0).x
    face_height = landmarks.part(8).y - landmarks.part(19).y

    if jaw_width / face_height < 1.1:
        return "round"
    elif jaw_width / face_height > 1.4:
        return "square"
    else:
        return "oval"

def apply_glasses_filter(glasses_image, color_shift=(0, 0, 0)):
    """Apply color filter to glasses."""
    hue_shift, brightness_shift, saturation_shift = color_shift
    hsv = cv2.cvtColor(glasses_image, cv2.COLOR_BGRA2BGR)
    hsv = cv2.cvtColor(hsv, cv2.COLOR_BGR2HSV)

    h, s, v = cv2.split(hsv)
    h = (h + hue_shift) % 180
    s = np.clip(s + saturation_shift, 0, 255)
    v = np.clip(v + brightness_shift, 0, 255)

    hsv_filtered = cv2.merge([h, s, v])
    result = cv2.cvtColor(hsv_filtered, cv2.COLOR_HSV2BGR)
    result = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)

    return result

def main():
    cap = cv2.VideoCapture(0)
    current_camera = 0  # 0 for front camera

    glasses_images = load_glasses_images_from_folder("glasses_images_folder")  # Load glasses from folder
    current_glasses = 0
    alpha_overlay = 1.0  # Default transparency
    glasses_scale = 1.0  # Default scale for glasses size
    color_shift = (0, 0, 0)  # Default color filter (hue, brightness, saturation)
    zoom_factor = 1.0  # Zoom level
    face_shapes = ["round", "square", "oval"]
    selected_face_shape = 0  # Default face shape (manual selection)

    def mouse_callback(event, x, y, flags, param):
        nonlocal glasses_scale
        if event == cv2.EVENT_MOUSEWHEEL:
            if flags > 0:  # Scroll up
                glasses_scale = min(glasses_scale + 0.05, 2.0)
            else:  # Scroll down
                glasses_scale = max(glasses_scale - 0.05, 0.5)

    cv2.namedWindow('Glasses Try-On App')
    cv2.setMouseCallback('Glasses Try-On App', mouse_callback)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect face and facial landmarks
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        sidebar = create_sidebar(glasses_images, current_glasses, frame.shape[0])

        if len(faces) == 0:
            cv2.putText(frame, "No face detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            for face in faces:
                landmarks = predictor(gray, face)
                face_shape = detect_face_shape(landmarks)

                # Allow manual face shape switching
                face_shape = face_shapes[selected_face_shape]

                # Overlay glasses
                x, y, w, h = face.left(), face.top(), face.width(), face.height()
                glasses = cv2.resize(glasses_images[current_glasses], (int(w * glasses_scale), int(h / 3 * glasses_scale)))
                glasses = apply_glasses_filter(glasses, color_shift)
                glasses_pos = (x, y + int(h / 4))
                frame = overlay_image_alpha(frame, glasses, glasses_pos, alpha_overlay)

        combined_frame = np.hstack((frame, sidebar))
        cv2.imshow('Glasses Try-On App', combined_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('n'):
            current_glasses = (current_glasses + 1) % len(glasses_images)
        elif key == ord('s'):
            save_snapshot(frame)
        elif key == ord('u'):
            alpha_overlay = min(1.0, alpha_overlay + 0.1)  # Increase transparency
        elif key == ord('d'):
            alpha_overlay = max(0.0, alpha_overlay - 0.1)  # Decrease transparency
        elif key == ord('+'):
            glasses_scale = min(glasses_scale + 0.05, 2.0)  # Increase glasses size
        elif key == ord('-'):
            glasses_scale = max(glasses_scale - 0.05, 0.5)  # Decrease glasses size
        elif key == ord('f'):
            selected_face_shape = (selected_face_shape + 1) % len(face_shapes)  # Switch face shape
        elif key == ord('h'):
            draw_help_menu(frame)
        elif key == ord('c'):
            current_camera = (current_camera + 1) % 2
            cap = cv2.VideoCapture(current_camera)
        elif key == ord('z'):
            zoom_factor = min(zoom_factor + 0.1, 2.0)  # Zoom in
        elif key == ord('x'):
            zoom_factor = max(zoom_factor - 0.1, 1.0)  # Zoom out

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

# GlassesTryOn