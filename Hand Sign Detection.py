import torch
from torch import nn
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
import time
import numpy as np
import cv2 as cv
import HandTrackingModule as htm
import joblib

detector = htm.HandDetector(max_num_hands=1, min_detection_confidence=0.6, min_tracking_confidence=0.5)
detector_relaxed = htm.HandDetector(max_num_hands=1, min_detection_confidence=0.4, min_tracking_confidence=0.3)

def capture_hand_img(img):
    handsInfo, frame = detector.findHands(img, draw=False)
    if not handsInfo:
        handsInfo, frame = detector_relaxed.findHands(img, draw=False)
    if not handsInfo:
        return None
    
    hand = handsInfo[0]
    x1, y1, x2, y2 = hand['bbox']

    # Adaptive offset
    hand_width = x2 - x1
    hand_height = y2 - y1
    adaptive_offset = int(max(20, min(hand_width, hand_height) * 0.15))

    # Clamp crop within frame bounds
    x1, y1 = max(0, x1 - adaptive_offset), max(0, y1 - adaptive_offset)
    x2, y2 = min(frame.shape[1], x2 + adaptive_offset), min(frame.shape[0], y2 + adaptive_offset)

    cropped_hand = frame[y1:y2, x1:x2]
    h, w = cropped_hand.shape[:2]

    if h == 0 or w == 0:
        return None

    # Resize with aspect ratio + padding
    bg_h, bg_w = (256, 256)
    aspect_ratio = w / h
    if aspect_ratio > 1:
        new_w, new_h = bg_w, int(bg_w / aspect_ratio)
    else:
        new_h, new_w = bg_h, int(bg_h * aspect_ratio)

    resized = cv.resize(cropped_hand, (new_w, new_h))
    img_new = np.full((bg_h, bg_w, 3), 128, dtype='uint8')

    x_offset, y_offset = (bg_w - new_w) // 2, (bg_h - new_h) // 2
    img_new[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized

    return img_new


class SignLanguageClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2   = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3   = nn.BatchNorm2d(128)

        self.pooling = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))

        # Use lighter spatial dropout for conv features
        self.drop_conv = nn.Dropout2d(p=0.15)
        # Use stronger dropout for the classifier head
        self.drop_fc   = nn.Dropout(p=0.40)

        self.fc1 = nn.Linear(128 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 128)
        self.output = nn.Linear(128, num_classes)

    def forward(self, x):
        for bn, conv in zip([self.bn1, self.bn2, self.bn3], [self.conv1, self.conv2, self.conv3]):
            x = self.pooling(F.relu(bn(conv(x))))
            x = self.drop_conv(x)

        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x)); x = self.drop_fc(x)
        x = F.relu(self.fc2(x)); x = self.drop_fc(x)
        return self.output(x)


device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load label encoder
encoder = joblib.load(r'D:\Work Files\OneDrive - Mountain View\Desktop\Machine Learning\Computer Vision\Hand Sign Detection 3\label_encoder.pkl')
idx_to_label = {i: label for i, label in enumerate(encoder.classes_)}
num_classes = len(idx_to_label)

# Initialize model
model = SignLanguageClassifier(num_classes).to(device)

# Load checkpoint - FIXED VERSION
checkpoint_path = r'D:\Work Files\OneDrive - Mountain View\Desktop\Machine Learning\Computer Vision\Hand Sign Detection 3\best_sign_language_model.pth'

try:
    # Try loading as dictionary first (new format)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # New checkpoint format with metadata
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
        print(f"📊 Model validation accuracy: {checkpoint.get('val_acc', 'unknown'):.2f}%")
    else:
        # Old format - direct state dict
        model.load_state_dict(checkpoint)
        print("✅ Loaded model (legacy format)")
        
except Exception as e:
    print(f"❌ Error loading model: {e}")
    print("Make sure the model file exists and matches the model architecture")
    exit(1)

model.eval()

# Image preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


def classify(frame):
    """Classify hand sign from frame"""
    if frame is None:
        return "No hand detected"
    
    try:
        image = Image.fromarray(frame)
        input_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            logits = model(input_tensor)
            probs = F.softmax(logits, dim=1)
            max_prob, pred_idx = torch.max(probs, dim=1)
            confidence = float(max_prob.item())
        
        predicted_label = idx_to_label[int(pred_idx.item())]
        
        # Only return prediction if confidence is high enough
        if confidence > 0.5:  # Adjust threshold as needed
            return f"{predicted_label} ({confidence:.2f})"
        else:
            return "Uncertain"
            
    except Exception as e:
        print(f"Classification error: {e}")
        return "Error"


# Initialize camera and detector
cap = cv.VideoCapture(0)
detector_main = htm.HandDetector(max_num_hands=1, min_tracking_confidence=0.7)
previous_time = 0

OFFSET = 15
BACKGROUND_SIZE = (300, 300)

if not cap.isOpened():
    print("Error: Could not open the camera.")
else:
    while True:
        success, frame = cap.read()
        if not success:
            print("Failed to read frame")
            break
            
        frame = cv.flip(frame, 1)  # Mirror the frame
        handsInfo, frame = detector_main.findHands(frame, draw=False)

        if handsInfo:
            hand = handsInfo[0]
            x1, y1, x2, y2 = hand["bbox"]
            
            # Get processed hand image for classification
            handImg = capture_hand_img(frame)
            
            if handImg is not None:
                # Classify the hand sign
                label = classify(handImg)
                
                # Draw bounding box and label
                cv.rectangle(frame, (x1-OFFSET, y1-OFFSET), (x2+OFFSET, y2+OFFSET), (0, 255, 0), 2)
                cv.putText(frame, label, (x1, y1 - 15), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                cv.putText(frame, "Hand processing failed", (x1, y1 - 15), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Calculate and display FPS
        current_time = time.time()
        fps = 1 / (current_time - previous_time) if previous_time > 0 else 0
        previous_time = current_time
        cv.putText(frame, f'FPS: {int(fps)}', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Show the frame
        cv.imshow('Sign Language Live Detection', frame)
        
        # Exit on ESC key
        key = cv.waitKey(1) & 0xFF
        if key == 27:  # ESC key
            break

cap.release()
cv.destroyAllWindows()