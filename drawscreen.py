import cv2
import mediapipe as mp
import numpy as np
import math

# Initialize Mediapipe drawing utilities
mp_drawing = mp.solutions.drawing_utils

# Initialize Mediapipe hands detection
mp_hands = mp.solutions.hands

# Initialize OpenCV window and mask layer
cv2.namedWindow('Hand Tracking')
mask_layer = 255 * np.ones((720, 1280, 3), dtype=np.uint8)

# Initialize finger tip coordinates
finger_tip = None

# Start video capture
cap = cv2.VideoCapture(0)

# Initialize hands detection
with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:

  while True:
    # Read frame from video capture
    success, image = cap.read()

    if not success:
      break

    # Flip image horizontally for selfie view
    image = cv2.flip(image, 1)

    # Convert BGR image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process image with hands detection
    results = hands.process(image_rgb)

    # Draw hands on image
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=hand_landmarks,
            connections=mp_hands.HAND_CONNECTIONS,
            landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=1, circle_radius=1),
            connection_drawing_spec=mp_drawing.DrawingSpec(color=(0,255,0), thickness=1)
        )

        # Get thumb and index finger tip coordinates
        thumb_tip = None
        index_tip = None
        for id, lm in enumerate(hand_landmarks.landmark):
          if id == 4:
            thumb_tip = (int(lm.x * image.shape[1]), int(lm.y * image.shape[0]))
          elif id == 8:
            index_tip = (int(lm.x * image.shape[1]), int(lm.y * image.shape[0]))

        # Draw on mask layer
        if thumb_tip and index_tip:
          # Calculate distance between thumb and index finger
          distance = math.sqrt((thumb_tip[0] - index_tip[0])**2 + (thumb_tip[1] - index_tip[1])**2)
          if distance < 30:
            cv2.circle(mask_layer, index_tip, radius=10, color=(0,0,255), thickness=-1)

            # Clear the canvas if the distance is greater than 100
          elif distance > 170:
            mask_layer = 255 * np.ones((720, 1280, 3), dtype=np.uint8)
    
    
    # Apply mask layer to image
    image_masked = cv2.bitwise_and(image, mask_layer)

    # Show image in OpenCV window
    cv2.imshow('Hand Tracking', image_masked)

    # Check for key press to exit
    if cv2.waitKey(5) & 0xFF == 27:
      break

# Release resources
cap.release()
cv2.destroyAllWindows()


