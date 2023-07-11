# https://google.github.io/mediapipe/solutions/selfie_segmentation.html
# https://www.youtube.com/watch?v=IZEkwUJ6QGQ
# NOTE: MediaPipe Selfie Segmentation - The intended use cases include selfie 
# effects and video conferencing, where the person is close (< 2m) to the camera.

# imports
import numpy as np
import cv2
import mediapipe as mp
import time
print("hello")

#setup
mp_drawing = mp.solutions.drawing_utils
mp_selfie_segmentation = mp.solutions.selfie_segmentation

#webcam input

cap = cv2.VideoCapture("resources/nicole-dance.mp4")
with mp_selfie_segmentation.SelfieSegmentation(
    model_selection=0) as selfie_segmentation: # model_selection = 0 for general or 1 for landscape
    # From landscape to general, the accuracy increases while the inference speed decreases.
  bg_image = None
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
    #start = time.time()
      # If loading a video, use 'break' instead of 'continue'.
      #continue
    

    image = cv2.resize(image, (720, 720))
    # Flip the image horizontally for a later selfie-view display, and convert
    # the BGR image to RGB.
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    results = selfie_segmentation.process(image)
    
    '''  FOR THE BINARY MASK '''
    cv2.imshow('Segmentation Mask', results.segmentation_mask)

    # Draw the pose annotation on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Draw selfie segmentation on the background image.
    # To improve segmentation around boundaries, consider applying a joint
    # bilateral filter to "results.segmentation_mask" with "image".
    condition = np.stack(
      (results.segmentation_mask,) * 3, axis=-1) > 0.99    # we can change this value
    # The background can be customized.
    #   a) Set a color for the BG
    #   b) Load an image (with the same width and height of the input image) to
    #      be the background, e.g., bg_image = cv2.imread('/path/to/image/file')
    #   c) Blur the input image by applying image filtering, e.g.,
    #      bg_image = cv2.GaussianBlur(image,(55,55),0)
    BG_COLOR = (192, 192, 192) # gray
    if bg_image is None:
      #bg_image = cv2.imread('spongebob.jpg')
      #bg_image = cv2.resize(bg_image, (640, 480))
      bg_image = np.zeros(image.shape, dtype=np.uint8)
      bg_image[:] = BG_COLOR

    output_image = np.where(condition, image, bg_image)

    #saved_images = np.stack((image, condition * 255))
    #end = time.time()
    #totalTime = end - start

    #fps = 1 / totalTime
    #print("FPS: ", fps)

    '''  FOR THE GREEN SCREEN '''
    #v2.imshow('MediaPipe Selfie Segmentation', output_image)

    #cv2.imwrite("results/saved_masks.mp4", condition)
    #output_image.save('results/saved_masks.mp4', fps=30, extra_args=['-vcodec', 'libx264'], dpi=300)

    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()