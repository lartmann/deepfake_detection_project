import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

# PREPROCESSING
def video_to_images_array(video):
  ''' 
  extracts the images of a video and saves them in an dictionary which is returned
  '''
  #reads one frame from video and moves current position of the video to next frame
  success,image = video.read() 
  count = 0
  out = []
  while success:
    # saves the Frame in a dictionary with the frameNumber as key
    out.append(image) 
    # read next frame  
    success,image = video.read()
    count += 1
  return out

def get_face_box(IMAGE_FILES):
  mp_face_detection = mp.solutions.face_detection
  mp_drawing = mp.solutions.drawing_utils

  face_boxes = []

  # For static images:

  with mp_face_detection.FaceDetection(
      model_selection=1, min_detection_confidence=0.5) as face_detection:
    for idx, file in enumerate(IMAGE_FILES):
      image = file
      # Convert the BGR image to RGB and process it with MediaPipe Face Detection.
      results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
      face_boxes.append(results.detections)

  return face_boxes

def process_images(imgs, max_h, max_w):
  '''
  Preprocess image including: cropping to face, padding/resize to right shape, and batching
  '''
  
  faces = np.zeros((1, max_h, max_w, 3), dtype=np.float16)
  batch_result = np.zeros((1, 50, max_h, max_w, 3))
  
  # extract face box
  face_boxes = get_face_box(imgs)
  num_no_face = 0

  # crop image to face
  for frame, image in zip(face_boxes, imgs):
    h,w,c = image.shape
    try:
      left = int(frame[0].location_data.relative_bounding_box.xmin*w)
      top = int(frame[0].location_data.relative_bounding_box.ymin*h)
      right = int(frame[0].location_data.relative_bounding_box.width*w + left)
      bottom = int(frame[0].location_data.relative_bounding_box.height*h + top)
    
      frame = frame
      face = Image.fromarray(image).crop((left, top, right, bottom))
      face_np = np.asarray(face, dtype=np.float16)/255.0
      
      # get padding if image is too small
      if face_np.shape[0] < max_h: 
          face_np = np.pad(face_np, ((0,max_h - face_np.shape[0]),(0,0),(0,0)))
      if face_np.shape[1] < max_w:
        face_np = np.pad(face_np, ((0,0),(0,max_w - face_np.shape[1]),(0,0)))
      #if face_np.shape != (max_h,max_w,3):
      # print(face_np.shape)
      
      # resize is image is too big
      if face_np.shape[0] > max_h or face_np.shape[1] > max_w:
        face_np = np.resize(face_np, (max_h,max_w,3))

      faces = np.vstack([faces, np.float16(np.expand_dims(face_np, axis=0))])
    except TypeError:
      num_no_face += 1
  print(f'could not detect the face in {num_no_face} frames')
  # get batches of size 50
  num_batches = int(len(imgs)/50)
  for i in range(num_batches):
    batch = faces[i*50:(i+1)*50]
    if len(batch) == 50:
      batch_result = np.vstack([batch_result, np.expand_dims(batch, axis = 0)])

  return np.float16(batch_result)


def find_max(videos):
  max_h = 0
  max_w = 0
  for v in videos:
    if v.shape[0] > max_h:
      max_h = v.shape[0]
    if v.shape[1] > max_w:
      max_w = v.shape[1]
    '''
    for f in v:
      if len(f) > max_h:
        max_h = len(f)
      if len(f[0]) > max_w:
        max_w = len(f[0])
      '''
  return (max_h, max_w)


def get_padding(videos, max_h, max_w):
  for n, v in enumerate(videos):
    for i, f in enumerate(v):
      if f.shape[0] < max_h:
        videos[n][i] = np.pad(f, ((0,max_h - f.shape[0]),(0,0),(0,0)))
        f = np.pad(f, ((0,max_h - f.shape[0]),(0,0),(0,0)))
      
      if len(f[0]) < max_w:
        videos[n][i] = np.pad(f, ((0,0),(0,max_w - f.shape[1]),(0,0)))
      if videos[n][i].shape != (310,310,3):
        print(videos[n][i].shape)
  return videos


def split_in_batches(results):
  batch_result = np.array([], dtype=np.float16)
  for v in results:
    num_batches = int(len(v)/50)

    for i in range(num_batches):
      batch = v[i*50:(i+1)*50]
      if len(batch) == 50:
        batch_result = np.append(batch_result, batch)
        batch_result.append(batch)
    
  return batch_result

# VISUALIZATION

def plot_loss_curves(history):
  """
  Plots the loss curve and accuracy for a given model
  Args:
    history: the history object returned by the fit method of a Keras model
  """

  loss = history.history["loss"]
  val_loss = history.history["val_loss"]
  accuracy = history.history["accuracy"]
  val_accuracy = history.history["val_accuracy"]
  epochs = range(len(loss))

  # plot loss
  plt.plot(epochs, loss, label="Training loss")
  plt.plot(epochs, val_loss, label="Validation loss")
  plt.title("Loss")
  plt.xlabel("Epochs")
  #plt.ylabel("")
  plt.legend()
  
  # plot accuracy
  plt.plot(epochs, accuracy, label="Training accuracy")
  plt.plot(epochs, val_accuracy, label="Validation accuracy")
  plt.title("Accuracy")
  plt.xlabel("Epochs")
  #plt.ylabel("")
  plt.legend()
  
  plt.show()

  
def plot_loss_curves_2(history):
  """
  Plots the accuracy for a given model
  Args:
    history: the history object returned by the fit method of a Keras model
  """

  accuracy = history.history["accuracy"]
  val_accuracy = history.history["val_accuracy"]
  epochs = range(len(accuracy))
  
  # plot accuracy
  plt.plot(epochs, accuracy, label="Training accuracy")
  plt.plot(epochs, val_accuracy, label="Validation accuracy")
  plt.title("Accuracy")
  plt.xlabel("Epochs")
  #plt.ylabel("")
  plt.legend()
  
  plt.show()
