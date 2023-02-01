import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
from matplotlib import pyplot as plt
import cv2
from google.colab.patches import cv2_imshow
import math
import mediapipe as mp
from matplotlib.collections import LineCollection
import matplotlib.patches as patches
import imageio
from IPython.display import HTML, display


#@title Helper functions for visualization



def _keypoints_and_edges_for_display(keypoints_with_scores,
                                     height,
                                     width,
                                     keypoint_threshold=0.11):
  """Returns high confidence keypoints and edges for visualization.

  Args:
    keypoints_with_scores: A numpy array with shape [1, 1, 17, 3] representing
      the keypoint coordinates and scores returned from the MoveNet model.
    height: height of the image in pixels.
    width: width of the image in pixels.
    keypoint_threshold: minimum confidence score for a keypoint to be
      visualized.

  Returns:
    A (keypoints_xy, edges_xy, edge_colors) containing:
      * the coordinates of all keypoints of all detected entities;
      * the coordinates of all skeleton edges of all detected entities;
      * the colors in which the edges should be plotted.
  """
  keypoints_all = []
  keypoint_edges_all = []
  edge_colors = []
  num_instances, _, _, _ = keypoints_with_scores.shape
  for idx in range(num_instances):
    kpts_x = keypoints_with_scores[0, idx, :, 1]
    kpts_y = keypoints_with_scores[0, idx, :, 0]
    kpts_scores = keypoints_with_scores[0, idx, :, 2]
    kpts_absolute_xy = np.stack(
        [width * np.array(kpts_x), height * np.array(kpts_y)], axis=-1)
    kpts_above_thresh_absolute = kpts_absolute_xy[
        kpts_scores > keypoint_threshold, :]
    keypoints_all.append(kpts_above_thresh_absolute)

    for edge_pair, color in KEYPOINT_EDGE_INDS_TO_COLOR.items():
      if (kpts_scores[edge_pair[0]] > keypoint_threshold and
          kpts_scores[edge_pair[1]] > keypoint_threshold):
        x_start = kpts_absolute_xy[edge_pair[0], 0]
        y_start = kpts_absolute_xy[edge_pair[0], 1]
        x_end = kpts_absolute_xy[edge_pair[1], 0]
        y_end = kpts_absolute_xy[edge_pair[1], 1]
        line_seg = np.array([[x_start, y_start], [x_end, y_end]])
        keypoint_edges_all.append(line_seg)
        edge_colors.append(color)
  if keypoints_all:
    keypoints_xy = np.concatenate(keypoints_all, axis=0)
  else:
    keypoints_xy = np.zeros((0, 17, 2))

  if keypoint_edges_all:
    edges_xy = np.stack(keypoint_edges_all, axis=0)
  else:
    edges_xy = np.zeros((0, 2, 2))
  return keypoints_xy, edges_xy, edge_colors


def draw_prediction_on_image(
    image, keypoints_with_scores, crop_region=None, close_figure=False,
    output_image_height=None):
  """Draws the keypoint predictions on image.

  Args:
    image: A numpy array with shape [height, width, channel] representing the
      pixel values of the input image.
    keypoints_with_scores: A numpy array with shape [1, 1, 17, 3] representing
      the keypoint coordinates and scores returned from the MoveNet model.
    crop_region: A dictionary that defines the coordinates of the bounding box
      of the crop region in normalized coordinates (see the init_crop_region
      function below for more detail). If provided, this function will also
      draw the bounding box on the image.
    output_image_height: An integer indicating the height of the output image.
      Note that the image aspect ratio will be the same as the input image.

  Returns:
    A numpy array with shape [out_height, out_width, channel] representing the
    image overlaid with keypoint predictions.
  """
  height, width, channel = image.shape
  aspect_ratio = float(width) / height
  fig, ax = plt.subplots(figsize=(12 * aspect_ratio, 12))
  # To remove the huge white borders
  fig.tight_layout(pad=0)
  ax.margins(0)
  ax.set_yticklabels([])
  ax.set_xticklabels([])
  plt.axis('off')

  im = ax.imshow(image)
  line_segments = LineCollection([], linewidths=(4), linestyle='solid')
  ax.add_collection(line_segments)
  # Turn off tick labels
  scat = ax.scatter([], [], s=60, color='#FF1493', zorder=3)

  (keypoint_locs, keypoint_edges,
   edge_colors) = _keypoints_and_edges_for_display(
       keypoints_with_scores, height, width)

  line_segments.set_segments(keypoint_edges)
  line_segments.set_color(edge_colors)
  if keypoint_edges.shape[0]:
    line_segments.set_segments(keypoint_edges)
    line_segments.set_color(edge_colors)
  if keypoint_locs.shape[0]:
    scat.set_offsets(keypoint_locs)

  if crop_region is not None:
    xmin = max(crop_region['x_min'] * width, 0.0)
    ymin = max(crop_region['y_min'] * height, 0.0)
    rec_width = min(crop_region['x_max'], 0.99) * width - xmin
    rec_height = min(crop_region['y_max'], 0.99) * height - ymin
    rect = patches.Rectangle(
        (xmin,ymin),rec_width,rec_height,
        linewidth=1,edgecolor='b',facecolor='none')
    ax.add_patch(rect)

  fig.canvas.draw()
  image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
  image_from_plot = image_from_plot.reshape(
      fig.canvas.get_width_height()[::-1] + (3,))
  plt.close(fig)
  if output_image_height is not None:
    output_image_width = int(output_image_height / height * width)
    image_from_plot = cv2.resize(
        image_from_plot, dsize=(output_image_width, output_image_height),
         interpolation=cv2.INTER_CUBIC)
  return image_from_plot


def progress(value, max=100):
  return HTML("""
      <progress
          value='{value}'
          max='{max}',
          style='width: 100%'
      >
          {value}
      </progress>
  """.format(value=value, max=max))


def movenet(input_image, interpreter):
    """Runs detection on an input image.

    Args:
      input_image: A [1, height, width, 3] tensor represents the input image
        pixels. Note that the height/width should already be resized and match the
        expected input resolution of the model before passing into this function.

    Returns:
      A [1, 1, 17, 3] float numpy array representing the predicted keypoint
      coordinates and scores.
    """
    # TF Lite format expects tensor type of uint8.
    input_image = tf.cast(input_image, dtype=tf.uint8)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], input_image.numpy())
    # Invoke inference.
    interpreter.invoke()
    # Get the model prediction.
    keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])
    return keypoints_with_scores

def main_visualize(image):
  input_size = 192
  # Resize and pad the image to keep the aspect ratio and fit the expected size.
  input_image = tf.expand_dims(image, axis=0)
  input_image = tf.image.resize_with_pad(input_image, input_size, input_size)

  # Run model inference.
  keypoints_with_scores = movenet(input_image)

  # Visualize the predictions with image.
  #display_image = tf.expand_dims(image, axis=0)
  #display_image = tf.cast(tf.image.resize_with_pad(
      #display_image, 1280, 1280), dtype=tf.int32)
  #output_overlay = draw_prediction_on_image(
  #    np.squeeze(display_image.numpy(), axis=0), keypoints_with_scores)

  #plt.figure(figsize=(5, 5))
  #plt.imshow(output_overlay)
  #_ = plt.axis('off')
  return keypoints_with_scores


def video_to_images(video):
  ''' 
  extracts the images of a video and saves them in an dictionary which is returned
  '''
  #reads one frame from video and moves current position of the video to next frame
  success,image = video.read() 
  count = 0
  out = {}
  while success:
    # saves the Frame in a dictionary with the frameNumber as key
    out.update({"frame%d" % count: image})  
    # read next frame  
    success,image = video.read()
    count += 1
  return out


def resize_and_show(image, DESIRED_HEIGHT, DESIRED_WIDTH):
  """
  shows the image for visualisation purposes
  """
  
  h, w = image.shape[:2]
  if h < w:
    img = cv2.resize(image, (DESIRED_WIDTH, math.floor(h/(w/DESIRED_WIDTH))))
  else:
    img = cv2.resize(image, (math.floor(w/(h/DESIRED_HEIGHT)), DESIRED_HEIGHT))
  cv2_imshow(img)

def extract_landmarks(images):
  """
  uses mediapip NN to extract the facial landmarks form an array of images
  returns an array with the MeadiaPipe output for each Image
  """
  mp_face_mesh = mp.solutions.face_mesh
  output = []
  with mp_face_mesh.FaceMesh(
      static_image_mode=True,
      refine_landmarks=True,
      max_num_faces=2,
      min_detection_confidence=0.5) as face_mesh:
    # loop through all images in the input array
    for name, image in images.items():
      # Convert the BGR image to RGB and process it with MediaPipe Face Mesh.
      results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
      output.append(results)
  return output

def draw_landmarks_on_img(images):
  """
  extract the landmarks of an array of images and visualizes how MediaPipe estimated the face mesh
  returns an array with the MeadiaPipe output for each Image
  """
  mp_face_mesh = mp.solutions.face_mesh

  # Load drawing_utils and drawing_styles
  mp_drawing = mp.solutions.drawing_utils 
  mp_drawing_styles = mp.solutions.drawing_styles
  output = []
  # Run MediaPipe Face Mesh.
  with mp_face_mesh.FaceMesh(
      static_image_mode=True,
      refine_landmarks=True,
      max_num_faces=2,
      min_detection_confidence=0.5) as face_mesh:
    for name, image in images.items():
      # Convert the BGR image to RGB and process it with MediaPipe Face Mesh.
      results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
      output.append(results)

      # Draw face landmarks of each face.
      print(f'Face landmarks of {name}:')
      if not results.multi_face_landmarks:
        continue
      annotated_image = image.copy()
      for face_landmarks in results.multi_face_landmarks:
        mp_drawing.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_tesselation_style())
        mp_drawing.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_contours_style())
        mp_drawing.draw_landmarks(
            image=annotated_image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles
            .get_default_face_mesh_iris_connections_style())
      resize_and_show(annotated_image)
  return output

def landmarks_to_df(results):
  """
  transforms the MediaPipe output of one frame into a dataframe that consists of ONE row and 3 columns for each coordinate
  returns the new df
  """
  d = {}
  for i in range(478):
    # catches TypeError in case the Mediapipe output of a coordinate is NaN (couldn't be idetified for the frame)
    try:
      d.update({
          # wites columns x,y,z for each coordinate
          'x_'+str(i): results.multi_face_landmarks[0].landmark[i].x,
          'y_'+str(i): results.multi_face_landmarks[0].landmark[i].y,
          'z_'+str(i): results.multi_face_landmarks[0].landmark[i].z,
      })
    except TypeError:
      # skips this coordinate because it is not usable
      pass
  return pd.DataFrame(d, index=range(1))


def join_image_dfs(input_dfs):
  """
  joins the dfs of every single frame and returns one df for the dfs in the input array (frames of one video)
  """
  # create first df that the others can be appended to
  video_df = landmarks_to_df(input_dfs[0])
  count = 0
  for result in input_dfs:
    # in case the face coudn't be detected in previous images
    if len(video_df.columns) > 5:
      # only append if the input_dfs is not empty
      #if len(video_df) > 5:
      video_df = video_df.append(landmarks_to_df(result))
      #print('append')
    else: 
      video_df = landmarks_to_df(result)
      #print('empty')
    count += 1
  return video_df

def transform_dfs(dfs_list):
  dfs_transformed_list = []
  # groupe coordinates together to get a tuple of 3 for each coordinate
  for df in dfs_list:
    dfs_transformed = pd.DataFrame()
    for i in range(478):
      try:
        dfs_transformed['c_' + str(i)] = list(zip(df['x_' + str(i)], df['y_' + str(i)], df['z_' + str(i)]))
      except KeyError:
        print(i)
    dfs_transformed_list.append(dfs_transformed)
  return dfs_transformed_list

# Dictionary that maps from joint names to keypoint indices.
KEYPOINT_DICT = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16
}

# Maps bones to a matplotlib color name.
KEYPOINT_EDGE_INDS_TO_COLOR = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}