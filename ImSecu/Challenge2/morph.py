
import cv2
import numpy as np
import scipy.spatial as spatial
import dlib

def weighted_average_points(start_points, end_points, percent=0.5):
  """ Weighted average of two sets of supplied points

  :param start_points: *m* x 2 array of start face points.
  :param end_points: *m* x 2 array of end face points.
  :param percent: [0, 1] percentage weight on start_points
  :returns: *m* x 2 array of weighted average points
  """
  if percent <= 0:
    return end_points
  elif percent >= 1:
    return start_points
  else:
    return np.asarray(start_points*percent + end_points*(1-percent), np.int32)

def bilinear_interpolate(img, coords):
  """ Interpolates over every image channel
  http://en.wikipedia.org/wiki/Bilinear_interpolation

  :param img: max 3 channel image
  :param coords: 2 x _m_ array. 1st row = xcoords, 2nd row = ycoords
  :returns: array of interpolated pixels with same shape as coords
  """
  int_coords = np.int32(coords)
  x0, y0 = int_coords
  dx, dy = coords - int_coords

  # 4 Neighour pixels
  q11 = img[y0, x0]
  q21 = img[y0, x0+1]
  q12 = img[y0+1, x0]
  q22 = img[y0+1, x0+1]

  btm = q21.T * dx + q11.T * (1 - dx)
  top = q22.T * dx + q12.T * (1 - dx)
  inter_pixel = top * dy + btm * (1 - dy)

  return inter_pixel.T
  
def grid_coordinates(points):
  """ x,y grid coordinates within the ROI of supplied points

  :param points: points to generate grid coordinates
  :returns: array of (x, y) coordinates
  """
  xmin = np.min(points[:, 0])
  xmax = np.max(points[:, 0]) + 1
  ymin = np.min(points[:, 1])
  ymax = np.max(points[:, 1]) + 1
  return np.asarray([(x, y) for y in range(ymin, ymax)
                     for x in range(xmin, xmax)], np.uint32)
def process_warp(src_img, result_img, tri_affines, dst_points, delaunay):
  """
  Warp each triangle from the src_image only within the
  ROI of the destination image (points in dst_points).
  """
  roi_coords = grid_coordinates(dst_points)
  # indices to vertices. -1 if pixel is not in any triangle
  roi_tri_indices = delaunay.find_simplex(roi_coords)

  for simplex_index in range(len(delaunay.simplices)):
    coords = roi_coords[roi_tri_indices == simplex_index]
    num_coords = len(coords)
    out_coords = np.dot(tri_affines[simplex_index],
                        np.vstack((coords.T, np.ones(num_coords))))
    x, y = coords.T
    result_img[y, x] = bilinear_interpolate(src_img, out_coords)

  return None

def triangular_affine_matrices(vertices, src_points, dest_points):
  """
  Calculate the affine transformation matrix for each
  triangle (x,y) vertex from dest_points to src_points

  :param vertices: array of triplet indices to corners of triangle
  :param src_points: array of [x, y] points to landmarks for source image
  :param dest_points: array of [x, y] points to landmarks for destination image
  :returns: 2 x 3 affine matrix transformation for a triangle
  """
  ones = [1, 1, 1]
  for tri_indices in vertices:
    src_tri = np.vstack((src_points[tri_indices, :].T, ones))
    dst_tri = np.vstack((dest_points[tri_indices, :].T, ones))
    mat = np.dot(src_tri, np.linalg.inv(dst_tri))[:2, :]
    yield mat

def warp_image(src_img, src_points, dest_points, dest_shape, dtype=np.uint8):
  # Resultant image will not have an alpha channel
  num_chans = 3
  src_img = src_img[:, :, :3]

  rows, cols = dest_shape[:2]
  result_img = np.zeros((rows, cols, num_chans), dtype)

  delaunay = spatial.Delaunay(dest_points)
  tri_affines = np.asarray(list(triangular_affine_matrices(
    delaunay.simplices, src_points, dest_points)))

  process_warp(src_img, result_img, tri_affines, dest_points, delaunay)

  return result_img

def positive_cap(num):
  """ Cap a number to ensure positivity

  :param num: positive or negative number
  :returns: (overflow, capped_number)
  """
  if num < 0:
    return 0, abs(num)
  else:
    return num, 0

def roi_coordinates(rect, size, scale):
  """ Align the rectangle into the center and return the top-left coordinates
  within the new size. If rect is smaller, we add borders.

  :param rect: (x, y, w, h) bounding rectangle of the face
  :param size: (width, height) are the desired dimensions
  :param scale: scaling factor of the rectangle to be resized
  :returns: 4 numbers. Top-left coordinates of the aligned ROI.
    (x, y, border_x, border_y). All values are > 0.
  """
  rectx, recty, rectw, recth = rect
  new_height, new_width = size
  mid_x = int((rectx + rectw/2) * scale)
  mid_y = int((recty + recth/2) * scale)
  roi_x = mid_x - int(new_width/2)
  roi_y = mid_y - int(new_height/2)

  roi_x, border_x = positive_cap(roi_x)
  roi_y, border_y = positive_cap(roi_y)
  return roi_x, roi_y, border_x, border_y
def resize_image(img, scale):
  """ Resize image with the provided scaling factor

  :param img: image to be resized
  :param scale: scaling factor for resizing the image
  """
  cur_height, cur_width = img.shape[:2]
  new_scaled_height = int(scale * cur_height)
  new_scaled_width = int(scale * cur_width)

  return cv2.resize(img, (new_scaled_width, new_scaled_height))
def scaling_factor(rect, size):
  """ Calculate the scaling factor for the current image to be
      resized to the new dimensions

  :param rect: (x, y, w, h) bounding rectangle of the face
  :param size: (width, height) are the desired dimensions
  :returns: floating point scaling factor
  """
  new_height, new_width = size
  rect_h, rect_w = rect[2:]
  height_ratio = rect_h / new_height
  width_ratio = rect_w / new_width
  scale = 1
  if height_ratio > width_ratio:
    new_recth = 0.8 * new_height
    scale = new_recth / rect_h
  else:
    new_rectw = 0.8 * new_width
    scale = new_rectw / rect_w
  return scale
def boundary_points(points, width_percent=0.1, height_percent=0.1):
  """ Produce additional boundary points
  :param points: *m* x 2 array of x,y points
  :param width_percent: [-1, 1] percentage of width to taper inwards. Negative for opposite direction
  :param height_percent: [-1, 1] percentage of height to taper downwards. Negative for opposite direction
  :returns: 2 additional points at the top corners
  """
  x, y, w, h = cv2.boundingRect(np.array([points], np.int32))
  spacerw = int(w * width_percent)
  spacerh = int(h * height_percent)
  return [[x+spacerw, y+spacerh],
          [x+w-spacerw, y+spacerh]]
          
def face_points_dlib(img,dlib_predictor, add_boundary_points=True):
  """ Locates 68 face points using dlib (http://dlib.net)
    Requires shape_predictor_68_face_landmarks.dat to be in face_morpher/data
    Download at: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
  :param img: an image array
  :param add_boundary_points: bool to add additional boundary points
  :returns: Array of x,y face points. Empty array if no face found
  """
  try:
    dlib_detector = dlib.get_frontal_face_detector()
    points = []
    rgbimg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rects = dlib_detector(rgbimg, 1)

    if rects and len(rects) > 0:
      # We only take the first found face
      shapes = dlib_predictor(rgbimg, rects[0])
      points = np.array([(shapes.part(i).x, shapes.part(i).y) for i in range(68)], np.int32)

      if add_boundary_points:
        # Add more points inwards and upwards as dlib only detects up to eyebrows
        points = np.vstack([
          points,
          boundary_points(points, 0.1, -0.03),
          boundary_points(points, 0.13, -0.05),
          boundary_points(points, 0.15, -0.08),
          boundary_points(points, 0.33, -0.12)])

    return points
  except Exception as e:
    print(e)
    return []

def resize_align(img, points, size):
  """ Resize image and associated points, align face to the center
    and crop to the desired size

  :param img: image to be resized
  :param points: *m* x 2 array of points
  :param size: (height, width) tuple of new desired size
  """
  new_height, new_width = size

  # Resize image based on bounding rectangle
  rect = cv2.boundingRect(np.array([points], np.int32))
  scale = scaling_factor(rect, size)
  img = resize_image(img, scale)

  # Align bounding rect to center
  cur_height, cur_width = img.shape[:2]
  roi_x, roi_y, border_x, border_y = roi_coordinates(rect, size, scale)
  roi_h = np.min([new_height-border_y, cur_height-roi_y])
  roi_w = np.min([new_width-border_x, cur_width-roi_x])

  # Crop to supplied size
  crop = np.zeros((new_height, new_width, 3), img.dtype)
  crop[border_y:border_y+roi_h, border_x:border_x+roi_w] = (
     img[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w])

  # Scale and align face points to the crop
  points[:, 0] = (points[:, 0] * scale) + (border_x - roi_x)
  points[:, 1] = (points[:, 1] * scale) + (border_y - roi_y)

  return (crop, points)

def load_image_points(img, size, dlib_predictor ):
  #img = cv2.imread(path)
  points =face_points_dlib(img, dlib_predictor, add_boundary_points=True)

  if len(points) == 0:
    print('No face detected')
    return None, None
  else:
    return resize_align(img, points, size)

def weighted_average(img1, img2, percent=0.5):
  if percent <= 0:
    return img2
  elif percent >= 1:
    return img1
  else:
    return cv2.addWeighted(img1, percent, img2, 1-percent, 0)

def overlay_image(foreground_image, mask, background_image):
  """ Overlay foreground image onto the background given a mask
  :param foreground_image: foreground image points
  :param mask: [0-255] values in mask
  :param background_image: background image points
  :returns: image with foreground where mask > 0 overlaid on background image
  """
  foreground_pixels = mask > 0
  background_image[..., :3][foreground_pixels] = foreground_image[..., :3][foreground_pixels]
  return background_image

def mask_from_points(size, points):
  """ Create a mask of supplied size from supplied points
  :param size: tuple of output mask size
  :param points: array of [x, y] points
  :returns: mask of values 0 and 255 where
            255 indicates the convex hull containing the points
  """
  radius = 10  # kernel size
  kernel = np.ones((radius, radius), np.uint8)

  mask = np.zeros(size, np.uint8)
  cv2.fillConvexPoly(mask, cv2.convexHull(points), 255)
  mask = cv2.erode(mask, kernel)

  return mask

