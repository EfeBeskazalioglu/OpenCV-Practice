# OpenCV Practice

A collection of hands-on OpenCV exercises covering core computer vision techniques.

## Scripts

| File | Description |
|------|-------------|
| [numpy_array_drawing.py](numpy_array_drawing.py) | Create a blank NumPy image array and draw a colored rectangle using array slicing |
| [drawing_shapes_on_image.py](drawing_shapes_on_image.py) | Draw lines, circles, rectangles, and text on a photo (`puppy.jpg`) |
| [green_color_detection_hsv.py](green_color_detection_hsv.py) | Detect green regions in `green_apple.jpg` using HSV color space and `cv2.inRange` masking |
| [thresholding.py](thresholding.py) | Compare simple, Otsu's, adaptive mean, and adaptive Gaussian thresholding on a book page scan |
| [canny.py](canny.py) | Explore Canny edge detection with varying thresholds, Gaussian blur pre-processing, and auto threshold selection via median pixel value |
| [morphology_operations.py](morphology_operations.py) | Apply morphological closing and opening to clean up a green-color detection mask |
| [contour.py](contour.py) | Detect and count coins in `coins.jpg` using adaptive thresholding, morphology, and `cv2.findContours` |
| [video.py](video.py) | Real-time moving object detection on `samples_data_vtest.avi` using MOG2 background subtraction, morphology cleanup, and bounding-box drawing |

## Sample Images / Video

| File | Used by |
|------|---------|
| `puppy.jpg` | drawing_shapes_on_image.py |
| `green_apple.jpg` | green_color_detection_hsv.py, morphology_operations.py |
| `book page.jpg` | thresholding.py |
| `car.jpg` | canny.py |
| `coins.jpg` | contour.py |
| `samples_data_vtest.avi` | video.py |

## Requirements

```
opencv-python
numpy
matplotlib
```

Install with:

```bash
pip install opencv-python numpy matplotlib
```

## Topics Covered

- NumPy array manipulation for image creation
- Drawing primitives (lines, circles, rectangles, text)
- Color space conversion (BGR → RGB, BGR → HSV)
- Color-based segmentation with HSV masking
- Image thresholding (simple, Otsu, adaptive)
- Edge detection (Canny) and Gaussian blur
- Morphological operations (erosion, dilation, opening, closing)
- Contour detection and object counting
- Video capture and background subtraction (MOG2)
