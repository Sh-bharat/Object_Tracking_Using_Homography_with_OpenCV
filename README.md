---

# Object Tracking Using Homography with OpenCV

This project demonstrates object tracking in real-time video using homography and feature matching techniques. It employs the Scale-Invariant Feature Transform (SIFT) for detecting and describing features, and the Fast Library for Approximate Nearest Neighbors (FLANN) for matching these features between a base image and live camera frames.

## Requirements

- Python 3.x
- OpenCV (cv2)
- Numpy

Ensure you have a compatible webcam and a base image file saved as `base.jpg`.

## Usage

1. **Run the Script**   

2. **Exit**:
    Press the `ESC` key to stop the video feed and close the application.

## Explanation

The script performs the following steps:

1. **Initialize Detectors**:
    - SIFT is used for feature detection.
    - FLANN is used for feature matching.

2. **Live Capture**:
    - The script captures frames from a webcam in real-time.

3. **Feature Matching**:
    - Keypoints and descriptors are detected in the base image and each video frame.
    - Matches are identified using FLANN.

4. **Homography Calculation**:
    - Homography matrix is computed if sufficient matches are found.
    - This matrix is used to outline the detected object.

5. **Display**:
    - Frames are displayed with the detected object's outline if homography is successfully calculated.

### Script

```python
import cv2
import numpy as np

# Initialize the SIFT detector
sift = cv2.SIFT_create()

# Initialize the FLANN matcher
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)

# Load base image and find its keypoints and descriptors
base_image_path = 'base.jpg'  # replace with your base image path
base = cv2.imread(base_image_path, cv2.IMREAD_GRAYSCALE)
base_kp, base_desc = sift.detectAndCompute(base, None)

# Capture video from the webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert frame to grayscale
    query_grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect keypoints and descriptors in the query frame
    query_kp, query_desc = sift.detectAndCompute(query_grayframe, None)
    
    # Match descriptors using FLANN matcher
    matches = flann.knnMatch(base_desc, query_desc, k=2)
    
    # Filter good matches
    good_points = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_points.append(m)
    
    # Draw matches and compute homography if there are enough good matches
    if len(good_points) > 10:
        # Extract location of good matches
        base_pts = np.float32([base_kp[m.queryIdx].pt for m in good_points]).reshape(-1, 1, 2)
        query_pts = np.float32([query_kp[m.trainIdx].pt for m in good_points]).reshape(-1, 1, 2)
        
        # Find homography
        matrix, mask = cv2.findHomography(base_pts, query_pts, cv2.RANSAC, 5.0)
        if matrix is not None:
            # Get dimensions of base image
            h, w = base.shape
            
            # Define points to draw the homography
            pts = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
            dst = cv2.perspectiveTransform(pts, matrix)
            
            # Draw the homography as a polygon
            homography = cv2.polylines(frame, [np.int32(dst)], True, (255, 0, 0), 3)
            cv2.imshow("Homography", homography)
        else:
            cv2.imshow("Homography", frame)
    else:
        cv2.imshow("Homography", frame)
    
    # Break the loop on 'ESC' key press
    if cv2.waitKey(1) == 27:
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
```

## Output
![Detected Object](https://github.com/Sh-bharat/Object_Tracking_Using_Homography_with_OpenCV/assets/133742551/4bbfa32a-314c-447c-ad26-0aa8832a338e)
<br><hr><br>
![Screenshot 2024-06-17 144155](https://github.com/Sh-bharat/Object_Tracking_Using_Homography_with_OpenCV/assets/133742551/b64a0543-0cec-4b9c-a0e4-8fbd3d27bc8b)



## Troubleshooting

- **No Matches Found**:
  - Ensure that the base image has distinct and detectable features.
  - Adjust the matching ratio or RANSAC parameters.

- **Camera Issues**:
  - Verify that your webcam is properly connected and accessible.

- **Performance**:
  - If the video is lagging, consider reducing the frame resolution or optimizing the matching parameters.

## References

- [OpenCV Documentation](https://docs.opencv.org/)
- [Feature Matching with FLANN](https://docs.opencv.org/master/dc/dc3/tutorial_py_matcher.html)
- [SIFT Algorithm](https://en.wikipedia.org/wiki/Scale-invariant_feature_transform)


---
