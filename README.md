# Ball Tracking and Bounce Detection System

This project is a computer vision application to track a ball's trajectory and detect its bounces relative to the boundaries of a sports field. The system is implemented in Python and utilizes OpenCV, numpy, and imutils for image processing, with real-time video input handling.

## Features
- **Real-Time Ball Tracking**: Detects and tracks a green ball in video streams using HSV color space.
- **Predictive Tracking**: Employs a Kalman filter to estimate the ball's position, enhancing the stability of tracking.
- **Dynamic Frame Adjustments**: Resizes frames adaptively based on the ball's distance from the center.
- **Contour and Shape Analysis**: Uses contour approximation to ensure accurate detection of circular objects.
- **Customizable Parameters**: Allows fine-tuning of HSV color ranges, tracking buffer size, and contour thresholds.

- In the project there is also a test-video used to see how it works not in webcam.

## Prerequisites
- Python 3.6+
- OpenCV
- numpy
- imutils
