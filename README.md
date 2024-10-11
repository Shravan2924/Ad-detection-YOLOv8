# YOLOv8 Advertisement Detection Application

## Overview

This repository contains a Python application that utilizes the YOLOv8 (You Only Look Once version 8) model to detect billboards in video streams and overlay selected advertisement images. The application features a user-friendly graphical interface that allows users to select input files, including the advertisement image, video, and trained YOLO model.

## Features

- Select advertisement image, video file, and YOLO model via a graphical user interface (GUI).
- Detect billboard areas in videos using YOLOv8.
- Overlay selected advertisement images onto detected billboards.
- Option to save the output video with the advertisement overlay.

## Prerequisites

### Software Requirements

- Python 3.x
- pip (Python package manager)

### Required Libraries

The application depends on several Python libraries. The key libraries required are:

- OpenCV (`opencv-python`): For video and image processing.
- NumPy: For numerical operations.
- Ultralytics: For the YOLOv8 model implementation.
- Tkinter: For creating the graphical user interface (GUI).

## Installation Instructions

1. **Clone the repository**:

   ```bash
   git clone https://github.com/yourusername/yolo-advertisement-detection.git
   cd yolo-advertisement-detection
