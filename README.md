# Project_IoT_HAOS
This project focuses on building an IoT gateway based on Home Assistant OS with integrated AI for image processing
The system combines IoT device management, real-time communication, and computer vision to enhance smart monitoring and automation capabilities.
The gateway is designed to receive data from IoT devices (such as ESP32-based cameras and sensors), process visual data using AI models, and provide intelligent responses such as detection, tracking, and notifications.
# Objectives
- Build a centralized IoT Gateway using Home Assistant OS
- Integrate AI-based image processing into IoT systems
- Perform real-time person detection and tracking from camera streams
- Communicate between devices and server using MQTT
- Provide monitoring and automation through the Home Assistant dashboard
# System Architecture
The system consists of the following main components:
1. IoT Devices
- ESP32 / ESP32-CAM
- Sensors and camera modules
- Send data and video streams via MQTT or HTTP
2. IoT Gateway (Raspberry Pi)
- Runs Home Assistant OS
- Acts as MQTT Broker and automation controller
- Integrates AI services for image processing
3. AI Image Processing Module
- Person detection using deep learning models (e.g. YOLO)
- Optional face detection / recognition
- Object tracking and bounding box visualization
4. User Interface
- Home Assistant Web Dashboard
- Real-time visualization of camera streams and detection results
- Notifications and automation rules
# AI Features
- Person Detection: Detect humans in camera streams using AI models
- Object Tracking: Track detected persons across video frames
- Bounding Box Visualization: Draw bounding boxes and labels on detected objects
- Event-Based Notifications: Send MQTT or Home Assistant notifications when a person is detected
# Technologies Used
1. Hardware
- Raspberry Pi (IoT Gateway)
- ESP32
- Camera modules and sensors
2. Software
- Home Assistant OS (HAOS)
- PyCharm
- Deep Learning Models (YOLO, optional FaceNet/MTCNN)
- MQTT (Mosquitto Broker)
