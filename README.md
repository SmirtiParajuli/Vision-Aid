# Vision-Aid
A Real-Time Object Detection System for Visually Impaired Individuals.

# ** Overview **
Vision Aid is an AI-powered assistive technology designed to help visually impaired individuals navigate outdoor environments safely. Utilizing the YOLOv8 object detection model, Vision Aid identifies and classifies various objects such as vehicles, pedestrians, and obstacles, providing real-time auditory feedback to enhance situational awareness. This system aims to improve mobility and independence by offering a scalable, cost-effective alternative to traditional mobility aids.

# ** Features **
    - Real-Time Object Detection: Utilizes YOLOv8 for detecting objects in live video streams.
    - Auditory Feedback: Provides verbal descriptions of detected objects to assist visually impaired users.
    - Voice Control Module: Enables hands-free interaction with commands such as "Start detection," "Stop detection," and "Settings."
    - Customizable GUI: Includes a high-contrast, user-friendly interface for partial vision users.
    - Proximity Alerts & Object Prioritization: Detects the proximity of objects and prioritizes critical hazards.
    - Adaptability to Different Environments: Works in various outdoor conditions, including streets, sidewalks, and intersections.

# ** Dataset **
  Vision Aid is trained on the 26 Class Object Detection Dataset, which includes 38,952 images annotated across 26 object classes, such as:
  
        - Bench, Bicycle, Branch, Bus, Bushes, Car, Crosswalk, Door, Elevator, Fire Hydrant, Green Light, Motorcycle, Person, Pothole, Rat, Red Light, Scooter, Stairs, Stop Sign, Traffic Cone, Train, Tree, Track, Umbrella, and Yellow Light.
        - The dataset has been pre-processed using Roboflow, including resizing, noise augmentation, and dataset structuring.

# ** System Requirements **

  Hardware
    
        - Processor: Intel i5/i7 or equivalent AMD processor.
        - RAM: 8GB minimum (16GB recommended).
        - GPU: NVIDIA GPU with CUDA support (Recommended: RTX 2060 or higher).
        - Camera and Microphone: USB or built-in webcam.
        
  Software
  
        - Operating System: Windows 10/11, Linux, or macOS.
        - Python Version: 3.10.11
        - CUDA Version: 12.5 (if using GPU acceleration)
        

# ** Installation & Setup **

   Step 1: Clone the Repository
     
        git clone https://github.com/yourusername/VisionAid.git
        cd VisionAid
        
   Step 2: Create a Virtual Environment
 
        python -m venv visionaid_env
        source visionaid_env/bin/activate  # For Linux/macOS
        visionaid_env\Scripts\activate     # For Windows
        
   Step 3: Install Dependencies

    pip install -r requirements.txt

   Step 4: Download and Set Up the Dataset  
   
     - Download the 26 Class Object Detection Dataset from Kaggle.
     - Extract and place it in the data/ directory inside the project folder.
     - Ensure the dataset follows the YOLO format with train, val, and test folders.


   Step 5: Run the Vision Aid Application
   
          python visionAidGUI.py
          
# ** How to Use Vision Aid **

    Launch the Application
     - Run visionAidGUI.py to start the object detection system.
     - Enable voice control
     - Say commands like "Start detection" or "Stop detection" to interact hands-free.
     
    GUI Interaction
     - Access settings through the high-contrast GUI to customize detection sensitivity and object preferences.
    Real-Time Feedback
    - Detected objects will be announced with proximity alerts to aid navigation.

# ** Model Training & Fine-Tuning **

  To retrain the YOLOv8 model on a custom dataset:
  
       python train.py --data data.yaml --epochs 50 --batch-size 16 --img 640

# ** Troubleshooting ** 

   - Issue: Model not detecting objects correctly

           - Ensure the dataset is correctly formatted and paths are set in data.yaml.
           - Increase the number of training epochs.

   - Issue: Voice commands not working

          - Ensure the microphone is enabled and configured properly.

   - Issue: Slow detection speed

          - Use a higher-end GPU or reduce the input image size in the config.py file.

# ** Future Enhancements ** 

    - Mobile Application Version: Develop a smartphone-compatible version.
    - Improved Dataset: Expand dataset diversity and balance class distributions.
    - Multi-Sensor Integration: Incorporate LiDAR or thermal imaging for low-light detection.
    - Enhanced Voice Command Set: Introduce additional commands for improved user experience.

# ** Contributors **

    - Smriti Parajuli  - Object Detection & Model Training
    - Gloria Hawkins-Roberts  - GUI & Voice Control Development
    
# ** References **

  For detailed references and literature review, please check the full PBT205-Final Prototype Report included in this repository.
  

    Note: If you use Vision Aid for academic or research purposes, kindly cite this project.
  

    

      

