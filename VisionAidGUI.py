#!/usr/bin/env python
# coding: utf-8
import sys
import cv2
import pyttsx3
import threading
import time
import queue
import speech_recognition as sr
from queue import Empty
from PyQt5.QtCore import QTimer, pyqtSignal, QObject, Qt
from PyQt5.QtGui import QImage, QPixmap, QTextCursor
from PyQt5.QtWidgets import QApplication, QLabel, QTextEdit, QVBoxLayout, QHBoxLayout, QWidget, QPushButton, QCheckBox, QSlider, QScrollArea, QStackedWidget
from ultralytics import YOLO
from PyQt5.QtWidgets import QLabel, QFrame, QVBoxLayout, QCheckBox, QScrollArea, QSlider, QPushButton


# Initialize voice recognition
recognizer = sr.Recognizer()

# Load YOLO model
model_path = 'trainModel_epoc50.pt'
model = YOLO(model_path)

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# Parameters for announcements
CENTER_ZONE_WIDTH_RATIO = 0.4
ANNOUNCEMENT_INTERVAL = 3  # seconds between repeated announcements for the same object

# Initialize video capture
cap = cv2.VideoCapture(0)

# Define fixed size for the video feed
VIDEO_WIDTH = 640
VIDEO_HEIGHT = 480
# Define specific behaviors, reactions, and messages for each object class
object_rules = {
    "bench": {
        "announce_distance": 4, 
        "ignore": True, 
        "approach_only": False, 
        "message": "", 
        "reactions": {"left": "", "center": "", "right": ""}
    },
    "bicycle": {
        "announce_distance": 4, 
        "ignore": False, 
        "approach_only": True,
        "message": "Bicycle approaching.",
        "reactions": {
            "left": "Move to the right to avoid the bicycle.",
            "center": "Stay on the side to avoid the bicycle in front.",
            "right": "Move to the left to avoid the bicycle."
        }
    },
    "branch": {
        "announce_distance": 2, 
        "ignore": False, 
        "approach_only": False,
        "message": "Branch detected.",
        "reactions": {
            "left": "Step around to your right to avoid the branch.",
            "center": "Step over carefully to avoid tripping.",
            "right": "Step around to your left to avoid the branch."
        }
    },
    "bus": {
        "announce_distance": 5, 
        "ignore": False, 
        "approach_only": True,
        "message": "Bus approaching.",
        "reactions": {
            "left": "Stay clear to the right side of the road.",
            "center": "Avoid standing near the road, the bus is ahead.",
            "right": "Stay clear to the left side of the road."
        }
    },
    "bushes": {
        "announce_distance": 3, 
        "ignore": True, 
        "approach_only": False, 
        "message": "", 
        "reactions": {"left": "bush to your left", "center": "slow down or stop, bush in front of you", "right": ""}
    },
    "car": {
        "announce_distance": 4, 
        "ignore": False, 
        "approach_only": True,
        "message": "Car in front.",
        "reactions": {
            "left": "Move to the right, away from the car.",
            "center": "Stay on the sidewalk, the car is in front.",
            "right": "Move to the left, away from the car."
        }
    },
    "crosswalk": {
        "announce_distance": 4, 
        "ignore": False, 
        "approach_only": False,
        "message": "Approaching a crosswalk.",
        "reactions": {
            "left": "Look both ways and use the crosswalk on your left.",
            "center": "Look both ways before crossing straight ahead.",
            "right": "Look both ways and use the crosswalk on your right."
        }
    },
    "door": {
        "announce_distance": 3, 
        "ignore": False, 
        "approach_only": False,
        "message": "Door detected.",
        "reactions": {
            "left": "Use the door handle on your left to enter or exit.",
            "center": "Open the door ahead to proceed.",
            "right": "Use the door handle on your right to enter or exit."
        }
    },
    "elevator": {
        "announce_distance": 1, 
        "ignore": False, 
        "approach_only": False,
        "message": "Elevator nearby.",
        "reactions": {
            "left": "The elevator is on your left; press the button to enter.",
            "center": "The elevator is ahead; press the button to enter.",
            "right": "The elevator is on your right; press the button to enter."
        }
    },
    "fire_hydrant": {
        "announce_distance": 1, 
        "ignore": True, 
        "approach_only": False, 
        "message": "", 
        "reactions": {"left": "", "center": "", "right": ""}
    },
    "green_light": {
        "announce_distance": 2, 
        "ignore": False, 
        "approach_only": False,
        "message": "Green light ahead.",
        "reactions": {
            "left": "You may proceed safely towards the left.",
            "center": "You may proceed straight ahead.",
            "right": "You may proceed safely towards the right."
        }
    },
    "gun": {
        "announce_distance": 5, 
        "ignore": False, 
        "approach_only": True,
        "message": "Warning: Potential firearm detected.",
        "reactions": {
            "left": "Alert authorities if necessary; firearm spotted on the left.",
            "center": "Avoid this area and alert authorities.",
            "right": "Alert authorities if necessary; firearm spotted on the right."
        }
    },
    "motorcycle": {
        "announce_distance": 4, 
        "ignore": False, 
        "approach_only": True,
        "message": "Motorcycle approaching.",
        "reactions": {
            "left": "Stay to your right to avoid the motorcycle.",
            "center": "Stay on the sidewalk, the motorcycle is ahead.",
            "right": "Stay to your left to avoid the motorcycle."
        }
    },
    "person": {
        "announce_distance": 5, 
        "ignore": False, 
        "approach_only": True,
        "message": "Person approaching.",
        "reactions": {
            "left": "Person approaching from the left; step right if needed.",
            "center": "Person walking towards you; allow space in front.",
            "right": "Person approaching from the right; step left if needed."
        }
    },
    "pothole": {
        "announce_distance": 1.5, 
        "ignore": False, 
        "approach_only": False,
        "message": "Pothole detected.",
        "reactions": {
            "left": "Carefully step to the right to avoid the pothole.",
            "center": "Carefully step around the pothole in front.",
            "right": "Carefully step to the left to avoid the pothole."
        }
    },
    "rat": {
        "announce_distance": 0, 
        "ignore": True, 
        "approach_only": False, 
        "message": "", 
        "reactions": {"left": "", "center": "", "right": ""}
    },
    "red_light": {
        "announce_distance": 7, 
        "ignore": False, 
        "approach_only": False,
        "message": "Red light ahead.",
        "reactions": {
            "left": "Stop if moving towards the left intersection.",
            "center": "Stop ahead at the red light.",
            "right": "Stop if moving towards the right intersection."
        }
    },
    "scooter": {
        "announce_distance": 5, 
        "ignore": False, 
        "approach_only": True,
        "message": "Scooter approaching.",
        "reactions": {
            "left": "Step right to avoid the scooter on your left.",
            "center": "Move aside, the scooter is in front.",
            "right": "Step left to avoid the scooter on your right."
        }
    },
    "stairs": {
        "announce_distance": 5, 
        "ignore": False, 
        "approach_only": False,
        "message": "Stairs detected.",
        "reactions": {
            "left": "Stairs on your left; hold the railing if available.",
            "center": "Stairs in front; proceed carefully.",
            "right": "Stairs on your right; hold the railing if available."
        }
    },
    "stop_sign": {
        "announce_distance": 6, 
        "ignore": False, 
        "approach_only": False,
        "message": "Stop sign ahead.",
        "reactions": {
            "left": "Prepare to stop if heading towards the left intersection.",
            "center": "Prepare to stop ahead at the stop sign.",
            "right": "Prepare to stop if heading towards the right intersection."
        }
    },
    "traffic_cone": {
        "announce_distance": 3, 
        "ignore": False, 
        "approach_only": False,
        "message": "Traffic cone ahead.",
        "reactions": {
            "left": "Navigate around the cone to your right.",
            "center": "Navigate around the cone in front.",
            "right": "Navigate around the cone to your left."
        }
    },
    "train": {
        "announce_distance": 3, 
        "ignore": False, 
        "approach_only": True,
        "message": "Train approaching.",
        "reactions": {
            "left": "Move to the right, away from the train tracks.",
            "center": "Stay far back, the train is in front.",
            "right": "Move to the left, away from the train tracks."
        }
    },
    "tree": {
        "announce_distance": 3, 
        "ignore": True, 
        "approach_only": False, 
        "message": "", 
        "reactions": {"left": "", "center": "", "right": ""}
    },
    "truck": {
        "announce_distance": 5, 
        "ignore": False, 
        "approach_only": True,
        "message": "Truck nearby.",
        "reactions": {
            "left": "Stay on the sidewalk; truck on the left.",
            "center": "Stay on the sidewalk, the truck is in front.",
            "right": "Stay on the sidewalk; truck on the right."
        }
    },
    "umbrella": {
        "announce_distance": 1, 
        "ignore": True, 
        "approach_only": False, 
        "message": "", 
        "reactions": {"left": "", "center": "", "right": ""}
    },
    "yellow_light": {
        "announce_distance": 2, 
        "ignore": False, 
        "approach_only": False,
        "message": "Yellow light ahead.",
        "reactions": {
            "left": "Prepare to stop if heading towards the left intersection.",
            "center": "Prepare to stop ahead, the light may turn red.",
            "right": "Prepare to stop if heading towards the right intersection."
        }
    }
}
class Communicate(QObject):
    update_log = pyqtSignal(str)
    start_detection_signal = pyqtSignal()
    stop_detection_signal = pyqtSignal()
    instructions_signal = pyqtSignal(str)
    speak_commands_signal = pyqtSignal()  # New signal for speaking voice commands
    settings_feedback_signal = pyqtSignal()  # New signal for settings feedback
    volume_up_signal = pyqtSignal()  # New signal for increasing volume
    volume_down_signal = pyqtSignal()  # New signal for decreasing volume
    exit_signal = pyqtSignal()  # Add signal for exiting the application
    toggle_high_contrast_signal = pyqtSignal(bool)
    toggle_distance_signal = pyqtSignal(bool)  # Signal for toggling distance announcement
    toggle_guidance_signal = pyqtSignal(bool)  # Signal for toggling guidance announcement
    main_feedback_signal = pyqtSignal()
    reset_to_default_signal = pyqtSignal()  # New signal for resetting settings
    unrecognized_command_signal = pyqtSignal()  # New signal for unrecognized commands

    toggle_bench_signal = pyqtSignal(bool)
    toggle_bicycle_signal = pyqtSignal(bool)
    toggle_branch_signal = pyqtSignal(bool)
    toggle_bus_signal = pyqtSignal(bool)
    toggle_bushes_signal = pyqtSignal(bool)
    toggle_car_signal = pyqtSignal(bool)
    toggle_crosswalk_signal = pyqtSignal(bool)
    toggle_door_signal = pyqtSignal(bool)
    toggle_elevator_signal = pyqtSignal(bool)
    toggle_fire_hydrant_signal = pyqtSignal(bool)
    toggle_green_light_signal = pyqtSignal(bool)
    toggle_gun_signal = pyqtSignal(bool)
    toggle_motorcycle_signal = pyqtSignal(bool)
    toggle_person_signal = pyqtSignal(bool)
    toggle_pothole_signal = pyqtSignal(bool)
    toggle_rat_signal = pyqtSignal(bool)
    toggle_red_light_signal = pyqtSignal(bool)
    toggle_scooter_signal = pyqtSignal(bool)
    toggle_stairs_signal = pyqtSignal(bool)
    toggle_stop_sign_signal = pyqtSignal(bool)
    toggle_traffic_cone_signal = pyqtSignal(bool)
    toggle_train_signal = pyqtSignal(bool)
    toggle_tree_signal = pyqtSignal(bool)
    toggle_truck_signal = pyqtSignal(bool)
    toggle_umbrella_signal = pyqtSignal(bool)
    toggle_yellow_light_signal = pyqtSignal(bool)


class ObjectDetectionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.comm = Communicate()
        self.tts_busy = False
        # Initialize lock for TTS to prevent simultaneous runAndWait() calls
        self.tts_lock = threading.Lock()
        self.setWindowTitle("Vision Aid - Accessible Object Detection")

        # Initialize settings with default values
        self.default_sensitivity = 5
        self.default_volume = 50  # Assuming 50% volume as default
        self.default_object_filters = {obj: True for obj in object_rules.keys()}
        
        # Current settings
        self.sensitivity_value = self.default_sensitivity
        self.volume_value = self.default_volume
        self.object_filters = self.default_object_filters.copy()
        
        # Toggle for distance announcements and guidance
        self.show_distance = True
        self.show_guidance = True
        
        # Initialize voice control attribute
        self.voice_control_enabled = True
        self.awaiting_instruction_response = False
        self.last_announced = {}
        self.announcement_queue = queue.Queue()  # Queue for announcements
        self.active_objects = {}  # Store objects detected in the current frame

        # Set up the announcement timer
        self.announcement_timer = QTimer()
        self.announcement_timer.timeout.connect(self.process_next_announcement)
        self.announcement_timer.start(500)  # Check the queue every 1 second

        # Set up the stacked layout (for main and settings view)
        self.stack = QStackedWidget()
        
        # Main detection view
        self.main_widget = QWidget()
        self.setup_main_view()
        self.stack.addWidget(self.main_widget)

        # Settings view
        self.settings_widget = QWidget()
        self.setup_settings_view()
        self.stack.addWidget(self.settings_widget)
        
        # Set initial view to main
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(self.stack)
        self.setLayout(main_layout)
        self.stack.setCurrentWidget(self.main_widget)
        
        # Signal object for communication
        self.comm.main_feedback_signal.connect(self.main_feedback)
        self.comm.update_log.connect(self.add_log)
        self.comm.start_detection_signal.connect(self.start_detection)
        self.comm.stop_detection_signal.connect(self.stop_detection)
        self.comm.instructions_signal.connect(self.handle_instruction_response)
        self.comm.speak_commands_signal.connect(self.speak_voice_commands)  # Connect new signal
        self.comm.settings_feedback_signal.connect(self.settings_feedback)  # Connect settings feedback signal
        self.comm.volume_up_signal.connect(self.volume_up_feedback)
        self.comm.volume_down_signal.connect(self.volume_down_feedback)
        self.comm.exit_signal.connect(self.exit_application)  # Connect exit signal to exit method
        self.comm.toggle_high_contrast_signal.connect(self.voice_toggle_high_contrast)
        self.comm.toggle_distance_signal.connect(self.voice_toggle_distance)
        self.comm.toggle_guidance_signal.connect(self.voice_toggle_guidance)
        self.comm.reset_to_default_signal.connect(self.reset_to_default_settings)
        self.comm.unrecognized_command_signal.connect(self.unrecognized_command_feedback)  # Connect unrecognized command signal
        
         # Connect the bus toggle signal
        self.comm.toggle_bench_signal.connect(lambda state: self.toggle_object_detection("bench", state))
        self.comm.toggle_bicycle_signal.connect(lambda state: self.toggle_object_detection("bicycle", state))
        self.comm.toggle_branch_signal.connect(lambda state: self.toggle_object_detection("branch", state))
        self.comm.toggle_bus_signal.connect(lambda state: self.toggle_object_detection("bus", state))
        self.comm.toggle_bushes_signal.connect(lambda state: self.toggle_object_detection("bushes", state))
        self.comm.toggle_car_signal.connect(lambda state: self.toggle_object_detection("car", state))
        self.comm.toggle_crosswalk_signal.connect(lambda state: self.toggle_object_detection("crosswalk", state))
        self.comm.toggle_door_signal.connect(lambda state: self.toggle_object_detection("door", state))
        self.comm.toggle_elevator_signal.connect(lambda state: self.toggle_object_detection("elevator", state))
        self.comm.toggle_fire_hydrant_signal.connect(lambda state: self.toggle_object_detection("fire_hydrant", state))
        self.comm.toggle_green_light_signal.connect(lambda state: self.toggle_object_detection("green_light", state))
        self.comm.toggle_gun_signal.connect(lambda state: self.toggle_object_detection("gun", state))
        self.comm.toggle_motorcycle_signal.connect(lambda state: self.toggle_object_detection("motorcycle", state))
        self.comm.toggle_person_signal.connect(lambda state: self.toggle_object_detection("person", state))
        self.comm.toggle_pothole_signal.connect(lambda state: self.toggle_object_detection("pothole", state))
        self.comm.toggle_rat_signal.connect(lambda state: self.toggle_object_detection("rat", state))
        self.comm.toggle_red_light_signal.connect(lambda state: self.toggle_object_detection("red_light", state))
        self.comm.toggle_scooter_signal.connect(lambda state: self.toggle_object_detection("scooter", state))
        self.comm.toggle_stairs_signal.connect(lambda state: self.toggle_object_detection("stairs", state))
        self.comm.toggle_stop_sign_signal.connect(lambda state: self.toggle_object_detection("stop_sign", state))
        self.comm.toggle_traffic_cone_signal.connect(lambda state: self.toggle_object_detection("traffic_cone", state))
        self.comm.toggle_train_signal.connect(lambda state: self.toggle_object_detection("train", state))
        self.comm.toggle_tree_signal.connect(lambda state: self.toggle_object_detection("tree", state))
        self.comm.toggle_truck_signal.connect(lambda state: self.toggle_object_detection("truck", state))
        self.comm.toggle_umbrella_signal.connect(lambda state: self.toggle_object_detection("umbrella", state))
        self.comm.toggle_yellow_light_signal.connect(lambda state: self.toggle_object_detection("yellow_light", state))
   

        # Start voice control in a separate thread
        self.voice_control_thread = threading.Thread(target=self.voice_control, daemon=True)
        self.voice_control_thread.start()

        # Ask for initial instructions
        self.ask_for_instructions()

    def setup_main_view(self):
        """Setup the main detection view."""
        layout = QVBoxLayout()
        button_layout = QHBoxLayout()
        
        # Video display setup
        self.video_label = QLabel(self)
        self.video_label.setFixedSize(VIDEO_WIDTH, VIDEO_HEIGHT)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: #333333;") #a dark grey background for improved contrast

        self.log_textbox = QTextEdit(self)
        self.log_textbox.setReadOnly(True)

        # Detection control buttons
        self.start_button = QPushButton("Start Detection")
        self.start_button.setStyleSheet("background-color: #4CAF50; color: white; font-size: 16px; padding: 10px; border-radius: 10px;")
        self.start_button.clicked.connect(self.start_detection)


        self.stop_button = QPushButton("Stop Detection")
        self.stop_button.setStyleSheet("background-color: #f44336; color: white; font-size: 16px; padding: 10px; border-radius: 10px;")
        self.stop_button.clicked.connect(self.stop_detection)

        # Exit button setup, connected to the exit signal
        self.exit_button = QPushButton("Exit")
        self.exit_button.setStyleSheet("background-color: #3F51B5; color: white; font-size: 16px; padding: 10px; border-radius: 10px;")
        self.exit_button.clicked.connect(self.comm.exit_signal.emit)  # Emit signal on click
        button_layout.addWidget(self.exit_button)

        #settings button controls 
        self.settings_button = QPushButton("Settings")
        self.settings_button.setStyleSheet("background-color: #607D8B; color: white; font-size: 16px; padding: 10px; border-radius: 10px;")
        self.settings_button.clicked.connect(lambda: self.comm.settings_feedback_signal.emit())

        # Button to hear all voice commands
        self.voice_commands_button = QPushButton("Hear Voice Commands")
        self.voice_commands_button.setStyleSheet("background-color: #FFA500; color: white; font-size: 16px; padding: 10px; border-radius: 10px;")
        self.voice_commands_button.clicked.connect(lambda: self.comm.speak_commands_signal.emit())

        # Add buttons to layout
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        button_layout.addWidget(self.settings_button)
        button_layout.addWidget(self.voice_commands_button) 
        button_layout.addWidget(self.exit_button)

        # Add widgets to main layout
        layout.addWidget(self.video_label)
        layout.addWidget(self.log_textbox)
        layout.addLayout(button_layout)
        self.main_widget.setLayout(layout)

    def exit_application(self):
        """Exit the application with feedback."""
        feedback_text = "Exiting application."
        self.add_log(feedback_text)
        
        # Provide TTS feedback if not busy
        if not self.tts_busy:
            with self.tts_lock:
                self.tts_busy = True
                engine.say(feedback_text)
                engine.runAndWait()
                self.tts_busy = False
        
        # Close the application after feedback
        self.close()

    def unrecognized_command_feedback(self):
        """Provide feedback for unrecognized voice commands."""
        message = "Command not recognized. Please try again or say 'list commands' for help."
        self.add_log(message)
      
        if not self.tts_busy:
            with self.tts_lock:
                self.tts_busy = True
                engine.say(message)
                engine.runAndWait()
                self.tts_busy = False

    def speak_voice_commands(self):
        """Speak out the available voice commands, ensuring no overlap with other TTS tasks."""
        if self.tts_busy:
            self.add_log("TTS is currently busy, please wait.")
            return

        commands_text = (
            "Available voice commands are: "
            "Start detection to begin object detection. "
            "Stop detection to stop object detection. "
            "Settings to open the settings menu. "
            "Exit to close the application. "
            "Back to main to return to the main screen. "
            "Volume up to increase the announcement volume. "
            "Volume down to decrease the announcement volume. "
            "distance on to activate distance announcements. "
            "distance off to turn off distance announcements. "
            "guidance on to activate guidance announcements. "
            "guidance off to turn off guidance announcements. "
            "high contrast on for high-contrast theme. "
            "high contrast off to revert to normal theme. "
            "You can toggle detection of objects off and on by saying toggle and the object name"
            "Reset settings to revert basic settings to their default values."
        )

        with self.tts_lock:
            self.tts_busy = True
            engine.say(commands_text)
            engine.runAndWait()
            self.tts_busy = False
            
    def settings_feedback(self):
        """Provide audio feedback when opening the settings."""
        if self.tts_busy:
            self.add_log("TTS is currently busy, please wait.")
            return

        feedback_text = "Opening settings. You can adjust detection sensitivity, volume, and more."

        with self.tts_lock:
            self.tts_busy = True
            engine.say(feedback_text)
            engine.runAndWait()
            self.tts_busy = False

    # Then open the settings view
        self.open_settings()

    def volume_up_feedback(self):
        """Increase volume with audio feedback."""
        if self.tts_busy:
            self.add_log("TTS is currently busy, please wait.")
            return

        self.volume_value = min(self.volume_value + 10, 100)  # Increase volume by 10, max 100
        self.volume_slider.setValue(self.volume_value)  # Sync slider with new value
        engine.setProperty('volume', self.volume_value / 100.0)
        feedback_text = f"Volume increased to {self.volume_value}%."

        with self.tts_lock:
            self.tts_busy = True
            engine.say(feedback_text)
            engine.runAndWait()
            self.tts_busy = False
        self.add_log(feedback_text)
        
    def volume_down_feedback(self):
        """Decrease volume with audio feedback."""
        if self.tts_busy:
            self.add_log("TTS is currently busy, please wait.")
            return

        self.volume_value = max(self.volume_value - 10, 0)  # Decrease volume by 10, min 0
        self.volume_slider.setValue(self.volume_value)  # Sync slider with new value
        engine.setProperty('volume', self.volume_value / 100.0)
        feedback_text = f"Volume decreased to {self.volume_value}%."

        with self.tts_lock:
            self.tts_busy = True
            engine.say(feedback_text)
            engine.runAndWait()
            self.tts_busy = False
        self.add_log(feedback_text)



    def setup_settings_view(self):
        """Setup the settings view."""
    # Use a scroll area to contain settings, so it doesn't overflow the screen
        scroll_area = QScrollArea()
        settings_container = QWidget()
        layout = QVBoxLayout(settings_container)

    # Sensitivity Header
        sensitivity_header = QLabel("Detection Settings")
        sensitivity_header.setStyleSheet("font-weight: bold; font-size: 16px; margin-top: 10px;")
        layout.addWidget(sensitivity_header)

    # Sensitivity slider
        self.sensitivity_slider = QSlider(Qt.Horizontal)
        self.sensitivity_slider.setMinimum(1)
        self.sensitivity_slider.setMaximum(10)
        self.sensitivity_slider.setValue(self.sensitivity_value)
        self.sensitivity_slider.valueChanged.connect(self.adjust_sensitivity)
        layout.addWidget(QLabel("Detection Sensitivity"))
        layout.addWidget(self.sensitivity_slider)

    # Create a separator line
        separator1 = QFrame()
        separator1.setFrameShape(QFrame.HLine)
        separator1.setFrameShadow(QFrame.Sunken)
        layout.addWidget(separator1)

     # Announcement Header
        announcement_header = QLabel("Announcement Settings")
        announcement_header.setStyleSheet("font-weight: bold; font-size: 16px; margin-top: 10px;")
        layout.addWidget(announcement_header)

    # Volume slider
        self.volume_slider = QSlider(Qt.Horizontal)
        self.volume_slider.setMinimum(0)
        self.volume_slider.setMaximum(100)
        self.volume_slider.setValue(self.volume_value)
        self.volume_slider.valueChanged.connect(self.update_volume_from_slider)
        layout.addWidget(QLabel("Announcement Volume"))
        layout.addWidget(self.volume_slider)

     # Distance announcement toggle
        self.distance_checkbox = QCheckBox("Enable Distance Announcement")
        self.distance_checkbox.setChecked(self.show_distance)
        self.distance_checkbox.stateChanged.connect(self.toggle_distance_announcement)
        layout.addWidget(self.distance_checkbox)

        # Guidance announcement toggle
        self.guidance_checkbox = QCheckBox("Enable Guidance Announcement")
        self.guidance_checkbox.setChecked(self.show_guidance)
        self.guidance_checkbox.stateChanged.connect(self.toggle_guidance_announcement)
        layout.addWidget(self.guidance_checkbox)

        # Create a separator line
        separator1 = QFrame()
        separator1.setFrameShape(QFrame.HLine)
        separator1.setFrameShadow(QFrame.Sunken)
        layout.addWidget(separator1)

        # Announcement Header
        announcement_header = QLabel("Interface Settings")
        announcement_header.setStyleSheet("font-weight: bold; font-size: 16px; margin-top: 10px;")
        layout.addWidget(announcement_header)

    
    # High-contrast theme toggle
        self.high_contrast_checkbox = QCheckBox("Enable High-Contrast Theme")
        self.high_contrast_checkbox.stateChanged.connect(self.toggle_high_contrast)
        layout.addWidget(self.high_contrast_checkbox)

    # Voice control toggle
        self.voice_control_checkbox = QCheckBox("Enable Voice Control")
        self.voice_control_checkbox.setChecked(self.voice_control_enabled)
        self.voice_control_checkbox.stateChanged.connect(self.toggle_voice_control)
        layout.addWidget(self.voice_control_checkbox)

    # UI scaling slider
        self.ui_scaling_slider = QSlider(Qt.Horizontal)
        self.ui_scaling_slider.setMinimum(10)
        self.ui_scaling_slider.setMaximum(30)
        self.ui_scaling_slider.setValue(16)  # Default font size
        self.ui_scaling_slider.valueChanged.connect(self.adjust_ui_scaling)
        layout.addWidget(QLabel("UI Scaling"))
        layout.addWidget(self.ui_scaling_slider)

    # Create a separator line
        separator1 = QFrame()
        separator1.setFrameShape(QFrame.HLine)
        separator1.setFrameShadow(QFrame.Sunken)
        layout.addWidget(separator1)

        # Announcement Header
        announcement_header = QLabel("Object Settings")
        announcement_header.setStyleSheet("font-weight: bold; font-size: 16px; margin-top: 10px;")
        layout.addWidget(announcement_header)


    # Object filters and distance adjustment sliders
        layout.addWidget(QLabel("Object Detection Settings"))
        self.object_checkboxes = {}
        self.distance_sliders = {}
        self.distance_labels = {}

    # Separate function for updating distances
        def update_distance_label(value, label):
            label.setText(f"{value} meters")
            
        # Helper functions to handle slider changes
        def toggle_object_detection(state, obj):
            self.object_filters[obj] = bool(state)
            
        def update_announce_distance(value, obj):
            object_rules[obj]["announce_distance"] = value
            self.distance_labels[obj].setText(f"{value} meters")

    # Create distance sliders and labels for each object
         # Create distance sliders and labels for each object
        for obj, rule in object_rules.items():
        # Object detection toggle
            checkbox = QCheckBox(f"Detect {obj.capitalize()}")
            checkbox.setChecked(self.object_filters[obj])
            checkbox.stateChanged.connect(lambda state, obj=obj: toggle_object_detection(state, obj))
            layout.addWidget(checkbox)
            self.object_checkboxes[obj] = checkbox

        # Distance slider
            distance_slider = QSlider(Qt.Horizontal)
            distance_slider.setMinimum(1)
            distance_slider.setMaximum(10)
            distance_slider.setValue(int(rule["announce_distance"]))
            distance_slider.valueChanged.connect(lambda value, obj=obj: update_announce_distance(value, obj))
            layout.addWidget(QLabel(f"{obj.capitalize()} Detection Distance"))

        # Distance label to show current slider value
            distance_label = QLabel(f"{int(rule['announce_distance'])} meters")
            layout.addWidget(distance_label)

            layout.addWidget(distance_slider)
            self.distance_sliders[obj] = distance_slider
            self.distance_labels[obj] = distance_label

    # Reset button
        reset_button = QPushButton("Reset to Default Settings")
        reset_button.setStyleSheet("background-color: #607D8B; color: white; font-size: 16px; padding: 10px; border-radius: 10px;")
        reset_button.clicked.connect(self.reset_to_default_settings)
        layout.addWidget(reset_button)

    # Back button to return to the main view
        back_button = QPushButton("Back to Main")
        back_button.setStyleSheet("background-color: #607D8B; color: white; font-size: 16px; padding: 10px; border-radius: 10px;")
        back_button.clicked.connect(self.open_main)
        layout.addWidget(back_button)

    # Add all settings to the scroll area and set the layout
        scroll_area.setWidgetResizable(True)
        scroll_area.setWidget(settings_container)
        self.settings_widget.setLayout(QVBoxLayout())
        self.settings_widget.layout().addWidget(scroll_area)

    def update_volume_from_slider(self, value):
        """Update volume based on slider position."""
        self.volume_value = value
        engine.setProperty('volume', value / 100.0)
        feedback_text = f"Volume set to {value}% via slider."
        self.add_log(feedback_text)  

    def voice_toggle_high_contrast(self, enable):
        self.toggle_high_contrast(enable)
        
    def toggle_high_contrast(self, state):
        """Toggle high contrast mode for accessibility."""
        if state:
            self.setStyleSheet("QWidget { background-color: black; color: white; }"
                           "QPushButton { background-color: #f0f0f0; color: black; }")
        else:
            self.setStyleSheet("")  # Reset to default
        self.high_contrast_checkbox.setChecked(state)  # Reflect state in settings checkbox
        feedback_text = "High contrast mode enabled." if state else "High contrast mode disabled."
        self.add_log(feedback_text)
        engine.say(feedback_text)
        engine.runAndWait()

    def adjust_ui_scaling(self, value):
        """Adjust the scaling of UI elements for accessibility."""
        font = self.font()
        font.setPointSize(value)
        self.setFont(font)
        self.add_log(f"UI scaled to font size {value}.")
        
    def voice_toggle_distance(self, enable):
        self.toggle_distance_announcement(enable)

    def voice_toggle_guidance(self, enable):
        self.toggle_guidance_announcement(enable)


    def toggle_distance_announcement(self, state):
        """Enable or disable distance announcement."""
        self.show_distance = bool(state)
        self.distance_checkbox.setChecked(state)  # Reflect state in settings checkbox
        feedback_text = "Distance announcement enabled." if state else "Distance announcement disabled."
        self.add_log(feedback_text)
        engine.say(feedback_text)
        engine.runAndWait()

    def toggle_guidance_announcement(self, state):
        """Enable or disable guidance announcement."""
        self.show_guidance = bool(state)
        self.guidance_checkbox.setChecked(state)  # Reflect state in settings checkbox
        feedback_text = "Guidance announcement enabled." if state else "Guidance announcement disabled."
        self.add_log(feedback_text)
        engine.say(feedback_text)
        engine.runAndWait()

    def adjust_sensitivity(self, value):
        """Adjust the detection sensitivity based on slider value."""
        global ANNOUNCEMENT_INTERVAL
        ANNOUNCEMENT_INTERVAL = max(1, 11 - value)  # Inverse relationship for sensitivity
        self.sensitivity_value = value  # Store the current sensitivity

    def adjust_volume(self, value):
        """Adjust the volume of the text-to-speech announcements."""
        self.volume_value = value
        engine.setProperty('volume', value / 100.0)  # Pyttsx3 volume range is 0.0 to 1.0

    def toggle_object_detection(self, obj, state):
        """Enable or disable detection for a specific object."""
        self.object_filters[obj] = bool(state)
    
    # Update the checkbox in settings if it exists
        if obj in self.object_checkboxes:
            self.object_checkboxes[obj].setChecked(state)
    
    # Provide audio feedback
        feedback_text = f"{obj.capitalize()} detection {'enabled' if state else 'disabled'}."
        self.add_log(feedback_text)
    
        if not self.tts_busy:
            with self.tts_lock:
                self.tts_busy = True
                engine.say(feedback_text)
                engine.runAndWait()
                self.tts_busy = False

    def reset_to_default_settings(self):
        """Reset all settings to their default values."""
        # Announce only "Resetting to default settings."
        reset_message = "Resetting basic default settings."
        self.add_log(reset_message)
    
        if not self.tts_busy:
            with self.tts_lock:
                self.tts_busy = True
                engine.say(reset_message)
                engine.runAndWait()
                self.tts_busy = False

        # Reset detection sensitivity
        self.sensitivity_slider.setValue(self.default_sensitivity)
        self.adjust_sensitivity(self.default_sensitivity)
    
    # Reset volume
        self.volume_slider.setValue(self.default_volume)
        self.adjust_volume(self.default_volume)
        
        # Reset high-contrast themethis 
        self.high_contrast_checkbox.setChecked(False)
        self.toggle_high_contrast(False)
    
    # Reset UI scaling
        
    
    # Reset distance and guidance announcements
        self.distance_checkbox.setChecked(True)
        self.toggle_distance_announcement(True)
    
        self.guidance_checkbox.setChecked(True)
        self.toggle_guidance_announcement(True)
    
       
        self.add_log("All settings have been reset to default.")

    def ask_for_instructions(self):
        """Ask if the user wants instructions."""
        self.awaiting_instruction_response = True
        engine.say("Welcome to VisionAid. Would you like to hear the instructions? Please say yes or no.")
        engine.runAndWait()

    def handle_instruction_response(self, response):
        """Handle the response to the instructions prompt."""
        self.awaiting_instruction_response = False
        if response.lower() == "yes":
            self.add_log("Instructions: You can use commands like 'Start detection', 'Stop detection', 'Settings', and 'Exit'.")
            engine.say("You can use commands like 'Start detection', 'Stop detection', 'Settings', and 'Exit'.")
        elif response.lower() == "no":
            self.add_log("Skipping instructions.")
            engine.say("Okay, skipping instructions.")
        engine.runAndWait()

    def start_detection(self):
        """Start object detection."""
        self.add_log("Starting detection...")
        with self.tts_lock:
            engine.say("Starting detection")
            engine.runAndWait()
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def stop_detection(self):
        """Stop object detection."""
        self.add_log("Stopping detection...")
        if hasattr(self, 'timer') and self.timer.isActive():
            self.timer.stop()
        with self.tts_lock:
            engine.say("Stopping detection")
            engine.runAndWait()

    def get_direction(self, x1, x2, frame_width):
        """Calculate direction (left, center, or right) based on bounding box."""
        center_x = (x1 + x2) / 2
        left_threshold = frame_width * (1 - CENTER_ZONE_WIDTH_RATIO) / 2
        right_threshold = frame_width * (1 + CENTER_ZONE_WIDTH_RATIO) / 2

        if center_x < left_threshold:
            return "left"
        elif center_x > right_threshold:
            return "right"
        else:
            return "center"

    def add_log(self, message):
        """Log a message to the GUI."""
        self.log_textbox.append(message)
        self.log_textbox.moveCursor(QTextCursor.End)

  



        
    def voice_control(self):
        """Voice control for basic commands."""
        while True:
            if not self.voice_control_enabled:
                continue
            with sr.Microphone() as source:
                recognizer.adjust_for_ambient_noise(source)
                try:
                    audio = recognizer.listen(source)
                    command = recognizer.recognize_google(audio).lower()
                    self.add_log(f"Voice command received: {command}")

                    if self.awaiting_instruction_response and command in ["yes", "no"]:
                        self.comm.instructions_signal.emit(command)
                    elif command == "start detection":
                        self.comm.start_detection_signal.emit()
                    elif command == "stop detection":
                        self.comm.stop_detection_signal.emit()
                    elif command in ["settings", "settings menu"]:
                        self.comm.settings_feedback_signal.emit()  # Emit signal for settings feedback
                    elif command == "exit":
                         self.comm.exit_signal.emit()  # Emit exit signal on voice command
                    elif command in ["list commands", "voice help", "commands", "help", "voice commands"]:
                        self.comm.speak_commands_signal.emit()  # Use the signal to call speak_voice_commands
                    elif command in ["high contrast on", "contrast on"]:
                        self.comm.toggle_high_contrast_signal.emit(True)
                    elif command in ["high contrast off", "contrast off"]:
                        self.comm.toggle_high_contrast_signal.emit(False)
                    elif command == "distance on":
                        self.comm.toggle_distance_signal.emit(True)
                    elif command == "distance off":
                        self.comm.toggle_distance_signal.emit(False)
                    elif command == "guidance on":
                        self.comm.toggle_guidance_signal.emit(True)
                    elif command == "guidance off":
                        self.comm.toggle_guidance_signal.emit(False)
                    elif command in ["increase volume", "volume up"]:
                        self.comm.volume_up_signal.emit()
                    elif command in ["decrease volume", "volume down"]:
                        self.comm.volume_down_signal.emit()
                    elif command == "toggle bus":
                        current_state = self.object_filters["bus"]
                        self.comm.toggle_bus_signal.emit(not current_state)    
                    elif command == "toggle bench":
                        current_state = self.object_filters["bench"]
                        self.comm.toggle_bench_signal.emit(not current_state)
                    elif command in["toggle bush", "toggle bushes"]:
                        current_state = self.object_filters["bushes"]
                        self.comm.toggle_bushes_signal.emit(not current_state)
                    elif command == "toggle bicycle":
                        current_state = self.object_filters["bicycle"]
                        self.comm.toggle_bicycle_signal.emit(not current_state)
                    elif command == "toggle branch":
                        current_state = self.object_filters["branch"]
                        self.comm.toggle_branch_signal.emit(not current_state)
                    elif command == "toggle car":
                        current_state = self.object_filters["car"]
                        self.comm.toggle_car_signal.emit(not current_state)
                    elif command == "toggle crosswalk":
                        current_state = self.object_filters["crosswalk"]
                        self.comm.toggle_crosswalk_signal.emit(not current_state)
                    elif command == "toggle door":
                        current_state = self.object_filters["car"]
                        self.comm.toggle_car_signal.emit(not current_state)
                    elif command == "toggle elevator":
                        current_state = self.object_filters["elevator"]
                        self.comm.toggle_elevator_signal.emit(not current_state)
                    elif command == "toggle fire hydrant":
                        current_state = self.object_filters["fire_hydrant"]
                        self.comm.toggle_fire_hydrant_signal.emit(not current_state)
                    elif command == "toggle green light":
                        current_state = self.object_filters["green_light"]
                        self.comm.toggle_green_light_signal.emit(not current_state)
                    elif command == "toggle gun":
                        current_state = self.object_filters["gun"]
                        self.comm.toggle_gun_signal.emit(not current_state)
                    elif command == "toggle motorcycle":
                        current_state = self.object_filters["motorcycle"]
                        self.comm.toggle_motorcycle_signal.emit(not current_state)
                    elif command == "toggle person":
                        current_state = self.object_filters["person"]
                        self.comm.toggle_person_signal.emit(not current_state)
                    elif command == "toggle pothole":
                        current_state = self.object_filters["pothole"]
                        self.comm.toggle_pothole_signal.emit(not current_state)
                    elif command == "toggle rat":
                        current_state = self.object_filters["rat"]
                        self.comm.toggle_rat_signal.emit(not current_state)
                    elif command == "toggle red light":
                        current_state = self.object_filters["red_light"]
                        self.comm.toggle_red_light_signal.emit(not current_state)
                    elif command == "toggle scooter":
                        current_state = self.object_filters["scooter"]
                        self.comm.toggle_scooter_signal.emit(not current_state)
                    elif command == "toggle stairs":
                        current_state = self.object_filters["stairs"]
                        self.comm.toggle_stairs_signal.emit(not current_state)
                    elif command == "toggle stop sign":
                        current_state = self.object_filters["stop_sign"]
                        self.comm.toggle_stop_sign_signal.emit(not current_state)
                    elif command == "toggle traffic cone":
                        current_state = self.object_filters["traffic_cone"]
                        self.comm.toggle_traffic_cone_signal.emit(not current_state)
                    elif command == "toggle train":
                        current_state = self.object_filters["train"]
                        self.comm.toggle_train_signal.emit(not current_state)
                    elif command == "toggle tree":
                        current_state = self.object_filters["tree"]
                        self.comm.toggle_tree_signal.emit(not current_state)
                    elif command == "toggle truck":
                        current_state = self.object_filters["truck"]
                        self.comm.toggle_truck_signal.emit(not current_state)
                    elif command == "toggle umbrella":
                        current_state = self.object_filters["umbrella"]
                        self.comm.toggle_umbrella_signal.emit(not current_state)
                    elif command == "toggle yellow light":
                        current_state = self.object_filters["yellow_light"]
                        self.comm.toggle_yellow_light_signal.emit(not current_state)

                        
                    elif command in ["back to main", "main", "main menu"]:
                        self.add_log("Back to main command recognized")  # Debug log
                        self.comm.main_feedback_signal.emit()  # Emit the correct signal
                    elif command == "reset settings" or command == "reset to default":
                        self.comm.reset_to_default_signal.emit()  # Emit reset signal
                    else:
                        # Emit unrecognized command signal
                        self.comm.unrecognized_command_signal.emit()
                except sr.UnknownValueError:
                    self.add_log("Could not understand audio")
                except sr.RequestError:
                    self.add_log("Voice recognition service unavailable")



    def toggle_voice_control(self, state):
        """Enable or disable voice control."""
        self.voice_control_enabled = bool(state)

    def open_settings(self):
        """Open the settings view."""
        self.stack.setCurrentWidget(self.settings_widget)

    def main_feedback(self):
        """Provide audio feedback when returning to the main screen."""
        if self.tts_busy:
            self.add_log("TTS is currently busy, please wait.")
            return

        feedback_text = "Returning to main screen."

        with self.tts_lock:
            self.tts_busy = True
            engine.say(feedback_text)
            engine.runAndWait()
            self.tts_busy = False

    # Switch to main view after feedback
        self.stack.setCurrentWidget(self.main_widget)

    def open_main(self):
        """Emit signal to provide feedback and return to the main detection view."""
        self.comm.main_feedback_signal.emit()  # Emit the feedback signal for main screen

    def update_frame(self):
        """Update frame with object detection."""
        success, frame = cap.read()
        if not success:
            self.add_log("Failed to capture frame.")
            return

        frame = cv2.resize(frame, (VIDEO_WIDTH, VIDEO_HEIGHT))
        results = model(frame, conf=0.5)
        frame_width = frame.shape[1]

        # Reset the active_objects dictionary for the current frame
        current_frame_objects = {}

        # Collect objects with their distance for prioritized announcement
        announcements = []

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = model.names[int(box.cls[0])]
            confidence = box.conf[0]
            distance = int(1000 / (x2 - x1))

            rule = object_rules.get(label.lower(), {"announce_distance": 3, "ignore": False, "approach_only": False})
            
            if not self.object_filters.get(label.lower(), True):
                continue

            if distance <= rule["announce_distance"]:
                direction = self.get_direction(x1, x2, frame_width)
                message_parts = [rule["message"]] if rule["message"] else []
                if self.show_guidance and direction in rule["reactions"]:
                    message_parts.append(rule["reactions"][direction])
                if self.show_distance:
                    message_parts.append(f"Distance: {distance} meters.")
                message = " ".join(message_parts)

                # Store the object in the current frame objects
                current_frame_objects[label] = message 

                # Add announcement message and distance for prioritization
                announcements.append((distance, message, label))

                # Draw bounding box on frame
                color = (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Update active_objects to reflect only objects in the current frame
        self.active_objects = current_frame_objects

        # Sort announcements by distance and announce them
        announcements.sort()  # Sort by distance, ascending
        for distance, message, label in announcements:
            if label in self.active_objects:  # Ensure the object is still in the frame
                self.announce(message, label)

        # Display updated frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        qt_image = QImage(frame_rgb.data, frame_rgb.shape[1], frame_rgb.shape[0], QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(qt_image))



    def announce(self, message, label):
        """Announce if not recently announced and prioritize important objects."""
        current_time = time.time()
    # Prioritize announcement for high-priority objects like "person" and "car"
        high_priority_labels = {"person", "bus", "car","bench", "motorcycle", "bicycle", "crosswalk","Truck", "Scooter"}
    
        if label not in self.last_announced or (current_time - self.last_announced[label] > ANNOUNCEMENT_INTERVAL):
            self.last_announced[label] = current_time
            self.add_log(message)
        
        # If it's a high-priority object, announce it immediately
            if label in high_priority_labels:
                with self.tts_lock:
                    self.tts_busy = True
                    engine.say(message)
                    engine.runAndWait()
                    self.tts_busy = False
            else:
            # Add lower-priority objects to the queue
                self.announcement_queue.put((message, label))
                

    def process_next_announcement(self):
        """Process the next announcement in the queue if the object is still in the frame."""
        try:
            while True:
                message, label = self.announcement_queue.get_nowait()
            
                # Ensure high-priority announcements are processed immediately
                if label in {"person", "car", "bus", "bench", "motorcycle", "bicycle", "crosswalk","Truck", "Scooter"} or label in self.active_objects:
                    with self.tts_lock:
                        self.tts_busy = True
                        engine.say(message)
                        engine.runAndWait()
                        self.tts_busy = False
                    self.announcement_queue.task_done()
                    break  # Announce only the next valid item
                else:
                    self.announcement_queue.task_done()
        except Empty:
            pass  # No message to announce

    def closeEvent(self, event):
        """Handle app close event."""
    # Check if TTS is busy and wait until it's free to speak the exit message
        with self.tts_lock:
            if not self.tts_busy:
                self.tts_busy = True  # Set busy flag to prevent other TTS calls
                engine.say("Exiting program")
                engine.runAndWait()
                self.tts_busy = False  # Clear busy flag after speaking

    # Release resources and accept the close event
        cap.release()
        event.accept()

# Main execution
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ObjectDetectionApp()
    window.show()
    try:
        sys.exit(app.exec_())
    except SystemExit:
        pass 


