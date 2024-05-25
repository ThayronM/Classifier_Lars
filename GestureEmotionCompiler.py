import cv2
import os
import time
from typing import Union
from modules.system.SystemSettings import *
from modules.auxiliary.FileHandler import FileHandler
from modules.auxiliary.TimeFunctions import TimeFunctions
from modules.gesture.DataProcessor import DataProcessor
from modules.gesture.GestureAnalyzer import GestureAnalyzer
from modules.gesture.FeatureExtractor import FeatureExtractor

from emotions import EmotionDetector
from imutils.video import FPS
from utils.utils import setup_logger
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' 

TAG_I = True
# This class likely represents a system designed for recognizing gestures.
class GestureEmotionCompiler:
    def __init__(self, model_name: str, model_option: str, backend_option: int,
                 providers: int, fp16: bool,num_faces: int,
                 config: InitializeConfig, operation: Union[ModeDataset, ModeValidate, ModeRealTime], 
                 file_handler: FileHandler, current_folder: str, data_processor: DataProcessor, 
                 time_functions: TimeFunctions, gesture_analyzer: GestureAnalyzer, tracking_processor, 
                 feature, classifier =None):
        
        self._initialize_camera(config)
        self._initialize_operation(operation)

        self.file_handler = file_handler
        self.current_folder = current_folder
        self.data_processor = data_processor
        self.time_functions = time_functions
        self.gesture_analyzer = gesture_analyzer
        self.classifier = classifier
        self.tracking_processor = tracking_processor
        self.feature = feature

        self._initialize_simulation_variables()
        self._initialize_storage_variables()
        
        # emotion inicialization
        self.emotions_list = {0: "BAD", 1: "GOOD", 2: "NEUTRAL"}
        self.logger = setup_logger(__name__)    
        self.Emotion = EmotionDetector(model_name, model_option, backend_option, providers, fp16, num_faces)

    def _initialize_camera(self, config: InitializeConfig) -> None:
        """
        The function `_initialize_camera` initializes camera settings based on the provided
        configuration.
        """
        self.cap = config.cap
        self.fps = config.fps
        self.dist = config.dist
        self.length = config.length

    def _initialize_operation(self, operation: Union[ModeDataset, ModeValidate, ModeRealTime]) -> None:
        """
        The function `_initialize_operation` initializes attributes based on the mode specified in the
        input operation.
        """
        self.mode = operation.mode
        if self.mode == 'D':
            self.database = operation.database
            self.file_name_build = operation.file_name_build
            self.max_num_gest = operation.max_num_gest
            self.dist = operation.dist
            self.length = operation.length
        elif self.mode == 'V':
            self.database = operation.database
            self.proportion = operation.proportion
            self.files_name = operation.files_name
            self.file_name_val = operation.file_name_val
        elif self.mode == 'RT':
            self.database = operation.database
            self.proportion = operation.proportion
            self.files_name = operation.files_name
        else:
            raise ValueError("Invalid mode")

    def _initialize_simulation_variables(self) -> None:
        """
        The function `_initialize_simulation_variables` initializes various simulation variables to
        default values.
        """
        self.stage = 0
        self.num_gest = 0
        self.dist_virtual_point = 1
        self.hands_results = None
        self.pose_results = None
        self.time_gesture = None
        self.time_action = None
        self.y_val = None
        self.frame_captured = None
        self.y_predict = []
        self.time_classifier = []

    def _initialize_storage_variables(self) -> None:
        """
        The function `_initialize_storage_variables` initializes storage variables using data processed
        by `data_processor`.
        """
        self.hand_history, _, self.wrists_history, self.sample = self.data_processor.initialize_data(self.dist, self.length)

    def run(self):
        """
        Run the gesture recognition system based on the specified mode.

        - If the mode is 'D' (Batch), initialize the database and set the loop flag to True.
        - If the mode is 'RT' (Real-Time), load and fit the classifier, and set the loop flag to True.
        - If the mode is 'V' (Validation), validate the classifier and set the loop flag to False.
        - If the mode is invalid, print a message and set the loop flag to False.

        During the loop:
        - Measure the time for each frame.
        - Check for user input to quit the loop (pressing 'q').
        - If the mode is 'D', break the loop if the maximum number of gestures is reached.
        - Process each stage of the gesture recognition system.

        After the loop, release the capture and close all OpenCV windows.

        Returns:
            None
        """
        if self.mode == 'D':
            self._initialize_database()
            self.loop = True
        elif self.mode == 'RT':
            self._load_and_fit_classifier()
            self.loop = True
        elif self.mode == 'V':
            self._validate_classifier()
            self.loop = False
        else:
            print(f"Operation mode invalid!")
            self.loop = False
            
        t_frame = self.time_functions.tic()
        while self.loop:
            if self.time_functions.toc(t_frame) > 1 / self.fps:
                t_frame = self.time_functions.tic()
                
                if cv2.waitKey(10) & 0xFF == ord("q"):
                    break
                
                if self.mode == "B":
                    if self.num_gest == self.max_num_gest:
                        break
                
                self._process_stage()
            # print(f"\n\n\nPFS: {1/self.time_functions.toc(t_frame)}")
        self.cap.release()
        cv2.destroyAllWindows()

    def _initialize_database(self):
        """
        This method initializes the target names and ground truth labels (y_val) by calling the
        initialize_database method of the file_handler object.
        """
        self.target_names, self.y_val = self.file_handler.initialize_database(self.database)

    def _load_and_fit_classifier(self):
        """
        This method loads the training data, fits the classifier with the training data, and performs
        model training.
        """
        x_train, y_train, _, _ = self.file_handler.load_database(self.current_folder, self.files_name, self.proportion)
        self.classifier.fit(x_train, y_train)

    def _validate_classifier(self):
        """
        This method validates the classifier with validation data and saves the validation results.
        """
        x_train, y_train, x_val, self.y_val = self.file_handler.load_database(self.current_folder, self.files_name, self.proportion)
        self.classifier.fit(x_train, y_train)
        self.y_predict, self.time_classifier = self.classifier.validate(x_val)
        self.target_names, _ = self.file_handler.initialize_database(self.database)
        self.file_handler.save_results(self.y_val, self.y_predict, self.time_classifier, self.target_names, os.path.join(self.current_folder, self.file_name_val))

    def _process_stage(self) -> None:
        """
        The `_process_stage` function handles different stages and modes of processing in the system.        
        Returns:
        - If conditions are met, the function may return `None` or continue execution without returning anything.
        """
        if self.stage in [0, 1] and self.mode in ['D', 'RT']:
            if not self.read_image():
                return
            if not self.image_processing():
                return
            self.extract_features()
        elif self.stage == 2 and self.mode in ['D', 'RT']:
            self.process_reduction()
            if self.mode == 'D':
                self.stage = 3
            elif self.mode == 'RT':
                self.stage = 4
        elif self.stage == 3 and self.mode == 'D':
            if self.update_database():
                self.loop = False
            self.stage = 0
        elif self.stage == 4 and self.mode == 'RT':
            self.classify_gestures()
            self.stage = 5
            self.counter = {"GOOD" : 10, "BAD" : 10}
            
        elif self.stage == 5:
            if not self.read_image():
                return
            self.classifier_emotion()
            #self.print_data(self.frame_captured)
 
            

    def read_image(self) -> None:
        """
        The function `read_image` reads an image from a camera capture device and returns a success flag
        along with the captured frame.
        """
        success, self.frame_captured = self.cap.read()
        if not success: 
            print(f"Image capture error.")
        return success

    def image_processing(self) -> None:
        """
        This function processes captured frames to detect and track an operator, extract features, and
        display the results.
        """
        try:
            # Find a person and build a bounding box around them, tracking them throughout the
            # experiment.
            results_people = self.tracking_processor.find_people(self.frame_captured)
            results_identifies = self.tracking_processor.identify_operator(results_people)
            
            # Cut out the bounding box for another image.
            projected_window = self.tracking_processor.track_operator(results_people, results_identifies, self.frame_captured)
            
            # Finds the operator's hand(s) and body
            self.hands_results, self.pose_results = self.feature.find_features(projected_window)
            
            # Draws the operator's hand(s) and body
            frame_results = self.feature.draw_features(projected_window, self.hands_results, self.pose_results)
            
            # Shows the skeleton formed on the body, and indicates which gesture is being 
            # performed at the moment.
            if self.mode == 'D':
                cv2.putText(frame_results, f"S{self.stage} N{self.num_gest+1}: {self.y_val[self.num_gest]} D{self.dist_virtual_point:.3f}" , (25,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 1, cv2.LINE_AA)
            elif self.mode == 'RT':
                cv2.putText(frame_results, f"S{self.stage} D{self.dist_virtual_point:.3f}" , (25,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, cv2.LINE_AA)
            cv2.imshow('RealSense Camera', frame_results)
            return  True
        except Exception as e:
            print(f"E1 - Error during operator detection, tracking or feature extraction: {e}")
            cv2.imshow('RealSense Camera', cv2.flip(self.frame_captured,1))
            self.hand_history = np.concatenate((self.hand_history, np.array([self.hand_history[-1]])), axis=0)
            self.wrists_history = np.concatenate((self.wrists_history, np.array([self.wrists_history[-1]])), axis=0)
            return False

    def extract_features(self) -> None:
        """
        The function `extract_features` processes hand and pose data to track specific joints and
        trigger gestures based on proximity criteria.
        """
        if self.stage == 0:
            try:
                # Tracks the fingertips of the left hand and centers them in relation to the center
                # of the hand.
                hand_ref = np.tile(self.gesture_analyzer.calculate_ref_pose(self.hands_results.multi_hand_landmarks[0], self.sample['joints_trigger_reference']), len(self.sample['joints_trigger']))
                hand_pose = [FeatureExtractor.calculate_joint_xy(self.hands_results.multi_hand_landmarks[0], marker) for marker in self.sample['joints_trigger']]
                hand_center = np.array([np.array(hand_pose).flatten() - hand_ref])
                self.hand_history = np.concatenate((self.hand_history, hand_center), axis=0)
            except:
                # If this is not possible, repeat the last line of the history.
                self.hand_history = np.concatenate((self.hand_history, np.array([self.hand_history[-1]])), axis=0)
            
            # Check that the fingertips are close together, if they are less than "dist" the 
            # trigger starts and the gesture begins.
            _, self.hand_history, self.dist_virtual_point = self.gesture_analyzer.check_trigger_enabled(self.hand_history, self.sample['par_trigger_length'], self.sample['par_trigger_dist'])
            if self.dist_virtual_point < self.sample['par_trigger_dist']:
                self.stage = 1
                self.dist_virtual_point = 1
                self.time_gesture = self.time_functions.tic()
                self.time_action = self.time_functions.tic()
        elif self.stage == 1:
            try:
                # Tracks the operator's wrists throughout the action
                track_ref = np.tile(self.gesture_analyzer.calculate_ref_pose(self.pose_results.pose_landmarks, self.sample['joints_tracked_reference'], 3), len(self.sample['joints_tracked']))
                track_pose = [FeatureExtractor.calculate_joint_xyz(self.pose_results.pose_landmarks, marker) for marker in self.sample['joints_tracked']]
                track_center = np.array([np.array(track_pose).flatten() - track_ref])
                self.wrists_history = np.concatenate((self.wrists_history, track_center), axis=0)
            except:
                # If this is not possible, repeat the last line of the history.
                self.wrists_history = np.concatenate((self.wrists_history, np.array([self.wrists_history[-1]])), axis=0)
            
            # Evaluates whether the execution time of a gesture has been completed
            if self.time_functions.toc(self.time_action) > 4:
                self.stage = 2
                self.sample['time_gest'] = self.time_functions.toc(self.time_gesture)
                self.t_classifier = self.time_functions.tic()

    def process_reduction(self) -> None:
        """
        The function `process_reduction` removes the zero's line from a matrix, applies filters, and
        reduces the matrix to a 6x6 matrix based on certain conditions.
        """
        # Remove the zero's line
        self.wrists_history = self.wrists_history[1:]
        
        # Test the use of filters before applying pca
        
        # Reduces to a 6x6 matrix 
        self.sample['data_reduce_dim'] = np.dot(self.wrists_history.T, self.wrists_history)

    def update_database(self) -> None:
        """
        This function updates a database with sample data and saves it in JSON format.
        
        Note: Exclusive function for database construction operating mode.
        """
        # Updates the sample
        self.sample['data_pose_track'] = self.wrists_history
        self.sample['answer_predict'] = self.y_val[self.num_gest]
        
        # Updates the database
        self.database[str(self.y_val[self.num_gest])].append(self.sample)
        
        # Save the database in JSON format
        self.file_handler.save_database(self.sample, self.database, os.path.join(self.current_folder, self.file_name_build))
        
        # Resets sample data variables to default values
        self.hand_history, _, self.wrists_history, self.sample = self.data_processor.initialize_data(self.dist, self.length)
        
        # Indicates the next gesture and returns to the image processing step
        self.num_gest += 1
        if self.num_gest == self.max_num_gest: 
            return True
        else: 
            return False

    def classify_gestures(self) -> None:
        """
        This function classifies gestures based on the stage and mode, updating predictions and
        resetting sample data variables accordingly.
        
        Note: Exclusive function for Real-Time operating mode
        """
        # Classifies the action performed
        self.y_predict.append(self.classifier.my_predict(self.sample['data_reduce_dim']))
        self.time_classifier.append(self.time_functions.toc(self.t_classifier))
        print(f"\nGesture: {self.y_predict[-1]}  and took {self.time_classifier[-1]:.3}ms to be classified.\n")
        
        # Resets sample data variables to default values
        self.hand_history, _, self.wrists_history, self.sample = self.data_processor.initialize_data(self.dist, self.length)
        # pause
        time.sleep(2)

##################################################################################################################################################
##################################################### EMOTIONS ###################################################################################
    
    def save_gesture(self, path: str):
        try:
            with open(path, 'a') as f:  # Abrir o arquivo no modo de 'append'
                f.write(f'{self.y_predict[-1]}\n')
        except Exception as e:
            print(f"Error: {e}")
        
    def save_emotions(self, path: str, emt):
        try:
            with open(path, 'a') as f:  # Abrir o arquivo no modo de 'append'
                f.write(f'{emt}\n')
        except Exception as e:
            print(f"Error: {e}")


    def get_emotion (self, img):
        height, width = img.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(img, (300, 300)),
            1.0,
            (300, 300),
            (104.0, 177.0, 123.0),
            swapRB=False,
            crop=False,
        )
        self.Emotion.face_model.setInput(blob)
        predictions = self.Emotion.face_model.forward()
        print('Face detected')

        for i in range(predictions.shape[2]):
            if predictions[0, 0, i, 2] > 0.5:
                bbox = predictions[0, 0, i, 3:7] * np.array(
                    [width, height, width, height]
                )
                (x_min, y_min, x_max, y_max) = bbox.astype("int")
                
                print(f"Bounding box coordinates: {x_min}, {y_min}, {x_max}, {y_max}")

                # draws red rectangle
                cv2.rectangle(
                    img, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2
                )

                face = img[y_min:y_max, x_min:x_max]
                # makes prediction 
                emotion = self.Emotion.recognize_emotion(face)
                print(f"Emotion detected: {self.emotions_list[emotion]}\n")

                im2 = cv2.flip(img, 1)
                # writes emotion name   
                cv2.putText(
                    im2,
                    self.emotions_list[emotion],
                    (x_min , y_min ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                
                font = cv2.FONT_HERSHEY_DUPLEX
                size = 0.8
                color_text = (0, 0, 255)
                thickness = 2

                # render results
                cv2.putText(
                    im2,
                    f'Resp.: {self.y_predict[-1]}',
                    (0, round(size*45)),
                    font, size, color_text, thickness
                )


            cv2.imshow('RealSense Camera', im2)
        try:
            return img, emotion
        except:
            return img, 0
    
    
    
    def classifier_emotion(self):
        _, emotion = self.get_emotion(self.frame_captured)
        
        match self.emotions_list[emotion]:
                        
            case "GOOD":        # confirms prediction
                self.counter["GOOD"] -= 1
                if self.counter["GOOD"] < 1:
                    print(f'\nSave Gesture: {self.y_predict[-1]}\n')
                    self.save_gesture(path= 'lists/gesture_txt.txt')
                    self.save_emotions(path= 'lists/emotions_txt.txt', emt=self.emotions_list[emotion])
                    self.stage = 0
                self.counter["BAD"] = 10     # reset other counter

            case "BAD":         # rejects prediction
                self.counter["BAD"] -= 1
                if self.counter["BAD"] < 1:
                    self.logger.info("\nDataSet Rejected\n")
                    self.save_emotions(path= 'lists/emotions_txt.txt', emt=self.emotions_list[emotion])
                    self.stage = 0
                self.counter["GOOD"] = 10
        