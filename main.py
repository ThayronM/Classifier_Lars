
from GestureEmotionCompiler import GestureEmotionCompiler
import os
from modules import *
from sklearn.neighbors import KNeighborsClassifier
import mediapipe as mp

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0' 

def main ():

    database = {'F': [], 'I': [], 'L': [], 'P': [], 'T': []}
    files_name= ['Datasets\DataBase_(5-10)_G.json',
                'Datasets\DataBase_(5-10)_H.json',
                'Datasets\DataBase_(5-10)_L.json',
                'Datasets\DataBase_(5-10)_M.json',
                'Datasets\DataBase_(5-10)_T.json',
                'Datasets\DataBase_(5-10)_1.json',
                'Datasets\DataBase_(5-10)_2.json',
                'Datasets\DataBase_(5-10)_3.json',
                'Datasets\DataBase_(5-10)_4.json',
                'Datasets\DataBase_(5-10)_5.json',
                'Datasets\DataBase_(5-10)_6.json',
                'Datasets\DataBase_(5-10)_7.json',
                'Datasets\DataBase_(5-10)_8.json',
                'Datasets\DataBase_(5-10)_9.json',
                'Datasets\DataBase_(5-10)_10.json'
                ]
    real_time_mode = ModeFactory.create_mode('real_time', files_name=files_name, database=database)
    mode = real_time_mode
    
    model = GestureEmotionCompiler(
        model_name = "resnet18.onnx",
        model_option = "onnx",
        backend_option = 2, #1
        providers = 1,
        fp16 = False,
        num_faces = 1,
        
        config=InitializeConfig(0,30),
        operation=mode,
        file_handler=FileHandler(),
        current_folder=os.path.dirname(__file__),
        data_processor=DataProcessor(), 
        time_functions=TimeFunctions(), 
        gesture_analyzer=GestureAnalyzer(),
        tracking_processor=YoloProcessor('yolov8n-pose.pt'), 
        feature=HolisticProcessor(
            mp.solutions.hands.Hands(
                static_image_mode=False, 
                max_num_hands=1, 
                model_complexity=1, 
                min_detection_confidence=0.75, 
                min_tracking_confidence=0.75
                ),
            mp.solutions.pose.Pose(
                static_image_mode=False, 
                model_complexity=1, 
                smooth_landmarks=True, 
                enable_segmentation=False, 
                smooth_segmentation=True, 
                min_detection_confidence=0.75, 
                min_tracking_confidence=0.75
                )
            ),
        classifier=KNN(
            KNeighborsClassifier(
                n_neighbors=mode.k, 
                algorithm='auto', 
                weights='uniform'
                )
            )
        )


    model.run()

if __name__ == "__main__":
    main()