# yolov8-streamlit-app
* Model used: https://github.com/ultralytics/ultralytics

# App Features
* Object detection/segmentation using pre-trained yoloV8 model (trained on COCO dataset)- Refer to coco_classnames.txt for the list of objects detectable using the base model
* Custom-trained yolov8 model to detect potholes (mAP50:**0.721**, mAP50-95:**0.407**)
* Custom-trained yolov8 model to detect car license plates (mAP50:**0.995**, mAP50-95:**0.828**)
* Integrated license plate detector with EasyOCR to read license plates (With Image preprocessing function to handle images with brightness and Image glare issues)
* Custom-trained yolov8 model to PPE (7 classes: ['Protective Helm', 'Shield', 'Jacket', 'Dust Mask', 'Eye Wear, 'Glove, 'Protective Boots')
* To use your custom trained model, just add your .pt files into the weights and make some minor changes to the settings.py and app.py files

# Possible improvements
* Improve the accuracy of the custom-trained models (train on higher quality data, Data augmentation, Hyperparameter tuning(computationally intensive, time-consuming)
* Integrate car license plate detector with SORT/DeepSORT which keep tracks of the car's information. (For real-world use case)

# Issues
* Currently webcam feature isnt working after deploying to streamlit cloud but it works locally.

  

