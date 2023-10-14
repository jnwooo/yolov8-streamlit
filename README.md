# yolov8-streamlit-app
* Model used: https://github.com/ultralytics/ultralytics
* App link: https://yolov8-app-object-detection.streamlit.app/

# App Features 
* Object detection/segmentation using pre-trained yoloV8 model (trained on COCO dataset) - Refer to coco_classnames.txt for the list of objects detectable using the base model
* Custom-trained yolov8 model to detect potholes (mAP50:**0.721**, mAP50-95:**0.407**) - Using yolov8m.pt
* Custom-trained yolov8 model to detect car license plates (mAP50:**0.995**, mAP50-95:**0.828**) - Using yolov8m.pt
* Integrated license plate detector with EasyOCR to read license plates (With Image preprocessing function to handle images with brightness and Image glare issues)
* Custom-trained yolov8 model to PPE (7 classes: ['Protective Helm', 'Shield', 'Jacket', 'Dust Mask', 'Eye Wear, 'Glove, 'Protective Boots')
* To use your custom trained model, just add your .pt files into the weights and make some minor changes to the settings.py and app.py files **(Note: If your model's weights are >25mb, you require Git LFS to upload your files)**

# Possible improvements
* Improve the accuracy of the custom-trained models (train on higher quality data, Data augmentation, Hyperparameter tuning(computationally intensive, time-consuming)
* Integrate car license plate detector with SORT/DeepSORT which keep tracks of the car's information. (For real-world use case)
* Experiment with using different size yolov8 models (smaller models offer faster inference but less accuracy), smaller size models may be more suitable if you're deploying your app on Streamlit's Community Cloud

# Issues
* Currently webcam feature isnt working after deploying to streamlit cloud but it works locally.
* App is currently deployed at Streamlit's Community Cloud which has limited resource, which may crash the app if the resources are exceeded.
* Video processing are slow running on Streamlit Cloud (Deploying the app on a paid-cloud service will help with the processing speed)




  

