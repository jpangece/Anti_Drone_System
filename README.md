# Anti Drone System Through Machine & Deep Learning (2023)

The extraction of a cost-effective and high-performance model capable of accurately distinguishing drones from non-drone objects. To ensure a comprehensive evaluation, Simulated noise with different variances to low-quality drone pictures within the FMCW datasets. As a result, GoogLeNet, MobileNetV2, SVC, and GBDT techniques maintained accuracy above 0.9 until a noise distribution of 1e-3. Notably, MobileNetV2 and Linear SVC exhibited the highest performance.

## **Datasets**
Datasets were uploaded and used through HuggingFace
Provides Task with Dataset Builder

  - HuggingFace(https://huggingface.co/Goorm-AI-04)
    
### Drone RF (Radio Frequency) Datasets

Drone RF: Radio signals used to communicate and control drones

Application: Used to identify drones, track locations, and analyze flight patterns of unmanned aerial vehicles

### Micro-Doppler Drone Detection Datasets

Micro Doppler: Technology to measure the Doppler effect due to the microscopic movement of an object Used to detect microscopic movements such as the rotation of a drone's propeller or rotor

Application: Used for the detection and classification of drones and the analysis of minute movements

### Real Doppler RAD-DAR database
  - (https://www.kaggle.com/datasets/iroldan/real-doppler-raddar-database)
  - Datasets recorded by the RAD-DAR radar system, which is widely used worldwide
    
### FMCW (Frequency Modulated Continuous Wave) Datasets (SELECTED)

FMCW Radar: Radar system that uses frequency modulated continuous waves to measure target distance and velocity simultaneously. Dataset of Actual Measurements for 17,000 Drones, Cars, People.

Application: Used to detect and track drones, planes, vehicles, etc., and to measure speed and distance of high-speed targets

Reason for selection: Radar method that can be secured from Kaggle and is widely used

  - [Datasets](https://ieee-dataport.org/open-access/drone-remote-controller-rf-signal-dataset)
  
## **OUTLINE**

1. Based on FMCW radar data, we use machine learning and CNN models to learn the patterns of radar images

2. Using the trained models, we accurately distinguish and identify drones from non-drone ones.

3. We evaluate each model based on different metrics such as size, inference speed, and accuracy.

4. Based on the evaluation results, we finally select a cost-effective, high-performance model.

## **Additional Experiment with **NOISE** added on the Image Datasets**

Accuracy experiments are conducted with noise-added data for ResNet101, GoogLeNet, MobileNetV2 and SVC, Linear SVC, Gradient Boosted Decision Tree, Random Forest techniques

![노이즈](https://github.com/jpangece/Anti_Drone_System/assets/122253772/659ee9b1-8a0a-4beb-b6fd-0992e47b3a35)

### RESULT

GoogLeNet, MobileNetV2, SVC, and Gradient Boosted Decision Tree techniques maintain accuracy above 0.9 until noise variance 1e-3. For more noise, MobileNetV2 and Linear SVC maintain the highest performance

![노이즈2](https://github.com/jpangece/Anti_Drone_System/assets/122253772/6452d63b-66d3-40a6-83be-84a95ae7c4c1)

## **CONCLUSION**
### Performance Centered 
GoogLeNet is suitable for performance-driven tasks with high accuracy and moderate inference speed

### velocity center
Since the Gradient Boosted Decision Tree shows good performance on the CPU, it is suitable for real-time or high throughput needs and limited hardware environments.

### Considering complex requirements
Considering all the accuracy, speed, and model size, GoogLeNet and SqueezeNet1.1 provide balanced performance.

![image](https://github.com/jpangece/Anti_Drone_System/assets/122253772/a99fd344-3edd-40c2-81ae-18354168bf66)


## **IMPORTANCE**

- With Micro Dopper Signature data, it is possible to obtain more information while being economical in H/W without expensive high-performance cameras.

- Unlike conventional CW radars, it can be seen that even location information can be grasped using FMCW radar data that modulates frequency.

- The FMCW radar dataset was approached as a classification task and its performance was compared by applying representative CNN techniques, and compared with mathematically verified classification ML techniques to know accurate performance.

- As a result of the experiment, as the quality of the dataset is high, most CNN and ML techniques showed classification accuracy close to 95%. Among the ML techniques, the Random Forest Classifier method showed very fast computational speed. Among the deep learning techniques, ILSVRC-2014 showed the highest accuracy in GoogLeNet, which guaranteed some performance, and SqueezeNet 1.1 specialized in weight reduction showed similar accuracy to other models even with few parameters.
