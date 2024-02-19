# Anti Drone System Through Machine & Deep Learning

The extraction of a cost-effective and high-performance model capable of accurately distinguishing drones from non-drone objects. To ensure a comprehensive evaluation, Simulated noise with different variances to low-quality drone pictures within the FMCW datasets. As a result, GoogLeNet, MobileNetV2, SVC, and GBDT techniques maintained accuracy above 0.9 until a noise distribution of 1e-3. Notably, MobileNetV2 and Linear SVC exhibited the highest performance.

## Datasets
Datasets were uploaded and used through HuggingFace
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
  
## OUTLINE

1. Based on FMCW radar data, we use machine learning and CNN models to learn the patterns of radar images

2. Using the trained models, we accurately distinguish and identify drones from non-drone ones.

3. We evaluate each model based on different metrics such as size, inference speed, and accuracy.

4. Based on the evaluation results, we finally select a cost-effective, high-performance model.

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code
of conduct, and the process for submitting pull requests to us.

## Versioning

We use [Semantic Versioning](http://semver.org/) for versioning. For the versions
available, see the [tags on this
repository](https://github.com/PurpleBooth/a-good-readme-template/tags).

## Authors

  - **Billie Thompson** - *Provided README Template* -
    [PurpleBooth](https://github.com/PurpleBooth)

See also the list of
[contributors](https://github.com/PurpleBooth/a-good-readme-template/contributors)
who participated in this project.

## License

This project is licensed under the [CC0 1.0 Universal](LICENSE.md)
Creative Commons License - see the [LICENSE.md](LICENSE.md) file for
details

## Acknowledgments

  - Hat tip to anyone whose code is used
  - Inspiration
  - etc
