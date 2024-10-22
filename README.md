# Generate Traffic Data from Multiple Cars Using Carla Simulator

This repository contains source code for generating traffic data, including images and bounding boxes for cars and pedestrians, using the Carla Simulator.

This code has been utilized to train object detection models in our published paper: [Driving Towards Efficiency: Adaptive Resource-Aware Clustered Federated Learning in Vehicular Networks](https://ieeexplore.ieee.org/abstract/document/10578208).

## Overview

The data is generated from cameras mounted on a selected number of vehicles simultaneously. It is intended for training object detection models, particularly in a federated learning setup where all nodes (vehicles) train local models that are subsequently aggregated by a central server.

With this source code, researchers can modify environmental conditions and select from four predefined scenarios: **Clear Day**, **Rainy Day**, **Clear Night**, and **Rainy Night**. Example images of these different scenarios can be found below. Additionally, the configurations can be extended via the `situation_config.py` file.

You can also customize the number of pedestrians and cars, as well as the data collectors (cars) involved in the simulation.

## Getting Started

### Prerequisites

1. **Install Carla Simulator**: Follow the installation guide here: [Carla Quickstart](https://carla.readthedocs.io/en/latest/start_quickstart/).

### Steps to Generate Traffic Data

1. **Run Carla Simulator**: Start the Carla server.
  
2. **Reset Files and Folders**: Execute `reset_files.py` to clear any previous data.

3. **Configure Situations**: Modify the environmental settings using `situation_config.py`.

4. **Generate Training Data**: Run `generate_traffic.py` to start collecting data.

5. **Extract Bounding Boxes**: Once you have collected the required training data, stop `generate_traffic.py` and run `extract.py` to extract bounding boxes, which will be stored in the `VehicleBBox` and `PedestrianBBox` folders.

## Example Images

![Clear Day](https://github.com/AhmadMkhalil/Generate-Traffic-Data-From-Multiple-Cars-Using-Carla-Simunlator/blob/main/situations/clearDay.png)

![Clear Night](https://github.com/AhmadMkhalil/Generate-Traffic-Data-From-Multiple-Cars-Using-Carla-Simunlator/blob/main/situations/clearNight.png)

![Rainy Day](https://github.com/AhmadMkhalil/Generate-Traffic-Data-From-Multiple-Cars-Using-Carla-Simunlator/blob/main/situations/rainyDay.png)

![Rainy Day](https://github.com/AhmadMkhalil/Generate-Traffic-Data-From-Multiple-Cars-Using-Carla-Simunlator/blob/main/situations/rainyNight.png)


## License


## Acknowledgments

