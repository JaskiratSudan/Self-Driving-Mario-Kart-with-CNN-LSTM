# Self-Driving Mario Kart with CNN-LSTM

https://github.com/user-attachments/assets/a41ad2d9-a68f-4a46-8d62-57a0f13db3e8

This project implements an autonomous driving system for Mario Kart using a Convolutional Neural Network (CNN) combined with Long Short-Term Memory (LSTM) networks. The system captures screen input, processes it through a trained model, and automatically controls the game using keyboard inputs.

## Features

- Real-time screen capture and processing
- CNN-LSTM model for temporal sequence understanding
- Automatic keyboard control for game actions
- Data collection and training capabilities
- Visual feedback during operation

## Requirements

- Python 3.7+
- TensorFlow 2.12.0 or higher
- Mupen64Plus emulator (for Mario Kart 64)
- Dependencies listed in `requirements.txt`

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/Self-Driving-Mario-Kart-with-CNN-LSTM.git
cd Self-Driving-Mario-Kart-with-CNN-LSTM
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Data Collection
1. Run the data collection script:
```bash
python record.py
```
2. Use the GUI to:
   - Set output directory
   - Start/stop recording
   - Control the game using:
     - Shift: Accelerate
     - Ctrl: Brake
     - Left/Right arrows: Steering

### Training
1. Organize your collected data in the samples directory
2. Run the training script:
```bash
python train.py <samples_folder>
```
3. Optional: Continue training from an existing model:
```bash
python train.py <samples_folder> <existing_model.keras>
```

### Autonomous Driving
1. Start Mario Kart 64 in Mupen64Plus
2. Run the autonomous driving script:
```bash
python autodrive.py
```
3. Focus the Mupen64Plus window when prompted
4. The system will automatically control the game

## Configuration

Key configurations can be adjusted in the respective Python files:

- `autodrive.py`: Control parameters, FPS, and screen capture region
- `train.py`: Model architecture, training parameters, and data processing
- `record.py`: Recording settings and key mappings

## Project Structure

- `autodrive.py`: Main autonomous driving script
- `train.py`: Model training script
- `record.py`: Data collection script
- `utils.py`: Utility functions
- `requirements.txt`: Python dependencies
- `samples/`: Directory for training data
- `models/`: Directory for saved models

## Contributing

Feel free to submit issues and enhancement requests. Pull requests are welcome!

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Inspired by autonomous driving research
- Built using TensorFlow and Python
- Special thanks to the open-source community
