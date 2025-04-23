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

1. Clone the repository:

```bash
git clone https://github.com/JaskiratSudan/Self-Driving-Mario-Kart-with-CNN-LSTM.git
cd Self-Driving-Mario-Kart-with-CNN-LSTM
```

2. Install required dependencies:

```bash
pip install -r requirements.txt
```

3. Install and run Mupen64Plus emulator:

**macOS:**

```bash
brew install mupen64plus
mupen64plus Mario\ Kart\ 64\ \(E\)\ \(V1.1\)\ \[\!\].z64
```

**Windows:**

- Download from [Mupen64Plus Website](https://mupen64plus.org/).
- Extract the zip file.
- Run from command prompt:

```bash
mupen64plus Mario\ Kart\ 64\ \(E\)\ \(V1.1\)\ \[\!\].z64
```
**Linux:**

```bash
sudo apt install mupen64plus
mupen64plus Mario\ Kart\ 64\ \(E\)\ \(V1.1\)\ \[\!\].z64
```

> **Important:** Position the emulator window at the top-left corner of your screen for correct screenshot capture.

## Usage

### Data Collection

Run the data collection script:

```bash
python record.py
```

- Ensure the emulator window is at the top-left corner.
- Focus the emulator window when prompted.

- Use GUI to set output directory and start/stop recording.
- Controls:
  - `Shift`: Accelerate
  - `Ctrl`: Brake
  - `Left/Right arrows`: Steering

### Training

- Organize your collected data into the `samples/` directory.
- Train your model:

```bash
python train.py samples/
```

- Continue training an existing model (optional):

```bash
python train.py samples/ existing_model.keras
```

### Autonomous Driving

- Start Mario Kart in the emulator:

```bash
mupen64plus Mario\ Kart\ 64\ \(E\)\ \(V1.1\)\ \[\!\].z64
```

- Run the autonomous driving script:

```bash
python autodrive.py
```

- Ensure the emulator window is at the top-left corner.
- Focus the emulator window when prompted.

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
