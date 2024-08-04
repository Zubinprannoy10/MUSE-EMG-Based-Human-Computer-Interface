# MUSE : EMG based Human Computer Interface 

This project implements a silent speech recognition system using EMG (Electromyography) signals. It records EMG data from an OpenBCI Cyton board, processes the signals, predicts words from silent speech, and uses these predictions to perform web searches or tell the time.

## Components

1. **EMG Recording** (`record_emg.py`): Records EMG data using an OpenBCI Cyton board.

2. **Signal Preprocessing** (`preprocess_emg.py`): Preprocesses the recorded EMG signals.

3. **Model Training** (`train_model.py`): Trains a PyTorch model on the preprocessed EMG data.

4. **Word Prediction** (`predict_word.py`): Uses the trained model to predict words from silent speech EMG signals.

5. **User Interface** (`interface.py`): Provides a command-line interface for interacting with the system.

6. **Search Engine** (`search_engine.py`): Uses predicted words to perform web searches or tell the time.

## Environment Setup

It is recommended to use a conda environment for this project. The project has been developed and tested with Python 3.9 and CUDA 12.2.

## Setup

1. Clone this repository:
git clone https://github.com/Zubinprannoy10/Silent_speech_OpenBCI.git
cd Silent_speech_OpenBCI

2. Install the required dependencies:
pip install -r requirements.txt

3. Ensure you have the OpenBCI Cyton board or anyother biosensing board having support for brainflow properly connected.

## Usage

1. Record EMG data:

    ```bash
    python record_emg.py
    ```

2. Preprocess the recorded data:

    ```bash
    python preprocess_emg.py
    ```

3. Train the model:

    ```bash
    python train_model.py
    ```

4. Run the interface to record, predict, and search:

    ```bash
    python interface.py
    ```
Note: Make sure to include your Google api key and  search engine id in the search_engine.py script.

## Contributing

Contributions to this project are welcome. Please fork the repository and submit a pull request with your changes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
