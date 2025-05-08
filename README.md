# Lip Reading with Deep Learning

This project investigates two deep learning–based methods for visual speech recognition using the GRID Corpus dataset. The goal is to recognize spoken language by analyzing sequences of lip movements from video frames. We explore both word-level and phoneme-level prediction models to understand their effectiveness in different granularity settings.

## Project Structure

```
DA526-End-Project/
├── lipread-4.ipynb          # Main Jupyter notebook with training and evaluation
├── assets/                  # Graphs and model architecture diagrams
├── models/                  # Saved model checkpoints (optional)
└── README.md                # Project documentation (this file)
```


## Dataset

- **Name**: GRID Corpus  
- **Description**: Audio-visual speech dataset with fixed sentence structures  
- **Speakers**: 34 individuals  
- **Format**: 25 fps video of frontal facial views  
- **Preprocessing**:
  - Lip region extraction using face detection
  - Conversion to grayscale and resizing
  - Normalization of pixel values
  - Fixed number of frames per word or sentence for input uniformity

## Methodology

### Approach 1: Time-shared CNN + LSTM (Word-Level Classification)

This architecture classifies individual words using a time-distributed CNN followed by an LSTM for temporal modeling.

- **Input**: A fixed-length video sequence for each word  
- **CNN Encoder**: 2D convolutions applied to each frame individually (shared weights)  
- **Temporal Layer**: Single-layer LSTM to model time dependency between frames  
- **Decoder**: Dense layer with softmax activation to classify among vocabulary words  
- **Loss Function**: Categorical cross-entropy  
- **Embedding**: Pretrained GloVe embeddings used in the decoder for better generalization  

### Approach 2: 3D CNN + BiLSTM + CTC (Phoneme-Level Prediction)

This model is designed for phoneme-level sequence decoding of entire sentences. It leverages spatiotemporal feature extraction and temporal modeling.

- **Input**: Video sequence of a complete sentence  
- **CNN Encoder**: 3D convolutional layers to capture both spatial and temporal features  
- **Temporal Layer**: Bidirectional LSTM layers to learn forward and backward context  
- **Output**: Per-timestep phoneme probabilities  
- **Loss Function**: Connectionist Temporal Classification (CTC) for alignment-free sequence training  

## Results

| Model                         | Input Level | Accuracy | Loss    | Notes                                      |
|------------------------------|-------------|----------|---------|--------------------------------------------|
| Time-shared CNN + LSTM       | Word        | ~99%     | 0.02    | High accuracy due to small vocabulary size |
| 3D CNN + LSTM                | Word        | ~98.5%   | 0.03    | Slightly slower convergence                |
| Seq2Seq + GloVe              | Sentence    | ~88%     | 1.58    | Cross-entropy loss                         |
| 3D CNN + BiLSTM + CTC        | Sentence    | ~84%     | --      | Used CTC loss (no cross-entropy reported)  |

Both sentence-level models (Approach 1 and 2) were developed based on the success of preliminary word-level classifiers.

## Dependencies

- Python 3.8+
- TensorFlow or PyTorch
- NumPy
- OpenCV
- scikit-learn
- tqdm
- matplotlib

Install all dependencies using:

```bash
pip install -r requirements.txt
````

## How to Run

Clone the repository and open the notebook:

```bash
git clone https://github.com/satyam-kumar2/DA526-End-Project.git
cd DA526-End-Project
jupyter notebook lipread-4.ipynb
```

Make sure the preprocessed dataset is available locally and paths are correctly set inside the notebook.

## Future Work

* Integrate attention mechanisms for better alignment
* Evaluate model generalization across unseen speakers
* Explore transformer-based visual encoders
* Optimize models for real-time lip reading

## References

* Assael et al., *LipNet: End-to-End Sentence-Level Lipreading*, 2016
* Graves et al., *Connectionist Temporal Classification*, 2006
* GRID Corpus: [https://spandh.dcs.shef.ac.uk/gridcorpus/](https://spandh.dcs.shef.ac.uk/gridcorpus/)

## License

This repository is intended for academic use only. Redistribution or commercial use of the GRID corpus must comply with the dataset's original license.

## Author

**Satyam Kumar**
GitHub: [https://github.com/satyam-kumar2](https://github.com/satyam-kumar2)
Email: satyam.sk343413 \[at] email.com

