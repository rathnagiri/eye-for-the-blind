# eye-for-the-blind
## Problem Statement
In this capstone project, the goal is to create a deep learning model that can explain the contents of an image in the form of speech. The caption generation process uses an attention mechanism on the Flickr8K dataset. This solution aims to assist blind people in understanding the contents of any image through generated captions converted into speech.

The model employs both deep learning and natural language processing techniques. A CNN-based encoder extracts features from the image, which are then decoded by an RNN model to generate captions.

This project is an extended application of the Show, Attend and Tell: Neural Image Caption Generation with Visual Attention paper.

## Dataset
The dataset used in this project is the Flickr8K dataset, available on Kaggle. It contains 8,000 images, each paired with five different captions that describe the contents of the image. This dataset is ideal for training models that aim to generate descriptive captions for images.

## Project Pipeline Architecture
The following steps outline the project pipeline architecture that has been implemented:

1. **Data Understanding & Cleaning**:
    - Image and caption data are cleaned and preprocessed to prepare them for the next stages.
   
2. **Image Preprocessing**:
    - Images are preprocessed by resizing and applying necessary transformations for compatibility with the feature extraction model.
   
3. **Caption Preprocessing**:
    - Captions are tokenized and cleaned, making them ready for input into the decoder.

4. **Image Preprocessing Layer**:
    - Images are loaded, resized, and preprocessed using InceptionV3 for feature extraction.

5. **Feature Extraction (Encoder)**:
    - InceptionV3 is used as the feature extractor, and the reshaped image features are fed into the encoder.

6. **Decoder with Attention Mechanism**:
    - The decoder leverages an attention mechanism to generate captions based on the encoded features of the image.

7. **Caption Generation**:
    - A greedy search is used to generate captions by predicting the next word in the sequence until the caption is complete.

8. **Evaluation with BLEU Scores**:
    - The generated captions are evaluated against the ground truth captions using BLEU scores to measure the accuracy of the captions.

9. **Audio Generation**:
    - The generated caption is converted to speech using the **gTTS** (Google Text-to-Speech) library.

10. **Visualization**:
    - The image and its corresponding attention map are visualized to show the areas the model focused on when generating the caption.

## Requirements
1. Python 3.x
2. TensorFlow
3. Matplotlib
4. PIL (Pillow)
5. gTTS (Google Text-to-Speech)
6. NLTK
7. NumPy
8. Pandas

## Conclusion
This project provides a deep learning-based solution for image captioning that can be extended for real-world applications, particularly for helping visually impaired individuals understand the content of images using speech. By combining convolutional neural networks (CNNs), recurrent neural networks (RNNs), and attention mechanisms, the model can generate detailed and contextually accurate captions for images.
