# Fish Freshness Estimator

A deep learning model and Streamlit web app that rates a fish as Highly Fresh, Fresh or Not Fresh from a photo of its eye. It aims to replace slow and subjective manual checks with a quick, repeatable one.

Final year project for the MSc and Engineer degree in Computer Science at ESI-SBA (Algeria), 2025, supervised by Dr Khaldi Belkacem. The full write-up is in [docs/fish-freshness-paper.pdf](docs/fish-freshness-paper.pdf).

<!-- Placeholder: add a screenshot of the app here -->

## How it works

- **Data:** the public Freshness of Fish Eyes (FFE) dataset, covering 8 species. The original 24 species and freshness classes were merged into 3 freshness classes and balanced to 1,043 training and 264 test images per class.
- **Model:** EfficientNetB3 pre-trained on ImageNet, with a new head (global average pooling, batch normalisation, dense layers of 512 and 256 units with dropout, and a 3-class softmax).
- **Training:** two phases. The head is trained first with the backbone frozen, then the top 50 backbone layers are fine-tuned at a lower learning rate. Training uses augmentation, early stopping and learning-rate reduction.
- **Iteration:** on the original 24 classes, EfficientNetB0 reached about 41% validation accuracy, ResNet50V2 stayed below 60% and EfficientNetB3 reached about 77%. Merging into 3 balanced classes gave the final model.
- **App:** upload a fish image to get the predicted class, the confidence for each class and an optional Grad-CAM heatmap showing which parts of the image drove the prediction.

## Results

On the held-out test set the final model reached **81.06% accuracy**, with an AUC above 0.90 for all three classes. Most errors were between Fresh and Not Fresh.

## Tech

- Python, TensorFlow and Keras
- EfficientNetB3, transfer learning, Grad-CAM
- Streamlit, OpenCV, Pillow, NumPy

## Run it locally

The trained model (`final_best_model.keras`, about 118 MB) is too large for the repository. Download it from the [latest release](../../releases/latest) and put it next to `app.py`.

```bash
pip install -r requirements.txt
streamlit run app.py
```

The training notebook is not included in this repository.

---

Ahmed Amine Tiffrent · [Portfolio](https://www.aminetiffrent.com/portfolio) · [LinkedIn](https://www.linkedin.com/in/ahmed-tiffrent-982275285)
