Chest X-Ray Classification using CLIP

Overview
Multi-label classification system for automated chest X-ray diagnosis using OpenAI's CLIP vision-language model and PyTorch.

Project Description
This project implements a deep learning pipeline for detecting multiple pathologies in chest X-ray images from the CheXpert dataset. The model combines visual and textual information using CLIP embeddings to classify 9 different medical conditions.

Features
- Multi-label classification for 9 pathologies (Pneumonia, Cardiomegaly, Edema, etc.)
- CLIP-based image and text embedding fusion
- Custom PyTorch DataLoader for efficient batch processing
- Class imbalance handling using weighted BCE loss
- Automated radiology report generation
- Model evaluation with ROC-AUC, F1-score, and accuracy metrics

Technologies Used
- Python 3.12
- PyTorch
- Transformers (Hugging Face)- CLIP model
- Pandas & NumPy- Data processing
- **Scikit-learn** - Model evaluation
- Matplotlib - Visualization
- PIL - Image processing

Dataset
- CheXpert: Large chest X-ray dataset with 200,000+ images
- Labels: 9 pathology categories including Pneumonia, Cardiomegaly, Pleural Effusion, etc.

Model Architecture
1. CLIP Model (openai/clip-vit-base-patch32) - Frozen pretrained encoder
2. Image Embeddings: 512-dimensional vectors
3. Text Embeddings: 512-dimensional vectors
4. Fusion Layer: Concatenated image + text embeddings (1024-dim)
5. Classifier: Custom neural network with dropout for regularization

Results
- Successfully trained multi-label classifier on large-scale medical imaging dataset
- Implemented class imbalance handling for improved minority class detection
- Generated automated diagnostic reports from model predictions
- Visualized actual vs predicted labels for model validation


