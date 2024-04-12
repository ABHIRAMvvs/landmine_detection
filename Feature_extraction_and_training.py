import os
import cv2
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Define function to calculate statistical moments
def calculate_moments(image):
    mean = np.mean(image)
    std_dev = np.std(image)
    if std_dev == 0:
        kurtosis = 0  # Handle case where standard deviation is zero
        skewness = 0  # Handle case where standard deviation is zero
    else:
        kurtosis = np.mean((image - mean) ** 4) / (std_dev ** 4) - 3
        skewness = np.mean((image - mean) ** 3) / (std_dev ** 3)
    return mean, std_dev, kurtosis, skewness

# Define function to extract features from ROI
def extract_features(roi):
    moments = calculate_moments(roi)
    min_val = np.min(roi)
    max_val = np.max(roi)

    # Calculate histogram
    hist = cv2.calcHist([roi.astype(np.uint8)], [0], None, [256], [0, 256])
    hist = hist.ravel() / hist.sum()  # Normalize histogram

    # Calculate GLCM features from histogram
    bins = 256
    graycomatrix = np.zeros((bins, bins))
    for i in range(bins):
        for j in range(bins):
            graycomatrix[i, j] = np.sum(hist[i] * hist[j])

    # Extract texture features
    contrast = np.sum(graycomatrix * (np.arange(bins) - np.mean(graycomatrix)) ** 2)
    energy = np.sum(graycomatrix ** 2)
    homogeneity = np.sum(graycomatrix / (1 + np.abs(np.arange(bins) - np.arange(bins)[:, np.newaxis])))

    return np.concatenate([moments, [min_val, max_val, contrast, energy, homogeneity]])

# Define function to load images from folder
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            images.append(img)
    return images

# Load landmine and non-landmine images
landmine_folder = r'C:\Users\Saiab\Desktop\landmine_detection\feature extraction\landmine'
non_landmine_folder = r'C:\Users\Saiab\Desktop\landmine_detection\feature extraction\non landmine'
landmine_images = load_images_from_folder(landmine_folder)
non_landmine_images = load_images_from_folder(non_landmine_folder)

# Extract features from images
landmine_features = [extract_features(img) for img in landmine_images]
non_landmine_features = [extract_features(img) for img in non_landmine_images]

# Create labels for landmine and non-landmine images
landmine_labels = np.ones(len(landmine_features))
non_landmine_labels = np.zeros(len(non_landmine_features))

# Combine features and labels
all_features = np.concatenate([landmine_features, non_landmine_features])
all_labels = np.concatenate([landmine_labels, non_landmine_labels])

# Normalize features
scaler = MinMaxScaler()
normalized_features = scaler.fit_transform(all_features)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(normalized_features, all_labels, test_size=0.2, random_state=42)

# Train MLP classifier
mlp_classifier = MLPClassifier(hidden_layer_sizes=(15,), activation='logistic', solver='lbfgs', random_state=42)
mlp_classifier.fit(X_train, y_train)

# Save the trained classifier
import joblib
joblib.dump(mlp_classifier, 'mlp_classifier.pkl')
 