import cv2
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
import joblib
from tkinter import Tk, filedialog

# Load the trained MLP classifier
mlp_classifier = joblib.load('mlp_classifier.pkl')

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

def classify_roi_batch(roi_batch, classifier):
    # Extract features from ROIs
    roi_features = [extract_features(roi) for roi in roi_batch]

    # Normalize features
    scaler = MinMaxScaler()
    normalized_features = scaler.fit_transform(roi_features)

    # Predict labels for ROIs
    predictions = classifier.predict(normalized_features)

    return predictions

def detect_landmines(image, step_size, window_size, classifier):
    # Initialize binary image
    binary_image = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

    # Collect ROIs for batch processing
    roi_batch = []

    # Process sliding windows
    for y in range(0, image.shape[0] - window_size[1] + 1, step_size):
        for x in range(0, image.shape[1] - window_size[0] + 1, step_size):
            # Extract ROI
            roi = image[y:y + window_size[1], x:x + window_size[0]]
            
            # Store ROI for batch processing
            roi_batch.append(roi)

            # Classify batch of ROIs
            if len(roi_batch) == 100:
                predictions = classify_roi_batch(roi_batch, classifier)

                # Update binary image
                for i, prediction in enumerate(predictions):
                    if prediction == 1:  # If landmine
                        # Calculate position in binary image
                        bx = x + i % ((min(image.shape[1], x + window_size[0]) - x) // step_size)
                        by = y + i // ((min(image.shape[1], x + window_size[0]) - x) // step_size)
                        
                        # Update binary image if within bounds
                        if by < binary_image.shape[0] and bx < binary_image.shape[1]:
                            binary_image[by, bx] += 1

                # Clear ROI batch
                roi_batch.clear()

    # Threshold binary image based on classification criteria
    binary_image[binary_image < 3] = 0
    binary_image[binary_image >= 3] = 255

    return binary_image


def select_image():
    # Open file dialog to select image
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()
    root.destroy()
    return file_path

# Load your original thermal image
image_path = select_image()

if image_path:
    original_image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    # Define sliding window parameters
    step_size = 4
    window_size = (16, 16)

    # Detect landmines
    binary_image = detect_landmines(original_image, step_size, window_size, mlp_classifier)

    # Find contours and draw them on original image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    result_image = cv2.drawContours(original_image.copy(), contours, -1, (0, 255, 0), 2)

    # Display the result
    cv2.imshow("Detection Result", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
