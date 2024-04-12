import cv2
import numpy as np
from tkinter import Tk, filedialog
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
import joblib

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

def classify_roi(roi, classifier):
    # Extract features from ROI
    roi_features = extract_features(roi.reshape(-1))

    # Normalize features
    scaler = MinMaxScaler()
    normalized_features = scaler.fit_transform([roi_features])

    # Predict label for ROI
    prediction = classifier.predict(normalized_features)

    return prediction[0]

def generate_binary_image(image, step_size, window_size, classifier):
    binary_image = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)

    # Slide window across the image
    for (x, y, window) in sliding_window(image, step_size, window_size):
        # Classify ROI
        prediction = classify_roi(window, classifier)

        # Assign label to ROI region in binary image
        if prediction == 1:  # If landmine
            binary_image[y:y + window_size[1], x:x + window_size[0]] += 1

    # Threshold binary image based on classification criteria
    binary_image[binary_image < 3] = 0
    binary_image[binary_image >= 3] = 255

    return binary_image

def find_contours(binary_image, original_image):
    # Find contours in binary image
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on original image
    cv2.drawContours(original_image, contours, -1, (0, 255, 0), 2)

    return original_image

def sliding_window(image, step_size, window_size):
    # Slide a window across the image
    for y in range(0, image.shape[0] - window_size[1] + 1, step_size):
        for x in range(0, image.shape[1] - window_size[0] + 1, step_size):
            # Yield the current window
            yield (x, y, image[y:y + window_size[1], x:x + window_size[0]])

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

    # Initialize binary image
    binary_image = np.zeros((original_image.shape[0], original_image.shape[1]), dtype=np.uint8)

    # Generate binary image for the selected image
    binary_image += generate_binary_image(original_image, step_size, window_size, mlp_classifier)

    # Threshold binary image based on classification criteria
    binary_image[binary_image < 3] = 0
    binary_image[binary_image >= 3] = 255

    # Find contours and draw them on original image
    result_image = find_contours(binary_image, original_image)

    # Display the result
    cv2.imshow("Detection Result", result_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

