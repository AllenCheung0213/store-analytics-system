import cv2

# Pre-defined labels for age buckets and genders
AGE_BUCKETS = ["0-2", "4-6", "8-13", "15-20", "25-32", "38-43", "48-53", "60+"]
GENDERS = ["Male", "Female"]

# Load models once
age_net = cv2.dnn.readNetFromCaffe(
    "models/age_deploy.prototxt", "models/age_net.caffemodel"
)
gender_net = cv2.dnn.readNetFromCaffe(
    "models/gender_deploy.prototxt", "models/gender_net.caffemodel"
)


def predict_age_gender(face_img):
    """
    Predicts an age bucket and gender for an aligned 227×227 face image.
    """
    blob = cv2.dnn.blobFromImage(
        face_img, 1.0, (227, 227),
        (78.4263377603, 87.7689143744, 114.895847746),
        swapRB=False
    )
    # Gender
    gender_net.setInput(blob)
    gender_preds = gender_net.forward()
    gender = GENDERS[gender_preds[0].argmax()]
    # Age
    age_net.setInput(blob)
    age_preds = age_net.forward()
    age = AGE_BUCKETS[age_preds[0].argmax()]
    return age, gender