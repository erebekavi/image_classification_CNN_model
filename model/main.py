
import keras
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

def load_img(img_path):
    # Load the trained model
    model = keras.models.load_model('model/CNN_model.h5')



    IMG_SIZE = (150, 150)  # Adjust this size to match the input size used during training

    def load_and_prepare_image(img_path):
        # Load and resize the image
        img = keras.preprocessing.image.load_img(img_path, target_size=IMG_SIZE)
        
        # Convert the image to a numpy array
        img_array = keras.preprocessing.image.img_to_array(img)
        
        # Expand dimensions to match the model input
        img_array = np.expand_dims(img_array, axis=0)
        
        # Normalize the image (if normalization was used during training)
        img_array /= 255.0
        
        return img_array

    

    # Prepare the image
    img_array = load_and_prepare_image(img_path)

    # Make predictions
    predictions = model.predict(img_array)

    # Get the predicted class index
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    print(f"Predicted class index: {predicted_class_index}")

    # Map the predicted index to class names
   
    def get_class_names_from_directory(directory):
        return sorted([d.name for d in os.scandir(directory) if d.is_dir()])


    class_names = get_class_names_from_directory('model/dataset/train')
    # Replace with your actual class names
    predicted_class_name = class_names[predicted_class_index]
    print(f"Predicted class name: {predicted_class_name}")




    # Display the image
    img = keras.preprocessing.image.load_img(img_path, target_size=IMG_SIZE)
    plt.imshow(img)
    plt.title(f"Predicted class: {predicted_class_name}")
    plt.axis('off')
    plt.show()

def take_img():
    
    cam = cv2.VideoCapture(0)

    cv2.namedWindow("capture")
    img_count = 0
    while True:
        ret,frame=cam.read()
        if not ret:
            print("failed")
            break
        cv2.imshow("capture",frame)

        k= cv2.waitKey(1)
        if k%256 == 27:
            print("close..")
            break
        if k%256 == 32:
            img_name = "photo.png"
            cv2.imwrite(img_name,frame)
            img_count+=1

    cam.release()
    cv2.destroyAllWindows()

    # Load the trained model
    model = keras.models.load_model('model/CNN_model.h5')

    IMG_SIZE = (150, 150)  # Adjust this size to match the input size used during training

    def load_and_prepare_image(img_path):
        # Load and resize the image
        img = keras.preprocessing.image.load_img(img_path, target_size=IMG_SIZE)
        
        # Convert the image to a numpy array
        img_array = keras.preprocessing.image.img_to_array(img)
        
        # Expand dimensions to match the model input
        img_array = np.expand_dims(img_array, axis=0)
        
        # Normalize the image (if normalization was used during training)
        img_array /= 255.0
        
        return img_array

    img_path = "model/photo.png"

    # Prepare the image
    img_array = load_and_prepare_image(img_path)

    # Make predictions
    predictions = model.predict(img_array)

    # Get the predicted class index
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    print(f"Predicted class index: {predicted_class_index}")

    # Map the predicted index to class names
   
    def get_class_names_from_directory(directory):
        return sorted([d.name for d in os.scandir(directory) if d.is_dir()])


    class_names = get_class_names_from_directory('model/dataset/train')
    # Replace with your actual class names
    predicted_class_name = class_names[predicted_class_index]
    print(f"Predicted class name: {predicted_class_name}")




    # Display the image
    img = keras.preprocessing.image.load_img(img_path, target_size=IMG_SIZE)
    plt.imshow(img)
    plt.title(f"Predicted class: {predicted_class_name}")
    plt.axis('off')
    plt.show()

load_img("model/cheesecake.jpg")
