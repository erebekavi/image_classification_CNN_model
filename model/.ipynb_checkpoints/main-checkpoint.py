
import keras
import numpy as np
import matplotlib.pyplot as plt
import os

# Load the trained model
model = keras.models.load_model('my_model.h5')



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

# Path to the new image
img_path = 'chickencurry.jpg'

# Prepare the image
img_array = load_and_prepare_image(img_path)

# Make predictions
predictions = model.predict(img_array)

# Get the predicted class index
predicted_class_index = np.argmax(predictions, axis=1)[0]
print(f"Predicted class index: {predicted_class_index}")

# Map the predicted index to class names
'''
class_names = ["apple_pie","baby_back_ribs","baklava","beef_carpaccio","beef_tartare","beet_salad","beignets","bibimbap","bread_pudding"
               ,"breakfast_burrito","bruschetta","caesar_salad","cannoli","caprese_salad","carrot_cake","ceviche","cheesecake","cheese_plate",
               "chicken_curry","chicken_quesadilla","chicken_wings","chocolate_cake","chocolate_mousse","churros","clam_chowder","club_sandwich",
               "crab_cakes","creme_brulee","croque_madame","cup_cakes","deviled_eggs","donuts","dumplings","edamame","eggs_benedict","escargots",
               "falafel","filet_mignon","fish_and_chips","foie_gras","french_fries","french_onion_soup","french_toast","fried_calamari"
               ,"fried_rice","frozen_yogurt","garlic_bread","gnocchi","greek_salad","grilled_cheese_sandwich","grilled_salmon","guacamole"
               ,"gyoza","hamburger","hot_and_sour_soup","hot_dog","huevos_rancheros","hummus","ice_cream","lasagna","lobster_bisque","lobster_roll_sandwich"
               ,"macaroni_and_cheese","macarons","miso_soup","mussels","nachos","omelette","onion_rings","oysters","pad_thai","paella"
               ,"pancakes","panna_cotta","peking_duck","pho","pizza","pork_chop","poutine","prime_rib","pulled_pork_sandwich","ramen"
               ,"ravioli","red_velvet_cake","risotto","samosa","sashimi","scallops","seaweed_salad","shrimp_and_grits","spaghetti_bolognese"
               ,"spaghetti_carbonara","spring_rolls","steak","strawberry_shortcake","sushi","tacos","takoyaki","tiramisu","tuna_tartare","waffles"]
class_names = ["chicken_wings","cheesecake","apple_pie","chocolate_cake","chicken_curry"]
'''

def get_class_names_from_directory(directory):
    return sorted([d.name for d in os.scandir(directory) if d.is_dir()])


class_names = get_class_names_from_directory('dataset/train')
 # Replace with your actual class names
predicted_class_name = class_names[predicted_class_index]
print(f"Predicted class name: {predicted_class_name}")




# Display the image
img = keras.preprocessing.image.load_img(img_path, target_size=IMG_SIZE)
plt.imshow(img)
plt.title(f"Predicted class: {predicted_class_name}")
plt.axis('off')
plt.show()
