# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import tensorflow as tf
import os
import glob
import cv2
import numpy as np
from matplotlib import pyplot as plt
import keras
import segmentation_models as sm

os.environ['KMP_DUPLICATE_LIB_OK']='True'

def dice_loss_plus_1focal_loss(y_true, y_pred):
    # Your custom loss implementation here
    return loss_value

 

def iou_score(y_true, y_pred):
    # Your custom metric implementation here
    return metric_value


def f1_score(y_true, y_pred):
    y_true = tf.keras.backend.flatten(y_true)
    y_pred = tf.keras.backend.flatten(tf.keras.backend.round(y_pred))

    # Convert tensors to numpy arrays for calculating the F1-score
    y_true_np = y_true.numpy()
    y_pred_np = y_pred.numpy()

 
    # Calculate the F1-score using sklearn's f1_score function
    f1 = f1_score(y_true_np, y_pred_np)

    return f1


model_filename = 'vgg_backbone_10_classes_50epochs 1.hdf5'
#model_filename = 'vgg_backbone_104_classes_50epochs.hdf5'

#this is path for model/
#C:\Tin N Nguyen\msu\FALL2023\CSC450\Project\WebProject\model
#model_path = os.path.join('Users', 'Tin N Nguyen','msu','FALL2023','CSC450','Project','WebProject','model', model_filename)
model_path = os.path.join('/Tin N Nguyen', 'msu', 'FALL2023', 'CSC450', 'Project', 'WebProject', 'be','model', model_filename)
#C:\Tin N Nguyen\msu\FALL2023\CSC450\Project\WebProject\be\uploads
 
# Register the custom metric function
with tf.keras.utils.custom_object_scope({'f1-score': f1_score},{'iou_score': iou_score},{'dice_loss_plus_1focal_loss': dice_loss_plus_1focal_loss}):

 
# Register the custom loss function

 
    # Load or define your model here
    model = tf.keras.models.load_model(model_path)  # Example of loading a model

 

SIZE_X = 128
SIZE_Y = 128


#this is path for image
images = []
directory = '/Tin N Nguyen/msu/FALL2023/CSC450/Project/WebProject/be/uploads'
file_pattern = '*.jpg'

image_paths = glob.glob(f'{directory}/{file_pattern}')
#image_path = os.path.join('/Users', 'ismaelal-hadhrami', 'csc450', 'FoodSeg103','Images', 'img_dir', 'practiceTest', '*.jpg')
sorted_image_path = sorted(image_paths)

 
for image_path in sorted_image_path:
    img = cv2.imread(image_path,cv2.IMREAD_COLOR)      
    img = cv2.resize(img, (SIZE_Y, SIZE_X))
    images.append(img)
images = np.array(images)


images.shape

plt.imshow(images[0])



BACKBONE = 'vgg16'
preprocess_input1 = sm.get_preprocessing(BACKBONE)

X_test = preprocess_input1(images)
plt.imshow(X_test[0])

pred = model.predict(X_test[:1])


pred_argmax = np.argmax(pred[0], axis=2)

plt.imshow(X_test[0])
plt.imshow(pred_argmax)

############################
food_dictionary = {
   0: ("background",0),
   1: ("Rice", 206),
   2: ("Green Beans", 31),
   3:("French Fries", 365),
   4: ("Carrot", 41),
   5: ("Shrimp", 99),
   6: ("Steak", 271),
   7: ("Onion",40),
   8: ("Tomato", 18),
   9: ("Egg", 155)
  }


# =============================================================================
# food_dictionary ={
#     0: ("background", 0),
#     1: ("candy", 200),  # Estimated for a small serving
#     2: ("egg tart", 300),  # Estimated for one tart
#     3: ("french fries", 365),  # Estimated for a medium serving
#     4: ("chocolate", 230),  # Estimated for a small ba
#     5: ("biscuit", 150),  # Estimated for a serving o 3 biscuits
#     6: ("popcorn", 100),  # Estimated for a small bag
#     7: ("pudding", 150),  # Estimated for one cup
#     8: ("ice cream", 200),  # Estimated for a scoop
#     9: ("cheese butter", 100),  # Estimated for a small serving
#     10: ("cake", 350),  # Estimated for a slice
#     11: ("wine", 125),  # Estimated for a glass (5 oz)
#     12: ("milkshake", 350),  # Estimated for a medium cup
#     13: ("coffee", 5),  # Estimated for black coffee, no sugar
#     14: ("juice", 110),  # Estimated for a glass (8 oz)
#     15: ("milk", 100),  # Estimated for a glass (8 oz)
#     16: ("tea", 2),  # Estimated for unsweetened tea
#     17: ("almond", 160),  # Estimated for an ounce (about 23 almonds)
#     18: ("red beans", 215),  # Estimated for a cup, cooked
#     19: ("cashew", 157),  # Estimated for an ounce
#     20: ("dried cranberries", 130),  # Estimated for a 1/4 cup serving
#     21: ("soy", 180),  # Estimated for a cup, cooked
#     22: ("walnut", 185),  # Estimated for an ounce
#     23: ("peanut", 166),  # Estimated for an ounce
#     24: ("egg", 155),  # Estimated for one large egg
#     25: ("apple", 95),  # Estimated for one medium apple
#     26: ("date", 66),  # Estimated for one date
#     27: ("apricot", 17),  # Estimated for one apricot
#     28: ("avocado", 234),  # Estimated for one medium avocado
#     29: ("banana", 105),  # Estimated for one medium banana
#     30: ("strawberry", 50),  # Estimated for a cup
#     31: ("cherry", 77),  # Estimated for a cup
#     32: ("blueberry", 85),  # Estimated for a cup
#     33: ("raspberry", 65),  # Estimated for a cup
#     34: ("mango", 202),  # Estimated for one mango
#     35: ("olives", 40),  # Estimated for 10 large olives
#     36: ("peach", 59),  # Estimated for one medium peach
#     37: ("lemon", 17),  # Estimated for one lemon
#     38: ("pear", 102),  # Estimated for one medium pear
#     39: ("fig", 47),  # Estimated for one medium fig
#     40: ("pineapple", 82),  # Estimated for a cup, chunks
#     41: ("grape", 62),  # Estimated for a cup
#     42: ("kiwi", 42),  # Estimated for one medium kiwi
#     43: ("melon", 64),  # Estimated for a cup, diced
#     44: ("orange", 62),  # Estimated for one medium orange
#     45: ("watermelon", 86),  # Estimated for a cup, diced
#     46: ("steak", 679),  # Estimated for a 6 oz serving
#     47: ("pork", 206),  # Estimated for a 3 oz serving
#     48: ("chicken duck", 237),  # Estimated for a 3 oz serving of chicken
#     49: ("sausage", 258),  # Estimated for a 3 oz serving
#     50: ("fried meat", 300),  # Estimated for a 3 oz serving
#     51: ("lamb", 250),  # Estimated for a 3 oz serving
#     52: ("sauce", 150),  # Estimated for a 1/2 cup serving
#     53: ("crab", 97),  # Estimated for a 3 oz serving
#     54: ("fish", 124),  # Estimated for a 3 oz serving
#     55: ("shellfish", 100),  # Estimated for a 3 oz serving
#     56: ("shrimp", 84),  # Estimated for a 3 oz serving
#     57: ("soup", 100),  # Estimated for a cup
#     58: ("bread", 79),  # Estimated for one slice
#     59: ("corn", 177),  # Estimated for a cup
#     60: ("hamburg", 354),  # Estimated for one medium hamburger
#     61: ("pizza", 285),  # Estimated for one slice
#     62: ("hanamaki baozi", 250),  # Estimated for one serving
#     63: ("wonton dumplings", 220),  # Estimated for a serving of 4
#     64: ("pasta", 221),  # Estimated for a cup, cooked
#     65: ("noodles", 221),  # Estimated for a cup, cooked
#     66: ("rice", 206),  # Estimated for a cup, cooke
#     67: ("pie", 300),  # Estimated for a slice
#     68: ("tofu", 76),  # Estimated for a 3 oz serving
#     69: ("eggplant", 35),  # Estimated for a cup, cooked
#     70: ("potato", 163),  # Estimated for one medium
#     71: ("garlic", 4),  # Estimated per clove
#     72: ("cauliflower", 25),  # Estimated for a cup, chopped
#     73: ("tomato", 22),  # Estimated for one medium
#     74: ("kelp", 43),  # Estimated for a cup
#     75: ("seaweed", 30),  # Estimated for a cup
#     76: ("spring onion", 32),  # Estimated for a cup, chopped
#     77: ("rape", 30),  # Estimated for a cup, cooked
#     78: ("ginger", 19),  # Estimated for an ounce
#     79: ("okra", 33),  # Estimated for a cup, sliced
#     80: ("lettuce", 5),  # Estimated for a cup, shredded
#     81: ("pumpkin", 30),  # Estimated for a cup, mashed
#     82: ("cucumber", 16),  # Estimated for a cup, sliced
#     83: ("white radish", 18),  # Estimated for a cup, sliced
#     84: ("carrot", 52),  # Estimated for a cup, chopped
#     85: ("asparagus", 27),  # Estimated for a cup
#     86: ("bamboo shoots", 13),  # Estimated for a cup, sliced
#     87: ("broccoli", 55),  # Estimated for a cup, chopped
#     88: ("celery stick", 6),  # Estimated per medium stak
#     89: ("cilantro mint", 2),  # Estimated for a tablespoon
#     90: ("snow peas", 67),  # Estimated for a cup
#     91: ("cabbage", 22),  # Estimated for a cup, chopped
#     92: ("bean sprouts", 31),  # Estimated for a cup
#     93: ("onion", 46),  # Estimated for one medium
#     94: ("pepper", 24),  # Estimated for one medium
#     95: ("green beans", 44),  # Estimated for a cup
#     96: ("French beans", 31),  # Estimated for a cup
#     97: ("king oyster mushroom", 35),  # Estimated for a cup, sliced
#     98: ("shiitake", 34),  # Estimated for a cup, cooked
#     99: ("enoki mushroom", 24),  # Estimated for a cup, cooked
#     100: ("oyster mushroom", 35),  # Estimated for a cup, cooked
#     101: ("white button mushroom", 21),  # Estimated for a cup, cooked
#     102: ("salad", 150),  # Estimated for a medium bowl
#     103: ("other ingredients", 50)  # Placeholder for generic ingredients
#     }
# =============================================================================

unique_pixels, counts = np.unique(pred_argmax, return_counts=True)
pixel_counts = dict(zip(unique_pixels, counts))

pixel_info_list = []

for pixel, count in pixel_counts.items():
    if count >=300 and pixel != 0:
        pixel_name, calories_per_serving = food_dictionary.get(pixel, "Unknown")
        servings = count / 2000
        total_cal = round(servings * calories_per_serving)
        pixel_info = f'{pixel_name}:{total_cal}'
        pixel_info_list.append(pixel_info)
    
print(pixel_info_list)
    
# =============================================================================
# total_calories = sum(int(info.split(': ')[1].split(' ')[0]) for info in pixel_info_list)
# 
# pixel_info_list.append(f'Total Calories: {total_calories}')
# =============================================================================
 
# =============================================================================
# print(f'Total Calories = {total_calories} calories')
# =============================================================================

# Writing the pixel_info_list to a text file
#C:\Tin N Nguyen\msu\FALL2023\CSC450\Project\WebProject\be
file_path = '/Tin N Nguyen/msu/FALL2023/CSC450/Project/WebProject/be/pixel_info_list.txt'

with open(file_path, 'w') as file:

    for item in pixel_info_list:

        file.write(item + '\n')

file_path