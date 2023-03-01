from django.shortcuts import render
from django.http import HttpResponse
import tensorflow as tf
import numpy as np
import urllib.request
import cv2
from django.core.files.storage import FileSystemStorage
from django.conf import settings
import os

# # Load the trained model
# model = tf.keras.models.load_model('path/to/your/trained/model')

# # Define a function to preprocess the input image
# def preprocess_image(image_url):
#     # Load the image from the URL
#     with urllib.request.urlopen(image_url) as url:
#         image = np.array(bytearray(url.read()), dtype=np.uint8)
#         image = cv2.imdecode(image, cv2.IMREAD_COLOR)
#     # Resize the image to the input size of the model
#     image = cv2.resize(image, (224, 224))
#     # Convert the image to a NumPy array
#     image = np.array(image, dtype=np.float32)
#     # Normalize the image pixels to have zero mean and unit variance
#     image = (image / 255.0 - 0.5) * 2.0
#     # Add an extra dimension to the array to represent the batch size of 1
#     image = np.expand_dims(image, axis=0)
#     return image

# # Define a view function to handle requests
# def index(request):
#     # Get the URL of the input image from the request parameters
#     image_url = request.GET.get('image_url')
#     # Preprocess the input image
#     image = preprocess_image(image_url)
#     # Use the model to predict the type of clothing in the image
#     predictions = model.predict(image)
#     # Get the index of the highest prediction value
#     predicted_class = np.argmax(predictions)
#     # Define a dictionary to map the class index to a human-readable label
#     class_labels = {0: 'T-shirt/top', 1: 'Trouser', 2: 'Pullover', 3: 'Dress', 4: 'Coat', 5: 'Sandal', 6: 'Shirt', 7: 'Sneaker', 8: 'Bag', 9: 'Ankle boot'}
#     # Get the label for the predicted class
#     predicted_label = class_labels[predicted_class]
#     # Render a response with the predicted label
#     return HttpResponse('The image contains a ' + predicted_label)


#     import tensorflow as tf
# from django.shortcuts import render
# from django.core.files.storage import FileSystemStorage
# from django.conf import settings
# import os

# Load ResNet50 model
model = tf.keras.applications.ResNet50(
    include_top=True, weights='imagenet', input_shape=None, pooling=None, classes=1000
)

# Define a function to identify images using ResNet50 model
def index(request):
    if request.method == 'POST' and request.FILES['image']:
        # Get the uploaded image file
        image_file = request.FILES['image']
        
        # Save the image to the media folder
        fs = FileSystemStorage()
        filename = fs.save(image_file.name, image_file)
        image_path = os.path.join(settings.MEDIA_ROOT, filename)

        # Load the image using TensorFlow
        image = tf.keras.preprocessing.image.load_img(image_path, target_size=(224, 224))

        # Preprocess the image for ResNet50 input
        image_array = tf.keras.preprocessing.image.img_to_array(image)
        image_array = tf.keras.applications.resnet50.preprocess_input(image_array)
        image_array = tf.expand_dims(image_array, 0)  # Create batch dimension

        # Use ResNet50 model to predict the image label
        predictions = model.predict(image_array)
        predicted_class = tf.keras.applications.resnet50.decode_predictions(predictions, top=1)[0][0][1]

        # Render the result template with the predicted class
        return render(request, 'result.html', {'predicted_class': predicted_class})

    # Render the form template for image upload
    return render(request, 'form.html')
