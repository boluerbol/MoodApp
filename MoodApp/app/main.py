import kivy
from kivy.app import App
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.camera import Camera
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image as PILImage

kivy.require('2.2.1')  # Ensure Kivy version is correct

class MoodApp(App):
    def build(self):
        self.model = self.load_model()
        self.layout = BoxLayout(orientation='vertical')

        # Create camera and capture button
        self.camera = Camera(play=True)
        self.layout.add_widget(self.camera)
        
        # Prediction label
        self.prediction_label = Label(text="Capture an image", font_size=20)
        self.layout.add_widget(self.prediction_label)

        # Button to take a picture
        self.capture_button = Button(text="Capture", size_hint=(None, None), size=(200, 50))
        self.capture_button.bind(on_press=self.capture_image)
        self.layout.add_widget(self.capture_button)

        return self.layout

    def load_model(self):
        # Load the TensorFlow Lite model
        interpreter = tf.lite.Interpreter(model_path="app/model/model.tflite")
        interpreter.allocate_tensors()
        return interpreter

    def capture_image(self, instance):
        # Capture image from the camera
        texture = self.camera.texture
        frame = np.frombuffer(texture.pixels, np.uint8).reshape(texture.height, texture.width, 4)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)

        # Preprocess image for the model
        image = cv2.resize(frame, (224, 224))  # Resize image to model's input size
        image = image / 255.0  # Normalize image to [0, 1]
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        
        # Run the model and get the prediction
        self.predict_mood(image)

    def predict_mood(self, image):
        # Set up the model input and output
        input_details = self.model.get_input_details()
        output_details = self.model.get_output_details()

        # Set input tensor
        self.model.set_tensor(input_details[0]['index'], image.astype(np.float32))
        
        # Run inference
        self.model.invoke()

        # Get the result
        output_data = self.model.get_tensor(output_details[0]['index'])
        predicted_class = np.argmax(output_data)

        # Map prediction to mood
        mood_classes = ['Happy', 'Sad', 'Surprised']
        predicted_mood = mood_classes[predicted_class]

        # Update the label with the result
        self.prediction_label.text = f"Predicted Mood: {predicted_mood}"

if __name__ == "__main__":
    MoodApp().run()
