import gradio as gr 
import keras 

# version 3.0.0
print(gr.__version__)

# Load CNN model
model = keras.saving.load_model("FinalModels/CNN.keras")


def recognize_digit(image):
    if image is not None: 
        image = image.reshape((1, 28, 28, 1)).astype('float32') / 255.0
        
        prediction = model.predict(image)
        
        return {str(i): float(prediction[0][i]) for i in range(10)}
    
    else:
        return ''
    
iface = gr.Interface(
    fn=recognize_digit,
    inputs=gr.Image(shape=(28, 28), image_mode='L', invert_colors=True, source='canvas'),
    outputs=gr.Label(num_top_classes=10)
)

iface.launch()

'''
ref: https://youtu.be/3DGLznJorT8
'''