from flask import Flask, request, render_template, send_file
import style_transfer  # Replace with the name of your style transfer function
from PIL import Image
import numpy as np
from io import BytesIO

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def upload_page():
    result_image = None

    if request.method == 'POST':
        content_image = request.files['content_image']
        style_image = request.files['style_image']
        
        # Retrieve the value of the "iterations" input
        iterations = int(request.form['iterations'])

        if content_image.filename and style_image.filename:
            content_data = content_image.read()
            style_data = style_image.read()

            result_image = style_transfer.main(content_data, style_data, iterations)
            image_data_0_255 = (result_image * 255).astype(np.uint8)
            pillow_image = Image.fromarray(image_data_0_255, 'RGB')
            # Save the image in memory and serve it directly
            image_io = BytesIO()
            pillow_image.save(image_io, 'JPEG')
            image_io.seek(0)
            return send_file(image_io, mimetype='image/jpeg')

    return render_template('index.html')

if __name__ == '__main__':
    app.run()
