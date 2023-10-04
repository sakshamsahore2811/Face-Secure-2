from flask import Flask, request, render_template, jsonify
from deepface import DeepFace
import os
from flask_sqlalchemy import SQLAlchemy
import base64

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///app.db'  # SQLite database
db = SQLAlchemy(app)
app.app_context().push()

class CapturedImage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    image_data = db.Column(db.LargeBinary, nullable=False)
    name = db.Column(db.String(255)) 

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        image_data_url = request.files['file']
        name = request.form['name']

        # Ensure the uploaded file is an image
        if image_data_url and allowed_file(image_data_url.filename):
            # Read and store the image
            image_data = image_data_url.read()

            # Save the captured image data and name in the database
            captured_image = CapturedImage(image_data=image_data, name=name)
            db.session.add(captured_image)
            db.session.commit()

            return jsonify({"message": f"Face registered successfully with ID {captured_image.id}"})
        else:
            return jsonify({"error": "Invalid file format. Please upload a valid image."})

    return render_template('register.html')

@app.route('/recognise', methods=['GET', 'POST'])
def recognise():
    if request.method == 'POST':
        image_data_url = request.files['image']

        # Ensure the uploaded file is an image
        if image_data_url and allowed_file(image_data_url.filename):
            # Read the binary image data from the form submission
            query_image_binary = image_data_url.read()

            # Fetch all stored images from the database
            stored_images = CapturedImage.query.all()

            matching_images = []

            for stored_image in stored_images:
                # Convert the stored image's binary data to Base64
                stored_image_base64 = base64.b64encode(stored_image.image_data).decode('utf-8')

                # Convert the query image binary data to Base64
                query_image_base64 = base64.b64encode(query_image_binary).decode('utf-8')

                # Perform face recognition using DeepFace
                result = DeepFace.verify(query_image_base64, stored_image_base64)

                if result['verified']:
                    matching_images.append({"id": stored_image.id, "name": stored_image.name})

            # Pass the matching_images list to the recognised.html template
            return render_template("recognised.html", matches=matching_images)

        else:
            return jsonify({"error": "Invalid file format. Please upload a valid image."})

    return render_template("recognise.html")

@app.route('/registered')
def registered():
    return render_template("registered.html")

@app.route('/recognised')
def recognised():
    return render_template("recognised.html")

if __name__ == "__main__":
    app.run(debug=True)
