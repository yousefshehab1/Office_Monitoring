from flask import Flask, request, render_template, send_from_directory
import os
import cv2
from main import main

# Initialize Flask app
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output_video'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')  # Simple HTML page with upload form

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'video' not in request.files:
        return "No video file provided", 400

    video = request.files['video']
    if video.filename == '':
        return "No selected file", 400

    # Save the uploaded file
    video_path = os.path.join(UPLOAD_FOLDER, video.filename)
    video.save(video_path)

    # Process the video
    output_path = os.path.join(OUTPUT_FOLDER, f"processed_{video.filename}")
    main(source_video=video_path)  # Modify 'main.py' to allow dynamic output paths

    return f"<h3>Processing complete! <a href='/download/{os.path.basename(output_path)}'>Download Video</a></h3>"

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(OUTPUT_FOLDER, filename, as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)
