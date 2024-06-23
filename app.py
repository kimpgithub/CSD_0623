from flask import Flask, request, jsonify, send_file
import os
import cv2
from hand_detect import detect_hand_pose
from bat_seg import BatSegmentation

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

bat_segmentation = BatSegmentation('yolov8x-seg.pt')

@app.route('/')
def index():
    return '''
    <!doctype html>
    <title>Upload Image</title>
    <h1>Upload an image to detect hand pose and segment baseball bat</h1>
    <form method=post enctype=multipart/form-data action="/upload">
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)
    print(f"File saved to {filepath}")

    hand_processed_image = detect_hand_pose(filepath)
    
    if hand_processed_image is None:
        return jsonify({'error': 'Hand pose detection failed'})
    
    final_processed_image = bat_segmentation.segment_bat(hand_processed_image)
    
    if final_processed_image is None:
        return jsonify({'error': 'Bat segmentation failed'})

    final_processed_image_path = os.path.join(PROCESSED_FOLDER, 'final_' + file.filename)
    cv2.imwrite(final_processed_image_path, final_processed_image)
    
    print(f"Processed image saved to {final_processed_image_path}")
    return send_file(final_processed_image_path, mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)
