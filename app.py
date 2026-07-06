from flask import Flask, request, jsonify, render_template
import os
from model2 import load_model, predict_image

app = Flask(__name__)

# Load model globally so it's loaded only once when the server starts
print("Loading model...")
model = load_model("models/corn_disease/densenet121_corn_20260628_0817_acc98-2_full_checkpoint.pt")
if model is not None:
    print("Model loaded successfully.")
else:
    print("Failed to load model. Check if densenet121_corn_20260628_0817_acc98-2_full_checkpoint.pt is present.")

@app.route('/')
def index():
    """
    Render the main web interface.
    """
    return render_template('index.html')

@app.route('/info')
def info():
    """
    Render the corn leaf encyclopedia/information page.
    """
    return render_template('info.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    API Endpoint to handle image upload and return prediction.
    """
    if 'file' not in request.files:
        return jsonify({"success": False, "error": "No file part in the request"}), 400
        
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"success": False, "error": "No selected file"}), 400
        
    if file:
        try:
            image_bytes = file.read()
            # Perform prediction using model.py
            result = predict_image(image_bytes, model)
            
            # Additional logic to set warning levels based on disease
            if result.get("success"):
                class_idx = result["class_index"]
                
                if class_idx == 0: # Bercak_Daun
                    result["status_color"] = "warning"
                    result["recommendation"] = "<ul><li>Beri penanganan dengan fungisida protektif kombinasi (seperti Difenokonazol + Azoksistrobin) secara berkala, terutama jika kondisi cuaca sedang hangat, lembap, dan sering turun hujan rintik-rintik (kondisi optimal penyebaran jamur Cercospora).</li></ul>"
                elif class_idx == 1: # Hawar_Daun (User referred to this as 'karat bercak')
                    result["status_color"] = "warning"
                    result["recommendation"] = "<ul><li>potong dan musnahkan daun yang terinfeksi agar spora tidak menyebar.</li><li>Lakukan penyemprotan fungisida sistemik berbahan aktif Azoksistrobin, Tebuconazole, atau Mankozeb sesuai dosis anjuran ketika gejala awal (bercak kecil seperti terendam air) mulai muncul di daun bagian bawah.</li></ul>"
                elif class_idx == 2: # Karat_Daun
                    result["status_color"] = "warning"
                    result["recommendation"] = "<ul><li>Perlebar jarak tanam untuk menjaga sirkulasi di area kanopi. Udara yang mengalir lancar akan menurunkan kelembapan mikro di sekitar daun, sehingga spora karat sulit berkecambah.</li><li>Penyemprotan fungisida berbahan aktif golongan triazol atau strobilurin seperti Propikonazol atau Pyraclostrobin. Penyemprotan difokuskan pada area bawah daun tempat pustul karat sering berkembang biak.</li></ul>"
                elif class_idx == 3: # Sehat
                    result["status_color"] = "success"
                    result["recommendation"] = "Daun sehat! Teruskan perawatan tanaman Anda dengan baik."
                    
            return jsonify(result)
        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)
