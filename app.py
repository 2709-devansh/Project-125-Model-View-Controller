from flask import Flask, jsonify, request
from alphabetRecog import get_prediction

app = Flask(__name__)
@app.route('/prediction', methods = ['POST'])
def takePrediction():
    image = request.files.get('Alphabet')
    prediction = get_prediction(image)
    return jsonify({
        "Prediction":prediction
    }), 402

if __name__ == '__main__':
    app.run()