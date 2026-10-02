"""
Flask Web Application for Sentiment Analysis of Media Posts.
Provides routes and REST APIs for Face Emotion Detection, Text Sentiment Analysis,
Multimodal Fusion, Prediction History, and Visual Analytics.
"""

import os
import io
import csv
import json
import joblib
from flask import Flask, render_template, request, jsonify, Response
from flask_cors import CORS

import database
from preprocessing.text_preprocessing import clean_text
from preprocessing.face_preprocessing import FaceEmotionClassifier

app = Flask(__name__)
CORS(app)

# Initialize Database
database.init_db()

# Load Models
TEXT_MODEL_PATH = "models/text_sentiment_model.pkl"
VECTORIZER_PATH = "models/tfidf_vectorizer.pkl"
text_model = None
tfidf_vectorizer = None

def load_text_models():
    global text_model, tfidf_vectorizer
    if os.path.exists(TEXT_MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        try:
            text_model = joblib.load(TEXT_MODEL_PATH)
            tfidf_vectorizer = joblib.load(VECTORIZER_PATH)
            print("[App] Loaded text sentiment model and vectorizer.")
        except Exception as e:
            print(f"[App] Error loading text model: {e}")

load_text_models()

# Initialize Face Classifier
face_classifier = FaceEmotionClassifier()

# Helper: Sentiment to Emoji
SENTIMENT_EMOJIS = {
    "Positive": "😊",
    "Negative": "😢",
    "Neutral": "😐"
}

# ----------------- PAGE ROUTES ----------------- #

@app.route('/')
def index():
    stats = database.get_statistics()
    return render_template('index.html', stats=stats)

@app.route('/face')
def face_page():
    return render_template('face.html')

@app.route('/text')
def text_page():
    return render_template('text.html')

@app.route('/multimodal')
def multimodal_page():
    return render_template('multimodal.html')

@app.route('/history')
def history_page():
    predictions = database.get_history(limit=100)
    return render_template('history.html', predictions=predictions)

@app.route('/analytics')
def analytics_page():
    stats = database.get_statistics()
    metrics = {}
    if os.path.exists("models/text_model_metrics.json"):
        with open("models/text_model_metrics.json") as f:
            metrics["text"] = json.load(f)
    if os.path.exists("models/face_model_metrics.json"):
        with open("models/face_model_metrics.json") as f:
            metrics["face"] = json.load(f)
    return render_template('analytics.html', stats=stats, metrics=metrics)

# ----------------- REST API ENDPOINTS ----------------- #

@app.route('/api/predict-text', methods=['POST'])
def predict_text():
    data = request.get_json() or {}
    raw_text = data.get('text', '').strip()
    
    if not raw_text:
        return jsonify({'error': 'Please provide text for analysis'}), 400
        
    global text_model, tfidf_vectorizer
    if text_model is None or tfidf_vectorizer is None:
        load_text_models()
        if text_model is None:
            # Fallback if model hasn't been trained yet
            return jsonify({'error': 'Text model not loaded. Please train model first.'}), 500
            
    cleaned = clean_text(raw_text)
    # If cleaning stripped everything (e.g. only symbols/numbers), fall back to raw
    text_to_vec = cleaned if cleaned else raw_text.lower()
    
    vec = tfidf_vectorizer.transform([text_to_vec])
    probs = text_model.predict_proba(vec)[0]
    classes = text_model.classes_
    
    prob_dict = {cls: round(float(prob) * 100.0, 1) for cls, prob in zip(classes, probs)}
    top_class = classes[int(probs.argmax())]
    confidence = prob_dict[top_class]
    
    # Save to database
    save_db = data.get('save_db', True)
    record_id = None
    if save_db:
        record_id = database.save_prediction(
            input_type='TEXT',
            input_content=raw_text[:200],
            emotion='-',
            sentiment=top_class,
            confidence=confidence
        )
        
    return jsonify({
        'status': 'success',
        'raw_text': raw_text,
        'cleaned_text': cleaned,
        'sentiment': top_class,
        'emoji': SENTIMENT_EMOJIS.get(top_class, '😐'),
        'confidence': confidence,
        'probabilities': prob_dict,
        'record_id': record_id
    })

@app.route('/api/predict-face', methods=['POST'])
def predict_face():
    data = request.get_json() or {}
    image_base64 = data.get('image', '')
    save_db = data.get('save_db', False)
    
    if not image_base64:
        return jsonify({'error': 'No image data received'}), 400
        
    img = face_classifier.decode_base64_image(image_base64)
    if img is None:
        return jsonify({'error': 'Invalid image format'}), 400
        
    result = face_classifier.detect_and_classify(img)
    
    record_id = None
    if save_db and result['face_detected']:
        record_id = database.save_prediction(
            input_type='FACE',
            input_content=f"Camera Snapshot ({result['faces_count']} face)",
            emotion=result['emotion'],
            sentiment=result['sentiment'],
            confidence=result['confidence']
        )
        
    result['status'] = 'success'
    result['record_id'] = record_id
    return jsonify(result)

@app.route('/api/predict-multimodal', methods=['POST'])
def predict_multimodal():
    """
    Multimodal fusion: combines text sentiment and facial emotion to yield unified sentiment.
    """
    data = request.get_json() or {}
    raw_text = data.get('text', '').strip()
    image_base64 = data.get('image', '')
    
    if not raw_text:
        return jsonify({'error': 'Text input is required'}), 400
        
    # 1. Analyze text
    global text_model, tfidf_vectorizer
    if text_model is None or tfidf_vectorizer is None:
        load_text_models()
        
    cleaned = clean_text(raw_text)
    text_to_vec = cleaned if cleaned else raw_text.lower()
    vec = tfidf_vectorizer.transform([text_to_vec])
    t_probs = text_model.predict_proba(vec)[0]
    t_classes = list(text_model.classes_)
    t_top_idx = int(t_probs.argmax())
    t_sentiment = t_classes[t_top_idx]
    t_conf = float(t_probs[t_top_idx]) * 100.0
    
    # 2. Analyze face if image provided
    face_result = None
    if image_base64:
        img = face_classifier.decode_base64_image(image_base64)
        if img is not None:
            face_result = face_classifier.detect_and_classify(img)
            
    # 3. Multimodal Decision Fusion
    # Weight: 60% Text, 40% Face Expression (or 100% Text if no face detected)
    if face_result and face_result['face_detected']:
        f_sentiment = face_result['sentiment']
        f_emotion = face_result['emotion']
        f_conf = face_result['confidence']
        
        # Calculate composite score (-1.0 to 1.0)
        score_map = {"Positive": 1.0, "Neutral": 0.0, "Negative": -1.0}
        t_score = score_map.get(t_sentiment, 0.0) * (t_conf / 100.0)
        f_score = score_map.get(f_sentiment, 0.0) * (f_conf / 100.0)
        
        composite = (0.6 * t_score) + (0.4 * f_score)
        
        if composite > 0.15:
            final_sentiment = "Positive"
        elif composite < -0.15:
            final_sentiment = "Negative"
        else:
            final_sentiment = "Neutral"
            
        combined_conf = round(min(99.0, (0.6 * t_conf) + (0.4 * f_conf)), 1)
        fusion_summary = f"Fused from Text ({t_sentiment}, {t_conf:.0f}%) & Face ({f_emotion}, {f_conf:.0f}%)"
    else:
        final_sentiment = t_sentiment
        f_emotion = "-"
        combined_conf = round(t_conf, 1)
        fusion_summary = f"Text Analysis only ({t_sentiment}, {t_conf:.0f}%) - No face detected"

    record_id = database.save_prediction(
        input_type='MULTIMODAL',
        input_content=raw_text[:200],
        emotion=f_emotion,
        sentiment=final_sentiment,
        confidence=combined_conf
    )
    
    return jsonify({
        'status': 'success',
        'final_sentiment': final_sentiment,
        'final_emoji': SENTIMENT_EMOJIS.get(final_sentiment, '😐'),
        'confidence': combined_conf,
        'fusion_summary': fusion_summary,
        'text_analysis': {
            'sentiment': t_sentiment,
            'confidence': round(t_conf, 1),
            'cleaned': cleaned
        },
        'face_analysis': face_result if face_result else {'face_detected': False},
        'record_id': record_id
    })

@app.route('/api/stats')
def api_stats():
    return jsonify(database.get_statistics())

@app.route('/api/history', methods=['GET'])
def api_history():
    filter_type = request.args.get('type')
    filter_sentiment = request.args.get('sentiment')
    search = request.args.get('search')
    limit = int(request.args.get('limit', 100))
    rows = database.get_history(limit=limit, filter_type=filter_type, filter_sentiment=filter_sentiment, search=search)
    return jsonify(rows)

@app.route('/api/history/<int:pred_id>', methods=['DELETE'])
def delete_history_item(pred_id):
    success = database.delete_prediction(pred_id)
    return jsonify({'status': 'success' if success else 'not_found'})

@app.route('/api/history/clear', methods=['DELETE'])
def clear_all_history():
    database.clear_history()
    return jsonify({'status': 'success', 'message': 'History cleared'})

@app.route('/api/export-csv')
def export_csv():
    rows = database.get_history(limit=1000)
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(['ID', 'Type', 'Input Content', 'Emotion', 'Sentiment', 'Confidence (%)', 'Timestamp'])
    for r in rows:
        writer.writerow([r['id'], r['input_type'], r['input_content'], r['emotion'], r['sentiment'], r['confidence'], r['created_at']])
        
    return Response(
        output.getvalue(),
        mimetype="text/csv",
        headers={"Content-disposition": "attachment; filename=sentiment_predictions_history.csv"}
    )

if __name__ == '__main__':
    print("\n" + "="*60)
    print(">>> SENTIMENT ANALYSIS APPLICATION RUNNING")
    print("--> Dashboard: http://127.0.0.1:5000")
    print("--> Face Emotion: http://127.0.0.1:5000/face")
    print("--> Text Sentiment: http://127.0.0.1:5000/text")
    print("--> Multimodal: http://127.0.0.1:5000/multimodal")
    print("="*60 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=False)
