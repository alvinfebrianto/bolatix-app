import json
import os
import uuid
import secrets
from datetime import datetime, timedelta
from functools import wraps

import bcrypt
import jwt
import numpy as np
import pandas as pd
import tensorflow as tf
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_mail import Mail, Message
from google.cloud import storage
from google.oauth2 import service_account
from werkzeug.utils import secure_filename
import requests

import firebase_admin
from firebase_admin import credentials, firestore

# Initialize Flask app
app = Flask(__name__)
load_dotenv()

# Load environment variables and initialize app
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY') or secrets.token_hex(32)
if not os.getenv('SECRET_KEY'):
    with open('.env', 'a') as f:
        f.write(f"\nSECRET_KEY={app.config['SECRET_KEY']}")

# Mail configuration
app.config['MAIL_SERVER'] = os.getenv('MAIL_SERVER')
app.config['MAIL_PORT'] = int(os.getenv('MAIL_PORT', 587))
app.config['MAIL_USE_TLS'] = os.getenv('MAIL_USE_TLS', 'True').lower() == 'true'
app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME')
app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD')

mail = Mail(app)

# Configure Firebase
try:
    if os.path.exists('serviceAccountKey.json'):
        cred = credentials.Certificate('serviceAccountKey.json')
    else:
        cred = credentials.ApplicationDefault()
    
    firebase_admin.initialize_app(cred)
    db = firestore.client()

    # Google Cloud Storage initialization
    if os.path.exists('serviceAccountKey.json'):
        cred_dict = json.load(open('serviceAccountKey.json'))
        storage_credentials = service_account.Credentials.from_service_account_info(cred_dict)
    else:
        storage_credentials = None

    storage_client = storage.Client(
        project='bolatix',
        credentials=storage_credentials
    )
    BUCKET_NAME = 'bolatix-user-profiles'
    global bucket
    bucket = storage_client.bucket(BUCKET_NAME)
    print(f"Successfully initialized bucket: {BUCKET_NAME}")
except Exception as e:
    print(f"Firebase/Storage initialization error: {e}")

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def upload_profile_picture(file, user_id):
    try:
        if not file or not allowed_file(file.filename):
            print(f"File validation failed: {file.filename if file else 'No file'}")
            return None
        
        # Delete existing profile pictures for this user
        blobs = bucket.list_blobs(prefix=f"profile_pictures/{user_id}/")
        for blob in blobs:
            print(f"Deleting existing profile picture: {blob.name}")
            blob.delete()
        
        # Create a unique filename
        original_extension = file.filename.rsplit('.', 1)[1].lower()
        filename = f"profile_pictures/{user_id}/{str(uuid.uuid4())}.{original_extension}"
        print(f"Attempting to upload to: {filename}")
        
        # Upload to Google Cloud Storage
        blob = bucket.blob(filename)
        print(f"Created blob: {blob.name}")
        
        file_content = file.read()
        print(f"Read file content, size: {len(file_content)} bytes")
        
        blob.upload_from_string(
            file_content,
            content_type=file.content_type
        )
        print("Upload completed")
        
        blob.make_public()
        
        public_url = blob.public_url
        print(f"Generated public URL: {public_url}")
        
        return public_url
    except Exception as e:
        print(f"Upload error: {str(e)}")
        raise

# Define bucket and file paths
HISTORY_MODEL_BLOB_PATH = "models/history.h5"
COLDSTART_MODEL_BLOB_PATH = "models/cold_start.h5"
DATASET_BLOB_PATH = "data/dataset.csv"

# Local temporary paths for downloaded files
HISTORY_MODEL_PATH = "/tmp/history.h5"
COLDSTART_MODEL_PATH = "/tmp/cold_start.h5"
DATASET_PATH = "/tmp/dataset.csv"

def download_from_gcs(blob_path, local_path):
    """Downloads a file from GCS to a local path."""
    try:
        blob = bucket.blob(blob_path)
        blob.download_to_filename(local_path)
        print(f"Downloaded {blob_path} to {local_path}")
    except Exception as e:
        print(f"Error downloading {blob_path}: {e}")
        raise

# Download models and dataset
try:
    download_from_gcs(HISTORY_MODEL_BLOB_PATH, HISTORY_MODEL_PATH)
    download_from_gcs(COLDSTART_MODEL_BLOB_PATH, COLDSTART_MODEL_PATH)
    download_from_gcs(DATASET_BLOB_PATH, DATASET_PATH)
    USE_DUMMY = False
except Exception as e:
    print(f"Failed to download necessary files from GCS: {e}")
    USE_DUMMY = True

# Load dataset globally
try:
    dataset = pd.read_csv(DATASET_PATH)
    dataset['Score tim home'] = dataset['Score tim home'].fillna(0).astype(int)
    dataset['Score tim away'] = dataset['Score tim away'].fillna(0).astype(int)
except Exception as e:
    print(f"Error loading dataset: {e}")
    dataset = pd.DataFrame()

# Load models
if not USE_DUMMY:
    try:
        model_history = tf.keras.models.load_model(HISTORY_MODEL_PATH)
        model_coldstart = tf.keras.models.load_model(COLDSTART_MODEL_PATH)
    except Exception as e:
        print(f"Error loading models: {e}")
        USE_DUMMY = True

def get_user_data(user_id):
    doc = db.collection('users').document(user_id).get()
    return doc.to_dict() if doc.exists else None

def generate_token(user_id):
    try:
        payload = {
            'iat': datetime.utcnow(),
            'sub': user_id,
            'type': 'persistent'
        }
        return jwt.encode(payload, app.config['SECRET_KEY'], algorithm='HS256')
    except Exception:
        return None

def verify_token(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_header = request.headers.get('Authorization', '')
        token = auth_header.split(" ")[1] if len(auth_header.split()) > 1 else None
        
        if not token:
            return jsonify({'status': False, 'message': 'Token is missing'}), 401
            
        try:
            payload = jwt.decode(token, app.config['SECRET_KEY'], 
                               algorithms=['HS256'], options={"verify_exp": False})
            request.user_id = payload['sub']
            
            user_data = get_user_data(payload['sub'])
            if user_data and user_data.get('token_invalidated_at'):
                return jsonify({'status': False, 'message': 'Token has been invalidated'}), 401
                
        except jwt.InvalidTokenError:
            return jsonify({'status': False, 'message': 'Invalid token'}), 401
            
        return f(*args, **kwargs)
    return decorated

def format_alldata(match):
    return {
        "id_match": match['ID Match'],
        "match": match['Match'],
        "home_score": int(match['Score tim home']) if not pd.isna(match['Score tim home']) else 0,
        "away_score": int(match['Score tim away']) if not pd.isna(match['Score tim away']) else 0,
        "home_team": match['Home'].strip(),
        "away_team": match['Away'].strip(),
        "lokasi": match['Lokasi'],
        "jam": match['Jam'].rsplit(':', 1)[0],
        "waktu": match['Waktu'],
        "stadion": match['Stadion'],
        "hari": match['Hari'],
        "tanggal": match['Tanggal'],
        "tiket_terjual": int(match['Jumlah Tiket Terjual']),
    }


def format_match_recommendation(match, action="Consider buying tickets"):
    return {
        "id_match": match['ID Match'],
        "match": match['Match'],
        "home_score": int(match['Score tim home']) if not pd.isna(match['Score tim home']) else 0,
        "away_score": int(match['Score tim away']) if not pd.isna(match['Score tim away']) else 0,
        "home_team": match['Home'].strip(),
        "away_team": match['Away'].strip(),
        "lokasi": match['Lokasi'],
        "jam": match['Jam'].rsplit(':', 1)[0],
        "waktu": match['Waktu'],
        "stadion": match['Stadion'],
        "hari": match['Hari'],
        "tanggal": match['Tanggal'],
        "tiket_terjual": int(match['Jumlah Tiket Terjual']),
        "suggested_action": action
    }

def get_recommendations_history(user_id):
    user_data = get_user_data(user_id)
    if not user_data or not user_data.get('purchase_history'):
        return []

    relevant_teams = {team.strip() for purchase in user_data['purchase_history']
                     for team in [purchase['home_team'], purchase['away_team']]}

    if USE_DUMMY:
        recommendations = [
            format_match_recommendation(match)
            for _, match in dataset.iterrows()
            if match['Home'].strip() in relevant_teams or match['Away'].strip() in relevant_teams
        ]
        return recommendations
    
    return process_predictions(model_history.predict(user_data))

def get_recommendations_new_user(favorite_team):
    if USE_DUMMY:
        recommendations = [
            format_match_recommendation(match, "New match for you!")
            for _, match in dataset.iterrows()
            if favorite_team in [match['Home'].strip(), match['Away'].strip()]
        ]
        return recommendations[:10]
    
    return process_predictions(model_coldstart.predict([favorite_team]))

def process_predictions(predictions):
    """Process ML model prediction results into readable recommendation format"""
    try:
        if dataset.empty:
            return []
            
        recommendations = []
        for idx, score in enumerate(predictions[0]):
            match = dataset.iloc[idx]
            recommendations.append({
                "id_match": str(match['ID Match']),
                "home_team": match['Home'].strip(),
                "away_team": match['Away'].strip(),
                "tanggal": match['Tanggal'].strip(),
                "jam": match['Jam'].rsplit(':', 1)[0],
                "stadion": match['Stadion'],
                "lokasi": match['Lokasi'],
                "tiket_terjual": int(match['Jumlah Tiket Terjual']),
                "score": float(score)
            })
        return sorted(recommendations, key=lambda x: x['score'], reverse=True)[:10]
    except Exception as e:
        print(f"Error processing predictions: {e}")
        return []

# API Endpoints
@app.route('/api/auth/register', methods=['POST'])
def register():
    try:
        data = request.json
        if not all(key in data for key in ['email', 'password']):
            return jsonify({
                'status': False,
                'message': 'Email and password are required'
            }), 400
            
        existing = db.collection('users').where('email', '==', data['email']).get()
        if len(list(existing)) > 0:
            return jsonify({
                'status': False,
                'message': 'Email already registered'
            }), 409
            
        user_data = {
            'email': data['email'],
            'password': bcrypt.hashpw(data['password'].encode('utf-8'), 
                                    bcrypt.gensalt()).decode('utf-8'),
            'name': data.get('name', ''),
            'favorite_team': data.get('favorite_team', ''),
            'birth_date': data.get('birth_date'),
            'profile_picture': data.get('profile_picture', ''),
            'purchase_history': []
        }
        
        new_user_ref = db.collection('users').document()
        
        user_data_with_timestamps = {
            **user_data,
            'created_at': firestore.SERVER_TIMESTAMP,
            'updated_at': firestore.SERVER_TIMESTAMP
        }
        new_user_ref.set(user_data_with_timestamps)
        
        token = generate_token(new_user_ref.id)
        
        response_data = {k: v for k, v in user_data.items() 
                        if k not in ['password']}
        
        return jsonify({
            'status': True,
            'message': 'User registered successfully',
            'data': {
                'token': token,
                'user_id': new_user_ref.id,
                'user': response_data
            }
        }), 201
        
    except Exception as e:
        return jsonify({
            'status': False,
            'message': str(e)
        }), 500

@app.route('/api/auth/login', methods=['POST'])
def login():
    try:
        data = request.json
        if not all(key in data for key in ['email', 'password']):
            return jsonify({
                'status': False,
                'message': 'Email and password are required'
            }), 400
            
        users = list(db.collection('users').where('email', '==', data['email']).get())
        if not users:
            return jsonify({
                'status': False,
                'message': 'Invalid credentials'
            }), 401
            
        user = users[0]
        user_data = user.to_dict()
        
        if not bcrypt.checkpw(data['password'].encode('utf-8'), 
                            user_data['password'].encode('utf-8')):
            return jsonify({
                'status': False,
                'message': 'Invalid credentials'
            }), 401
            
        user.reference.update({'token_invalidated_at': None})
        token = generate_token(user.id)
        
        return jsonify({
            'status': True,
            'message': 'Login successful',
            'data': {
                'token': token,
                'user_id': user.id
            }
        }), 200
        
    except Exception as e:
        return jsonify({
            'status': False,
            'message': str(e)
        }), 500

@app.route('/api/auth/logout', methods=['POST'])
@verify_token
def logout():
    try:
        user_ref = db.collection('users').document(request.user_id)
        user_ref.update({'token_invalidated_at': firestore.SERVER_TIMESTAMP})
        return jsonify({
            'status': True,
            'message': 'Logout successful'
        }), 200
    except Exception as e:
        return jsonify({
            'status': False,
            'message': str(e)
        }), 500

@app.route('/api/users/<user_id>', methods=['GET'])
def read_user(user_id):
    try:
        user_data = get_user_data(user_id)
        if not user_data:
            return jsonify({
                'status': False,
                'message': 'User not found'
            }), 404
            
        user_data.pop('password', None)
        return jsonify({
            'status': True,
            'message': 'User data retrieved successfully',
            'data': user_data
        }), 200
        
    except Exception as e:
        return jsonify({
            'status': False,
            'message': str(e)
        }), 500

@app.route('/api/users/<user_id>', methods=['PUT'])
def update_user(user_id):
    try:
        data = request.json
        if not data:
            return jsonify({
                'status': False,
                'message': 'No data provided for update'
            }), 400
        
        user_ref = db.collection('users').document(user_id)
        user_doc = user_ref.get()
        
        if not user_doc.exists:
            return jsonify({
                'status': False,
                'message': 'User not found'
            }), 404
        
        update_data = {}
        allowed_fields = ['name', 'favorite_team', 'birth_date', 'profile_picture']
        for field in allowed_fields:
            if field in data:
                update_data[field] = data[field]
                
        update_data_with_timestamp = {
            **update_data,
            'updated_at': firestore.SERVER_TIMESTAMP
        }
        
        user_ref.update(update_data_with_timestamp)
        
        return jsonify({
            'status': True,
            'message': 'User updated successfully',
            'data': update_data
        }), 200
        
    except Exception as e:
        return jsonify({
            'status': False,
            'message': str(e)
        }), 500

@app.route('/api/users/<user_id>', methods=['DELETE'])
def delete_user(user_id):
    try:
        user_ref = db.collection('users').document(user_id)
        if not user_ref.get().exists:
            return jsonify({
                'status': False,
                'message': 'User not found'
            }), 404
        
        user_ref.delete()
        return jsonify({
            'status': True,
            'message': 'User deleted successfully'
        }), 200
        
    except Exception as e:
        return jsonify({
            'status': False,
            'message': str(e)
        }), 500

@app.route('/api/users/<user_id>/purchases', methods=['POST'])
def add_purchase(user_id):
    try:
        data = request.json
        required_fields = [
            'match_id', 'home_team', 'away_team', 'stadium', 
            'match_date', 'purchase_date', 'ticket_quantity'
        ]
        if not all(field in data for field in required_fields):
                return jsonify({
                    'status': False,
                    'message': f'Required fields: {", ".join(required_fields)}'
                }), 400
        
        user_ref = db.collection('users').document(user_id)
        user_doc = user_ref.get()
        
        if not user_doc.exists:
            return jsonify({
                'status': False,
                'message': 'User not found'
            }), 404
        
        purchase = {
            'match_id': data['match_id'],
            'home_team': data['home_team'],
            'away_team': data['away_team'],
            'stadium': data['stadium'],
            'match_date': data['match_date'],
            'purchase_date': data['purchase_date'],
            'ticket_quantity': data['ticket_quantity']
        }
        
        user_ref.update({
            'purchase_history': firestore.ArrayUnion([purchase])
        })
        
        return jsonify({
            'status': True,
            'message': 'Purchase added to history successfully',
            'data': purchase
        }), 201
        
    except Exception as e:
        return jsonify({
            'status': False,
            'message': str(e)
        }), 500

@app.route('/api/users/<user_id>/purchases', methods=['GET'])
def get_purchase_history(user_id):
    try:
        user_ref = db.collection('users').document(user_id)
        user_doc = user_ref.get()
        
        if not user_doc.exists:
            return jsonify({
                'status': False,
                'message': 'User not found'
            }), 404
        
        user_data = user_doc.to_dict()
        purchase_history = user_data.get('purchase_history', [])
        
        return jsonify({
            'status': True,
            'message': 'Purchase history retrieved successfully',
            'data': purchase_history
        }), 200
        
    except Exception as e:
        return jsonify({
            'status': False,
            'message': str(e)
        }), 500

@app.route('/standings', methods=['GET'])
def get_standings():
    try:
        url = 'https://gist.githubusercontent.com/alhifnywahid/223b6d759c75c6e1be7e7c83fe4a3cf6/raw/bolatix-standings.json'
        response = requests.get(url)
        response.raise_for_status()
        
        standings_data = response.json()
        
        return jsonify({
            'status': True,
            'message': 'Standings retrieved successfully',
            'data': standings_data
        }), 200
    
    except requests.RequestException as e:
        return jsonify({
            'status': False,
            'message': f'Error fetching standings: {str(e)}'
        }), 500
    except ValueError as e:
        return jsonify({
            'status': False,
            'message': f'Error parsing JSON: {str(e)}'
        }), 500

@app.route('/api/recommend-teamfavorite', methods=['GET'])
def recommend_teamfavorite():
    try:
        user_id = request.args.get('user_id')
        if not user_id:
            return jsonify({
                'status': False,
                'message': 'User ID is required'
            }), 400
        
        # Fetch user data
        user_data = get_user_data(user_id)
        if not user_data:
            return jsonify({
                'status': False,
                'message': 'User not found'
            }), 404

        today_date = datetime.today().date()

        # Dummy data processing
        if USE_DUMMY:
            favorite_team = user_data.get('favorite_team', '')
            if not favorite_team:
                return jsonify({
                    'status': False,
                    'message': 'Favorite team is required for dummy recommendations'
                }), 400

            # Convert favorite_team to lowercase for case-insensitive matching
            favorite_team = favorite_team.lower()

            recommendations = []
            for _, match in dataset.iterrows():
                match_date = None
                try:
                    match_date = datetime.strptime(match['Tanggal'], '%d/%m/%Y').date()
                except ValueError:
                    try:
                        match_date = datetime.strptime(match['Tanggal'], '%d-%m-%Y').date()
                    except ValueError:
                        continue

                # Convert team names to lowercase for case-insensitive matching
                home_team = match['Home'].strip().lower()
                away_team = match['Away'].strip().lower()

                if favorite_team in [home_team, away_team] and match_date >= today_date:
                    recommendations.append(format_match_recommendation(match))

            recommendations = recommendations[:10]
        else:
            # Predict recommendations based on user data
            if user_data.get('purchase_history'):
                predictions = model_history.predict([user_id])
            else:
                favorite_team = user_data.get('favorite_team')
                if not favorite_team:
                    return jsonify({
                        'status': False,
                        'message': 'Favorite team is required for recommendations'
                    }), 400
                
                predictions = model_coldstart.predict([[favorite_team]])

            # Process predictions and filter by date
            recommendations = []
            for match in process_predictions(predictions):
                match_date = None
                try:
                    match_date = datetime.strptime(match['tanggal'], '%d/%m/%Y').date()
                except ValueError:
                    try:
                        match_date = datetime.strptime(match['tanggal'], '%d-%m-%Y').date()
                    except ValueError:
                        continue

                if match_date >= today_date:
                    recommendations.append(match)

        # Ensure exactly 10 recommendations or fewer if not enough matches
        recommendations = recommendations[:10]

        return jsonify({
            'status': True,
            'message': 'Recommendations retrieved successfully',
            'data': recommendations
        }), 200

    except Exception as e:
        # Log and return the error
        print(f"Recommendation error: {e}")
        return jsonify({
            'status': False,
            'message': 'An error occurred while retrieving recommendations',
            'error': str(e)
        }), 500

@app.route('/api/recommend-history', methods=['GET'])
def recommend_history():
    try:
        user_id = request.args.get('user_id')
        if not user_id:
            return jsonify({
                'status': False,
                'message': 'User ID is required'
            }), 400
        
        # Fetch user data
        user_data = get_user_data(user_id)
        if not user_data:
            return jsonify({
                'status': False,
                'message': 'User not found'
            }), 404

        if not user_data.get('purchase_history'):
            return jsonify({
                'status': False,
                'message': 'Purchase history is required for recommendations'
            }), 400

        today_date = datetime.today().date()

        # Extract relevant teams from purchase history
        relevant_teams = {team.strip().lower() for purchase in user_data['purchase_history']
                          for team in [purchase['home_team'], purchase['away_team']]}

        recommendations = []

        if USE_DUMMY:
            # Generate recommendations from dummy data
            for _, match in dataset.iterrows():
                match_date = None
                try:
                    match_date = datetime.strptime(match['Tanggal'], '%d/%m/%Y').date()
                except ValueError:
                    try:
                        match_date = datetime.strptime(match['Tanggal'], '%d-%m-%Y').date()
                    except ValueError:
                        continue

                # Convert team names to lowercase for case-insensitive matching
                home_team = match['Home'].strip().lower()
                away_team = match['Away'].strip().lower()

                if (home_team in relevant_teams or away_team in relevant_teams) and match_date >= today_date:
                    recommendations.append(format_match_recommendation(match))

        else:
            # Use the prediction model to generate recommendations
            predictions = model_history.predict([user_id])
            for match in process_predictions(predictions):
                match_date = None
                try:
                    match_date = datetime.strptime(match['tanggal'], '%d/%m/%Y').date()
                except ValueError:
                    try:
                        match_date = datetime.strptime(match['tanggal'], '%d-%m-%Y').date()
                    except ValueError:
                        continue

                if match_date >= today_date:
                    recommendations.append(match)

        # Limit to top 10 recommendations
        recommendations = recommendations[:10]

        return jsonify({
            'status': True,
            'message': 'Recommendations retrieved successfully',
            'data': recommendations
        }), 200

    except Exception as e:
        # Log and return the error
        print(f"Recommendation error: {e}")
        return jsonify({
            'status': False,
            'message': 'An error occurred while retrieving recommendations',
            'error': str(e)
        }), 500

@app.route('/api/alldata', methods=['GET'])
def alldata():
    try:
        # Pastikan dataset sudah dimuat
        if dataset.empty:
            return {
                "status": False,
                "message": "Dataset is empty or not loaded"
            }, 500

        # Format semua data dari dataset
        all_data = [format_alldata(row) for _, row in dataset.iterrows()]

        return {
            "status": True,
            "message": "All data retrieved successfully",
            "data": all_data
        }, 200

    except Exception as e:
        # Log dan kembalikan error dalam struktur JSON
        print(f"Error retrieving all data: {e}")
        return {
            "status": False,
            "message": "An error occurred while retrieving all data"
        }, 500
    
@app.route('/api/users/<user_id>/profile-picture', methods=['GET', 'POST', 'PUT', 'DELETE'])
def manage_profile_picture(user_id):
    # GET: Retrieve profile picture URL
    if request.method == 'GET':
        try:
            user_ref = db.collection('users').document(user_id)
            user_doc = user_ref.get()
            
            if not user_doc.exists:
                return jsonify({
                    'status': False, 
                    'message': 'User not found'
                }), 404
            
            profile_picture = user_doc.to_dict().get('profile_picture')
            return jsonify({
                'status': True,
                'data': {
                    'profile_picture_url': profile_picture or None
                }
            }), 200

        except Exception as e:
            return jsonify({
                'status': False,
                'message': str(e)
            }), 500

    # POST/PUT: Upload or Replace Profile Picture
    if request.method in ['POST', 'PUT']:
        try:
            if 'profile_picture' not in request.files:
                return jsonify({
                    'status': False,
                    'message': 'No file provided'
                }), 400
            
            file = request.files['profile_picture']
            
            # Validate file type
            if not allowed_file(file.filename):
                return jsonify({
                    'status': False,
                    'message': f'Invalid file type. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'
                }), 400

            # Delete existing profile picture
            user_ref = db.collection('users').document(user_id)
            user_doc = user_ref.get()
            
            if user_doc.exists:
                old_picture_url = user_doc.to_dict().get('profile_picture')
                if old_picture_url:
                    try:
                        # Extract the full blob path, not just the filename
                        blob_name = f"profile_pictures/{user_id}/{old_picture_url.split('/')[-1]}"
                        old_blob = bucket.blob(blob_name)
                        old_blob.delete()
                        print(f"Deleted blob: {blob_name}")
                    except Exception as storage_error:
                        print(f"Error deleting from storage: {storage_error}")

            # Upload new profile picture
            picture_url = upload_profile_picture(file, user_id)
            
            # Update Firestore
            user_ref.update({
                'profile_picture': picture_url,
                'updated_at': firestore.SERVER_TIMESTAMP
            })

            return jsonify({
                'status': True,
                'message': 'Profile picture updated successfully',
                'data': {
                    'profile_picture_url': picture_url
                }
            }), 200

        except Exception as e:
            return jsonify({
                'status': False,
                'message': str(e)
            }), 500

    # DELETE: Remove Profile Picture
    if request.method == 'DELETE':
        try:
            user_ref = db.collection('users').document(user_id)
            user_doc = user_ref.get()
            
            if not user_doc.exists:
                return jsonify({
                    'status': False, 
                    'message': 'User not found'
                }), 404
            
            user_data = user_doc.to_dict()
            old_picture_url = user_data.get('profile_picture')
            
            if old_picture_url:
                try:
                    blob_name = f"profile_pictures/{user_id}/{old_picture_url.split('/')[-1]}"
                    old_blob = bucket.blob(blob_name)
                    old_blob.delete()
                    print(f"Deleted blob: {blob_name}")
                except Exception as storage_error:
                    print(f"Error deleting from storage: {storage_error}")
                
                user_ref.update({
                    'profile_picture': '',
                    'updated_at': firestore.SERVER_TIMESTAMP
                })
            
            return jsonify({
                'status': True,
                'message': 'Profile picture removed successfully'
            }), 200

        except Exception as e:
            return jsonify({
                'status': False,
                'message': str(e)
            }), 500

def send_reset_email(user_email, reset_token):
    msg = Message('BolaTix Password Reset',
                  sender=app.config['MAIL_USERNAME'],
                  recipients=[user_email])
    msg.body = f'''Your password reset code for BolaTix app:

{reset_token}

Enter this code in the app to reset your password. This code will expire in 1 day.

If you did not request a password reset, please ignore this email and ensure your account is secure.
'''
    mail.send(msg)

@app.route('/forgot-password', methods=['POST'])
def forgot_password():
    try:
        data = request.get_json()
        email = data.get('email')
        
        if not email:
            return jsonify({
                'status': False,
                'message': 'Email is required'
            }), 400
            
        # Check if user exists
        users_ref = db.collection('users')
        query = users_ref.where('email', '==', email).limit(1).get()
        
        if not query:
            return jsonify({
                'status': True,
                'message': 'If an account exists with this email, a password reset link will be sent.'
            }), 200
            
        user = query[0]
        
        # Generate reset token
        reset_token = secrets.token_urlsafe(32)
        expiration = datetime.utcnow() + timedelta(days=1)
        
        # Store reset token in user document
        user.reference.update({
            'reset_token': reset_token,
            'reset_token_exp': expiration
        })
        
        # Send reset email
        send_reset_email(email, reset_token)
        
        return jsonify({
            'status': True,
            'message': 'If an account exists with this email, a password reset link will be sent.'
        }), 200
        
    except Exception as e:
        return jsonify({
            'status': False,
            'message': str(e)
        }), 500

@app.route('/reset-password', methods=['POST'])
def reset_password():
    try:
        data = request.get_json()
        token = data.get('token')
        new_password = data.get('new_password')
        
        if not token or not new_password:
            return jsonify({
                'status': False,
                'message': 'Token and new password are required'
            }), 400
            
        # Find user with this reset token
        users_ref = db.collection('users')
        query = users_ref.where('reset_token', '==', token).limit(1).get()
        
        if not query:
            return jsonify({
                'status': False,
                'message': 'Invalid or expired reset token'
            }), 400
            
        user = query[0]
        
        # Check if token is expired
        reset_token_exp = user.get('reset_token_exp')
        if isinstance(reset_token_exp, datetime):
            # Convert reset_token_exp to UTC naive datetime if it's timezone-aware
            reset_token_exp = reset_token_exp.replace(tzinfo=None)
        if datetime.utcnow() > reset_token_exp:
            return jsonify({
                'status': False,
                'message': 'Reset token has expired'
            }), 400
            
        # Hash new password
        hashed_password = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        
        # Update password and remove reset token
        user.reference.update({
            'password': hashed_password,
            'reset_token': firestore.DELETE_FIELD,
            'reset_token_exp': firestore.DELETE_FIELD
        })
        
        return jsonify({
            'status': True,
            'message': 'Password has been reset successfully'
        }), 200
        
    except Exception as e:
        return jsonify({
            'status': False,
            'message': str(e)
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port)
