# app.py
import os
import logging
from datetime import timedelta
from flask import Flask, render_template, request, redirect, url_for, jsonify
from flask_jwt_extended import (
    JWTManager, create_access_token, create_refresh_token,
    jwt_required, get_jwt_identity, verify_jwt_in_request, get_jwt
)
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps

# Attempt to import transformers, but keep a safe fallback if it's not available
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
    TRANSFORMERS_AVAILABLE = True
except Exception:
    TRANSFORMERS_AVAILABLE = False

logging.basicConfig(level=logging.INFO) 
logger = logging.getLogger(__name__)

app = Flask(__name__, static_folder="static", template_folder="templates")

# ---------------- JWT Configuration ----------------
app.config['JWT_SECRET_KEY'] = os.environ.get('JWT_SECRET_KEY', 'your-secret-key-change-in-production-123456789')
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=1)
app.config['JWT_REFRESH_TOKEN_EXPIRES'] = timedelta(days=30)
app.config['JWT_TOKEN_LOCATION'] = ['cookies']
app.config['JWT_COOKIE_SECURE'] = False  # Set to True in production with HTTPS
app.config['JWT_COOKIE_CSRF_PROTECT'] = False  # Enable in production
app.config['JWT_ACCESS_COOKIE_PATH'] = '/'
app.config['JWT_REFRESH_COOKIE_PATH'] = '/token/refresh'

jwt = JWTManager(app)

# ---------------- In-memory user database (Replace with real database in production) ----------------
USERS_DB = {
    'admin': {
        'password': generate_password_hash('admin123'),
        'email': 'admin@example.com',
        'role': 'admin'
    },
    'demo': {
        'password': generate_password_hash('demo123'),
        'email': 'demo@example.com',
        'role': 'user'
    },
    # Add your own users below:
    'abhinav': {
        'password': generate_password_hash('abhinav2024'),
        'email': 'abhinav@example.com',
        'role': 'user'
    },
    'john': {
        'password': generate_password_hash('john123'),
        'email': 'john@company.com',
        'role': 'user'
    },
    'sarah': {
        'password': generate_password_hash('sarah456'),
        'email': 'sarah@example.com',
        'role': 'admin'
    }
}

# ---------------- Authentication helpers ----------------
def login_required_page(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            verify_jwt_in_request()
            return f(*args, **kwargs)
        except Exception:
            return redirect(url_for('login'))
    return decorated_function

# ---------------- Model loading (resilient) ----------------
MODEL_NAME = "distilbert-base-uncased-finetuned-sst-2-english"
classifier = None

if TRANSFORMERS_AVAILABLE:
    try:
        logger.info(f"Loading tokenizer & model: {MODEL_NAME} (this may take a moment)...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)
        logger.info("Model loaded.")
    except Exception as e:
        logger.exception("Failed to load transformers model. Falling back to a dummy classifier. Error: %s", e)
        TRANSFORMERS_AVAILABLE = False

if not TRANSFORMERS_AVAILABLE:
    # Fallback classifier: returns NEUTRAL-like response so app continues to work when transformers aren't installed
    logger.info("Using fallback classifier (transformers not available). Install `transformers` for real classification.")
    def _dummy_classifier(text, truncation=True, max_length=512):
        # Very simple heuristic: look for positive/negative words for a rough label
        t = text.lower()
        positive_words = ["improve", "improves", "improved", "better", "significant", "increase", "enhance"]
        negative_words = ["decrease", "fail", "limitations", "poor", "worse", "not"]
        score = 0.5
        label = "NEUTRAL"
        if any(w in t for w in positive_words):
            label = "POSITIVE"
            score = 0.85
        elif any(w in t for w in negative_words):
            label = "NEGATIVE"
            score = 0.75
        return [{"label": label, "score": score}]
    classifier = _dummy_classifier

# ---------------- Small helpers ----------------
def _normalize_text(text: str) -> str:
    return (text or "").strip().lower()

# ---------------- Academic text check ----------------
def is_academic_text(text: str) -> bool:
    """
    Heuristic to check whether the text looks like an academic abstract:
    - Must have reasonable length (>= 15 words)
    - Must contain at least one academic keyword
    """
    if not text or not text.strip():
        return False

    text_lower = _normalize_text(text)
    words = text_lower.split()
    if len(words) < 15:
        # allow slightly shorter abstracts if they contain very strong academic keywords
        strong_keywords = ["randomized", "methodology", "controlled trial", "experiment", "dataset", "regression", "econometrics"]
        if not any(k in text_lower for k in strong_keywords):
            return False

    academic_keywords = [
        "study", "research", "paper", "proposes", "investigates", "analysis",
        "experiment", "methodology", "results", "objective", "findings",
        "model", "dataset", "approach", "algorithm", "performance",
        "evaluation", "conclusion", "abstract", "method", "randomized", "trial", "we show", "we propose"
    ]
    return any(word in text_lower for word in academic_keywords)

# ---------------- Field detection (weighted scoring) ----------------
def detect_academic_field(text: str) -> str:
    """
    Uses weighted keyword matching to pick the most likely academic field.
    Returns a field code such as 'econ.GN', 'cs.LG', etc.
    """
    text = _normalize_text(text)

    # field -> keyword list (phrase keywords are allowed)
    fields = {
        "cs.LG": [
            "machine learning", "deep learning", "neural network", "training",
            "classification", "regression model", "supervised", "unsupervised",
            "reinforcement learning", "attention", "embedding"
        ],
        "cs.AI": [
            "artificial intelligence", "agent", "reasoning", "knowledge graph",
            "planning system", "intelligent system", "knowledge representation"
        ],
        "cs.CL": [
            "nlp", "natural language", "language model", "bert", "gpt", "text processing",
            "sentiment analysis", "tokenization", "translation", "summarization"
        ],
        "cs.CV": [
            "image", "vision", "object detection", "segmentation",
            "computer vision", "recognition", "camera", "video", "pose estimation", "yolo"
        ],
        "eess.SP": [
            "signal", "frequency", "fft", "audio processing", "speech signal",
            "filtering", "waveform", "spectrogram", "time-frequency", "sensor signal"
        ],
        "stat.ML": [
            "bayesian", "statistical model", "likelihood", "probabilistic",
            "markov chain", "mcmc", "variance", "distribution", "hypothesis test", "estimator"
        ],
        "q-bio": [
            "genomics", "protein", "biological", "dna", "cellular",
            "enzyme", "molecular", "transcription", "sequencing", "biosystem"
        ],
        "physics.comp-ph": [
            "simulation", "quantum", "particle", "atomic", "molecular dynamics",
            "computational physics", "monte carlo", "lattice", "density functional"
        ],
        "econ.GN": [
            "economics", "market", "finance", "gdp", "inflation",
            "trade", "forecast", "economic model", "policy impact", "regression discontinuity",
            "macroeconomic", "microeconomic", "econometrics", "household consumption", "labor market"
        ]
    }

    # Score each field
    scores = {field: 0 for field in fields}
    for field, keywords in fields.items():
        for kw in keywords:
            if kw in text:
                # longer/multi-word keywords get more weight
                weight = 2 if " " in kw else 1
                scores[field] += weight

    # If nothing matched, try single-word fallback keywords (reduce false negatives)
    if all(score == 0 for score in scores.values()):
        fallback_map = {
            "cs.LG": ["model", "training", "dataset"],
            "econ.GN": ["economy", "prices", "inflation", "gdp"],
            "cs.CL": ["text", "language", "sentence"],
            "cs.CV": ["image", "pixel", "frame"],
            "q-bio": ["cell", "gene", "protein"],
            "eess.SP": ["signal", "frequency"],
            "stat.ML": ["probability", "bayes", "regression"],
            "physics.comp-ph": ["simulation", "quantum"]
        }
        for field, kws in fallback_map.items():
            for kw in kws:
                if kw in text:
                    scores[field] += 1

    # Choose the field with the highest score
    top_field = max(scores, key=scores.get)
    if scores[top_field] == 0:
        return "cs.Other"

    # tie-breaker: if cs.LG ties with a more specific field, prefer the more specific field
    # This helps when ML words are present alongside domain-specific words (e.g., economics + model)
    # If cs.LG has the same score as another non-ML field and that other field has any unique keyword,
    # prefer the non-ML field.
    tied = [f for f, s in scores.items() if s == scores[top_field]]
    if len(tied) > 1 and "cs.LG" in tied:
        # try to prefer econ, q-bio, cs.CL, cs.CV, etc., if they are also tied
        prefer_order = ["econ.GN", "q-bio", "cs.CL", "cs.CV", "eess.SP", "stat.ML", "physics.comp-ph", "cs.AI"]
        for p in prefer_order:
            if p in tied:
                return p

    return top_field

def get_field_name(field_code: str) -> str:
    names = {
        "cs.LG": "Machine Learning",
        "cs.AI": "Artificial Intelligence",
        "cs.CL": "Natural Language Processing",
        "cs.CV": "Computer Vision",
        "eess.SP": "Signal Processing",
        "stat.ML": "Statistical Machine Learning",
        "q-bio": "Quantitative Biology",
        "physics.comp-ph": "Computational Physics",
        "econ.GN": "Economics",
        "cs.Other": "Other Computer Science"
    }
    return names.get(field_code, "Unknown Field")

# ---------------- Sentiment / polarity classification ----------------
def classify_abstract(text: str):
    """
    Returns a single dict like {'label': 'POSITIVE', 'score': 0.95}
    Using the loaded classifier or the fallback.
    """
    if not text or not text.strip():
        return None
    try:
        # pipeline returns a list, take first element
        out = classifier(text, truncation=True, max_length=512)
        if isinstance(out, list) and len(out) > 0:
            return out[0]
        return out
    except Exception as e:
        logger.exception("Error while classifying text: %s", e)
        # fallback simple response
        return {"label": "NEUTRAL", "score": 0.5}

# ---------------- Sample abstracts ----------------
SAMPLE_ABSTRACTS = {
    "Select a sample...": "",
    "Renewable Energy & Power Grids": (
        "This paper investigates the integration of renewable energy sources "
        "into existing power grids, focusing on optimizing energy distribution "
        "and minimizing losses. We propose a novel algorithm for load balancing "
        "that improves grid stability and efficiency."
    ),
    "Deep Learning & Computer Vision": (
        "We present a novel deep learning architecture for image segmentation "
        "that improves accuracy and inference speed. Experiments on public "
        "benchmarks demonstrate state-of-the-art performance."
    ),
    "Natural Language Processing": (
        "This work introduces a new approach to sentiment analysis using pretrained "
        "transformers and contrastive learning, achieving improved robustness across domains."
    ),
    "Medical Research": (
        "Our study examines the efficacy of a new therapeutic approach in treating "
        "a chronic disease. We conducted a randomized controlled trial to evaluate outcomes."
    ),
    "Climate Science": (
        "This research analyzes the impact of climate change on regional precipitation "
        "patterns using long-term datasets and climate simulations."
    ),
    "Economics Example": (
        "This paper analyzes the impact of inflation and GDP growth on market stability. "
        "Using econometric models and macroeconomic panel data, we estimate policy effects "
        "and provide forecasts to support evidence-based economic policy."
    )
}

# ---------------- Routes ----------------
@app.route("/login", methods=["GET"])
def login():
    return render_template("login.html")


@app.route("/api/login", methods=["POST"])
def api_login():
    data = request.get_json() or {}
    username = data.get('username', '')
    password = data.get('password', '')

    if not username or not password:
        return jsonify({'error': 'Username and password required'}), 400

    user = USERS_DB.get(username)
    if not user or not check_password_hash(user['password'], password):
        return jsonify({'error': 'Invalid username or password'}), 401

    # Create tokens
    access_token = create_access_token(
        identity=username,
        additional_claims={'role': user['role'], 'email': user['email']}
    )
    refresh_token = create_refresh_token(identity=username)

    # Create response with cookies
    response = jsonify({
        'message': 'Login successful',
        'user': {
            'username': username,
            'email': user['email'],
            'role': user['role']
        }
    })

    # Set JWT cookies
    response.set_cookie('access_token_cookie', access_token,
                       httponly=True, samesite='Lax', max_age=3600)
    response.set_cookie('refresh_token_cookie', refresh_token,
                       httponly=True, samesite='Lax', max_age=2592000)

    return response, 200


@app.route("/api/register", methods=["POST"])
def api_register():
    data = request.get_json() or {}
    username = data.get('username', '').strip()
    password = data.get('password', '')
    email = data.get('email', '').strip()

    if not username or not password or not email:
        return jsonify({'error': 'All fields are required'}), 400

    if username in USERS_DB:
        return jsonify({'error': 'Username already exists'}), 409

    if len(password) < 6:
        return jsonify({'error': 'Password must be at least 6 characters'}), 400

    # Add new user
    USERS_DB[username] = {
        'password': generate_password_hash(password),
        'email': email,
        'role': 'user'
    }

    return jsonify({'message': 'Registration successful'}), 201


@app.route("/api/logout", methods=["POST"])
def api_logout():
    response = jsonify({'message': 'Logout successful'})
    response.set_cookie('access_token_cookie', '', expires=0)
    response.set_cookie('refresh_token_cookie', '', expires=0)
    return response, 200


@app.route("/api/user", methods=["GET"])
@jwt_required()
def get_user():
    current_user = get_jwt_identity()
    claims = get_jwt()

    return jsonify({
        'username': current_user,
        'email': claims.get('email'),
        'role': claims.get('role')
    }), 200


@app.route("/", methods=["GET"])
@login_required_page
def index():
    current_user = get_jwt_identity()
    return render_template("index.html",
                           abstract="",
                           label=None,
                           confidence=None,
                           field_code=None,
                           field_name=None,
                           sample_options=SAMPLE_ABSTRACTS,
                           username=current_user)


@app.route("/predict", methods=["POST"]) 
@login_required_page
def predict():
    current_user = get_jwt_identity()
    abstract = request.form.get("abstract", "")
    selected_sample = request.form.get("sample_select", "")

    if selected_sample and selected_sample != "Select a sample...":
        abstract = SAMPLE_ABSTRACTS.get(selected_sample, "")

    if not abstract.strip():
        return render_template("index.html",
                               warning="⚠️ Please enter an abstract.",
                               abstract="",
                               sample_options=SAMPLE_ABSTRACTS,
                               username=current_user)

    if not is_academic_text(abstract):
        return render_template("index.html",
                               warning="⚠️ This does NOT look like an academic abstract.",
                               abstract=abstract,
                               sample_options=SAMPLE_ABSTRACTS,
                               username=current_user)

    field_code = detect_academic_field(abstract)
    field_name = get_field_name(field_code)

    result = classify_abstract(abstract)
    label = result.get("label") if result else None
    confidence = round(result.get("score", 0) * 100, 2) if result else None

    return render_template("index.html",
                           abstract=abstract,
                           label=label,
                           confidence=confidence,
                           field_code=field_code,
                           field_name=field_name,
                           sample_options=SAMPLE_ABSTRACTS,
                           username=current_user)


# ---------------- Run ----------------
if __name__ == "__main__":
    app.run(debug=True)
