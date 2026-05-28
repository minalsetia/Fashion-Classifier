#Central configuration for the Fashion Attribute Classification app.
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent  # project root
DATA_DIR = BASE_DIR / "data"
IMAGES_DIR = DATA_DIR / "images"
MODELS_DIR = BASE_DIR / "models"

METADATA_CSV = DATA_DIR / "metadata.csv"          # raw scraped products
LABELED_CSV = DATA_DIR / "labeled.csv"            # after labeler.py runs
DB_PATH = BASE_DIR / "predictions.db"             # SQLite tracking DB

# Models
GENDER_MODEL_PATH = MODELS_DIR / "gender_model.pt"
SLEEVE_MODEL_PATH = MODELS_DIR / "sleeve_model.pt"

# Scraping
TARGET_IMAGE_COUNT = 500  # how many product images to collect

# we get a balanced spread of men/women tops
SCRAPE_QUERIES = [
    {"query": "men-tshirts", "gender_hint": "men"},
    {"query": "men-shirts", "gender_hint": "men"},
    {"query": "women-tshirts", "gender_hint": "women"},
    {"query": "women-shirts", "gender_hint": "women"},
    {"query": "women-tops", "gender_hint": "women"},
    {"query": "men-kurtas", "gender_hint": "men"},
]

REQUEST_TIMEOUT = 20          
SLEEP_BETWEEN_REQUESTS = 1.5  
MAX_RETRIES = 3
FETCH_PRODUCT_DETAILS = True
PDP_WORKERS = 8   

# Labels
GENDER_CLASSES = ["female", "male"]
SLEEVE_CLASSES = ["full", "half"]
SLEEVE_LENGTH_MAP = {
    "long sleeves": "full",
    "short sleeves": "half",
}

# Structured sleeve lengths we deliberately drop
SLEEVE_LENGTH_EXCLUDE = [
    "sleeveless", "three-quarter sleeves", "cap sleeves", "short kimono sleeves",
]

# Fallback weak-supervision keyword rules
SLEEVE_KEYWORDS = {
    "full": ["full sleeve", "long sleeve", "full-sleeve", "long-sleeve"],
    "half": ["half sleeve", "short sleeve", "half-sleeve", "short-sleeve"],
}
# Text containing any of these is dropped from the sleeve dataset (ambiguous).
SLEEVE_EXCLUDE = ["sleeveless", "3/4 sleeve", "three quarter", "cap sleeve"]

GENDER_MAP = {
    "men": "male",
    "boys": "male",
    "women": "female",
    "girls": "female",
}

# Model / training
BACKBONE = "resnet18"     
IMG_SIZE = 224            # ResNet expects 224x224
BATCH_SIZE = 32
NUM_EPOCHS = 8
LEARNING_RATE = 1e-3      # for the new head; backbone is fine-tuned at lr/10
VAL_SPLIT = 0.2           # 20% held out for validation
RANDOM_SEED = 42
NUM_WORKERS = 4

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Version strings recorded with every prediction in the DB.
MODEL_VERSION = {
    "gender": f"{BACKBONE}_gender_v1",
    "sleeve": f"{BACKBONE}_sleeve_v1",
}
