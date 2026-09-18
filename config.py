import os

class Config:
    # Base directory of the project
    BASE_DIR = os.path.abspath(os.path.dirname(__file__))
    
    # Secret key for session signing
    SECRET_KEY = os.environ.get('SECRET_KEY', 'retailiq-secret-key-1859-prod')
    
    # Database path
    DATABASE_PATH = os.path.join(BASE_DIR, 'database', 'retail.db')
    
    # Data directory paths
    RAW_DATA_DIR = os.path.join(BASE_DIR, 'data', 'raw')
    PROCESSED_DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
    REPORTS_DIR = os.path.join(BASE_DIR, 'reports')
    
    # File uploads
    UPLOAD_FOLDER = RAW_DATA_DIR
    ALLOWED_EXTENSIONS = {'xlsx', 'csv'}
    MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100 MB max upload size
    
    # Create directories if they don't exist
    @classmethod
    def init_app(cls):
        os.makedirs(os.path.join(cls.BASE_DIR, 'database'), exist_ok=True)
        os.makedirs(cls.RAW_DATA_DIR, exist_ok=True)
        os.makedirs(cls.PROCESSED_DATA_DIR, exist_ok=True)
        os.makedirs(cls.REPORTS_DIR, exist_ok=True)
        os.makedirs(os.path.join(cls.BASE_DIR, 'static', 'css'), exist_ok=True)
        os.makedirs(os.path.join(cls.BASE_DIR, 'static', 'js'), exist_ok=True)
        os.makedirs(os.path.join(cls.BASE_DIR, 'static', 'images'), exist_ok=True)
        os.makedirs(os.path.join(cls.BASE_DIR, 'templates'), exist_ok=True)
        os.makedirs(os.path.join(cls.BASE_DIR, 'models', 'forecasting'), exist_ok=True)
        os.makedirs(os.path.join(cls.BASE_DIR, 'models', 'segmentation'), exist_ok=True)
