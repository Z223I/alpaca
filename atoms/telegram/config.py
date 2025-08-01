import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class TelegramConfig:
    """Configuration class for Telegram bot settings."""

    BOT_TOKEN = os.getenv('BOT_TOKEN')
    DEFAULT_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
    ENABLED = os.getenv('TELEGRAM_ENABLED', 'true').lower() == 'true'
    PARSE_MODE = os.getenv('TELEGRAM_PARSE_MODE', 'HTML')
    TIMEOUT = int(os.getenv('TELEGRAM_TIMEOUT', '30'))
    DISABLE_NOTIFICATION = os.getenv('TELEGRAM_DISABLE_NOTIFICATION', 'false').lower() == 'true'

    # CSV file configuration
    CSV_FILE_PATH = os.path.join(os.getcwd(), '.telegram_users.csv')

    @classmethod
    def validate_config(cls):
        """Validate that required configuration is present."""
        if not cls.BOT_TOKEN:
            raise ValueError("BOT_TOKEN environment variable is required")

        if ':' not in cls.BOT_TOKEN:
            raise ValueError("BOT_TOKEN format is invalid. Expected format: 123456:ABC-DEF...")

        return True