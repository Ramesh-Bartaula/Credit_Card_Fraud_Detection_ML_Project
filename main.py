import sys
from pathlib import Path
import logging

from src import train_fraud_model

# Setup logging to see what's happening
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def main():
    """
    Executive Orchestrator.
    It doesn't do the math; it tells the modules when to work.
    """
    print("\n" + "=" * 60)
    print(" 🛡️  FRAUD DETECTION SYSTEM - PIPELINE STARTING")
    print("=" * 60)

    try:
        # We call the main training function from src/models/train_model.py
        # Note: train_fraud_model() already handles Step 1 (Data) and Step 2 (Training)
        logger.info("Executing Training and Evaluation Pipeline...")

        train_fraud_model()

        print("\n" + "=" * 60)
        print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        return True

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        print("\n❌ CRITICAL ERROR: Please check logs above.")
        return False


if __name__ == "__main__":
    success = main()
    # Exit with 0 if success, 1 if failure (standard for production/automation)
    sys.exit(0 if success else 1)