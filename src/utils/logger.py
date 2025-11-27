import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

class Logger:
    """
    Centralized logging for the entire system.
    Logs to both file and console.
    """

    def __init__(self, name="SSA-PromptTuning", log_dir="outputs/logs", log_level=logging.INFO):
        """
        Initialize logger.
        
        Args:
            name: Logger name
            log_dir: Directory to save log files
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger(name)
        self.logger.setLevel(log_level)
        
        # Prevent duplicate handlers
        if self.logger.hasHandlers():
            self.logger.handlers.clear()
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        simple_formatter = logging.Formatter(
            '[%(levelname)s] %(message)s'
        )
        
        # Console handler (simple format)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(simple_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler (detailed format)
        log_file = self.log_dir / f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(detailed_formatter)
        self.logger.addHandler(file_handler)
        
        self.log_file = log_file
    
    def info(self, message):
        """Log info message"""
        self.logger.info(message)
    
    def debug(self, message):
        """Log debug message"""
        self.logger.debug(message)
    
    def warning(self, message):
        """Log warning message"""
        self.logger.warning(message)
    
    def error(self, message):
        """Log error message"""
        self.logger.error(message)
    
    def critical(self, message):
        """Log critical message"""
        self.logger.critical(message)
    
    def section(self, title):
        """Log a section header"""
        self.info(f"\n{'='*60}")
        self.info(f"  {title}")
        self.info(f"{'='*60}\n")
    
    def get_log_file(self):
        """Get path to current log file"""
        return str(self.log_file)


# Global logger instance
_logger = None

def get_logger(name = "SSA-PromptTuning"):
    """Get or create global logger instance"""
    global _logger
    if _logger is None:
        _logger = Logger(name)
    return _logger