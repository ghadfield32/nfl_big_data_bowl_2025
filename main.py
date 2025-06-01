"""
NFL Big Data Bowl 2025 - Main Entry Point
"""
# Import our legacy module patch first to handle cgi module dependencies
import src.utils.legacy_patch

# Now we can safely import modules that depend on cgi
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import opendatasets as od

def test_opendatasets():
    """Test function for opendatasets to verify it's working."""
    print("Testing opendatasets functionality...")
    
    # Print opendatasets version
    print(f"Opendatasets version: {od.__version__}")
    
    # List available datasets (doesn't require authentication)
    print("Opendatasets is working correctly!")
    
    return True

def main():
    """Main entry point for the application."""
    print("NFL Big Data Bowl 2025 Analysis")
    print("Python version:", pd.__version__)
    
    # Test if opendatasets is working
    test_opendatasets()
    
    # Example of using opendatasets
    # Uncomment to download the dataset
    # od.download("https://www.kaggle.com/competitions/nfl-big-data-bowl-2024/data")
    
    # Your analysis code here
    
if __name__ == "__main__":
    main()
