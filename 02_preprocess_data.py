"""
🚀 STEP 2: DATA PREPROCESSING
Clean raw data and engineer features for forecasting models
"""

import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    """Execute complete data preprocessing pipeline"""
    
    print("🛍️  RETAIL SALES FORECASTING - DATA PREPROCESSING")
    print("=" * 65)
    
    try:
        # Import modules
        from src.data_loader import DataLoader
        from src.preprocessor import SalesPreprocessor
        
        # Step 1: Load raw data
        print("STEP 1: Loading raw data...")
        loader = DataLoader()
        raw_data = loader.load_data()
        
        if raw_data is None or len(raw_data) == 0:
            print("❌ No data available. Please run 01_generate_data.py first.")
            return None
            
        # Step 2: Initialize preprocessor
        print("\nSTEP 2: Initializing preprocessor...")
        preprocessor = SalesPreprocessor()
        
        # Step 3: Execute full preprocessing pipeline
        print("\nSTEP 3: Executing preprocessing pipeline...")
        processed_data = preprocessor.full_preprocessing_pipeline(raw_data)
        
        # Step 4: Show results summary
        print("\n📋 PREPROCESSING SUMMARY")
        print("=" * 35)
        print(f"✅ Original data shape: {raw_data.shape}")
        print(f"✅ Processed data shape: {processed_data.shape}")
        print(f"✅ New features created: {processed_data.shape[1] - raw_data.shape[1]}")
        
        # Show sample of processed data
        print(f"\n📊 SAMPLE OF PROCESSED DATA:")
        sample_cols = ['date', 'store_id', 'sales', 'sales_lag_1', 'sales_lag_7', 
                      'sales_rolling_mean_7', 'promotion', 'is_weekend', 'month']
        available_cols = [col for col in sample_cols if col in processed_data.columns]
        print(processed_data[available_cols].head(10).to_string(index=False))
        
        # Feature categories summary
        all_features = [col for col in processed_data.columns if col not in ['date', 'store_id', 'sales']]
        
        feature_categories = {
            'Time Features': [col for col in all_features if any(x in col for x in ['year', 'month', 'day', 'week', 'quarter', 'sin', 'cos'])],
            'Lag Features': [col for col in all_features if 'lag' in col],
            'Rolling Features': [col for col in all_features if 'rolling' in col],
            'Interaction Features': [col for col in all_features if any(x in col for x in ['promotion_', 'holiday_', 'weekend_', 'store_'])],
            'Encoding Features': [col for col in all_features if 'encoded' in col],
            'Original Features': [col for col in all_features if col in ['promotion', 'is_holiday', 'day_of_week']]
        }
        
        print(f"\n🏷️  FEATURE CATEGORIES:")
        for category, features in feature_categories.items():
            if features:
                print(f"   {category}: {len(features)} features")
                if len(features) <= 5:
                    print(f"      {', '.join(features)}")
                else:
                    print(f"      {', '.join(features[:3])}... (+{len(features)-3} more)")
        
        print(f"\n🎯 READY FOR MODELING!")
        print(f"📁 Processed data saved to: data/processed/processed_sales_data.csv")
        print(f"📄 Feature metadata saved to: data/processed/feature_metadata.txt")
        print(f"📈 Next step: Run EDA and start model training")
        
        return processed_data
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all required modules are available")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    processed_data = main()