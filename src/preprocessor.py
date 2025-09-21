"""
🧹 DATA PREPROCESSING MODULE
Professional data cleaning and feature engineering for forecasting
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from config import PROCESSED_DATA_PATH, LAG_FEATURES, ROLLING_WINDOWS

class SalesPreprocessor:
    """Complete data preprocessing pipeline"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_stats = {}
        
    def clean_data(self, data):
        """Step 1: Clean raw sales data"""
        print("🧹 STEP 1: DATA CLEANING")
        print("=" * 30)
        
        cleaned_data = data.copy()
        initial_shape = cleaned_data.shape
        
        # Remove exact duplicates
        cleaned_data = cleaned_data.drop_duplicates()
        duplicates_removed = initial_shape[0] - cleaned_data.shape[0]
        print(f"✅ Removed {duplicates_removed} duplicate rows")
        
        # Handle missing values
        missing_before = cleaned_data.isnull().sum().sum()
        
        # Forward fill sales values within each store
        for store_id in cleaned_data['store_id'].unique():
            store_mask = cleaned_data['store_id'] == store_id
            cleaned_data.loc[store_mask, 'sales'] = (
                cleaned_data.loc[store_mask, 'sales'].fillna(method='ffill')
            )
        
        # Fill remaining missing values with store averages
        store_means = cleaned_data.groupby('store_id')['sales'].mean()
        for store_id in cleaned_data['store_id'].unique():
            store_mask = cleaned_data['store_id'] == store_id
            cleaned_data.loc[store_mask, 'sales'] = (
                cleaned_data.loc[store_mask, 'sales'].fillna(store_means[store_id])
            )
        
        missing_after = cleaned_data.isnull().sum().sum()
        print(f"✅ Handled {missing_before - missing_after} missing values")
        
        # Remove statistical outliers using IQR method per store
        outliers_removed = 0
        for store_id in cleaned_data['store_id'].unique():
            store_mask = cleaned_data['store_id'] == store_id
            store_sales = cleaned_data.loc[store_mask, 'sales']
            
            Q1 = store_sales.quantile(0.25)
            Q3 = store_sales.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outlier_mask = (store_sales < lower_bound) | (store_sales > upper_bound)
            outliers_in_store = outlier_mask.sum()
            outliers_removed += outliers_in_store
            
            # Remove outliers for this store
            cleaned_data = cleaned_data[~(store_mask & outlier_mask)]
        
        print(f"✅ Removed {outliers_removed} outlier records")
        
        # Sort by store and date
        cleaned_data = cleaned_data.sort_values(['store_id', 'date']).reset_index(drop=True)
        
        print(f"✅ Final clean data shape: {cleaned_data.shape}")
        print("   Data cleaning completed!")
        
        return cleaned_data
    
    def create_time_features(self, data):
        """Step 2: Create time-based features"""
        print("\n🕐 STEP 2: TIME-BASED FEATURES")
        print("=" * 35)
        
        featured_data = data.copy()
        
        # Extract comprehensive time features
        featured_data['year'] = featured_data['date'].dt.year
        featured_data['month'] = featured_data['date'].dt.month
        featured_data['quarter'] = featured_data['date'].dt.quarter
        featured_data['day_of_week'] = featured_data['date'].dt.dayofweek
        featured_data['day_of_month'] = featured_data['date'].dt.day
        featured_data['day_of_year'] = featured_data['date'].dt.dayofyear
        featured_data['week_of_year'] = featured_data['date'].dt.isocalendar().week
        
        # Boolean time features
        featured_data['is_weekend'] = (featured_data['day_of_week'] >= 5).astype(int)
        featured_data['is_month_start'] = featured_data['date'].dt.is_month_start.astype(int)
        featured_data['is_month_end'] = featured_data['date'].dt.is_month_end.astype(int)
        featured_data['is_quarter_start'] = featured_data['date'].dt.is_quarter_start.astype(int)
        featured_data['is_quarter_end'] = featured_data['date'].dt.is_quarter_end.astype(int)
        
        # Cyclical encoding for periodic features
        featured_data['month_sin'] = np.sin(2 * np.pi * featured_data['month'] / 12)
        featured_data['month_cos'] = np.cos(2 * np.pi * featured_data['month'] / 12)
        featured_data['day_of_week_sin'] = np.sin(2 * np.pi * featured_data['day_of_week'] / 7)
        featured_data['day_of_week_cos'] = np.cos(2 * np.pi * featured_data['day_of_week'] / 7)
        
        print("✅ Created time-based features:")
        print("   - Year, month, quarter, day features")
        print("   - Boolean weekend/month-end flags")
        print("   - Cyclical encodings for periodicity")
        
        return featured_data
    
    def create_lag_features(self, data):
        """Step 3: Create lag features"""
        print("\n📈 STEP 3: LAG FEATURES")
        print("=" * 25)
        
        featured_data = data.copy()
        
        print(f"Creating lag features for periods: {LAG_FEATURES}")
        
        # Create lag features for each store separately
        for store_id in featured_data['store_id'].unique():
            store_mask = featured_data['store_id'] == store_id
            store_data = featured_data[store_mask].copy().sort_values('date')
            
            for lag in LAG_FEATURES:
                col_name = f'sales_lag_{lag}'
                featured_data.loc[store_mask, col_name] = store_data['sales'].shift(lag)
                
                # Also create lag features for promotions (important for forecasting)
                promo_col_name = f'promotion_lag_{lag}'
                featured_data.loc[store_mask, promo_col_name] = store_data['promotion'].shift(lag)
        
        lag_cols_created = len([col for col in featured_data.columns if 'lag' in col])
        print(f"✅ Created {lag_cols_created} lag features")
        
        return featured_data
    
    def create_rolling_features(self, data):
        """Step 4: Create rolling window features"""
        print("\n📊 STEP 4: ROLLING WINDOW FEATURES")
        print("=" * 38)
        
        featured_data = data.copy()
        
        print(f"Creating rolling features for windows: {ROLLING_WINDOWS}")
        
        # Create rolling features for each store separately
        for store_id in featured_data['store_id'].unique():
            store_mask = featured_data['store_id'] == store_id
            store_data = featured_data[store_mask].copy().sort_values('date')
            
            for window in ROLLING_WINDOWS:
                # Rolling statistics for sales
                featured_data.loc[store_mask, f'sales_rolling_mean_{window}'] = (
                    store_data['sales'].rolling(window=window, min_periods=1).mean()
                )
                featured_data.loc[store_mask, f'sales_rolling_std_{window}'] = (
                    store_data['sales'].rolling(window=window, min_periods=1).std()
                )
                featured_data.loc[store_mask, f'sales_rolling_min_{window}'] = (
                    store_data['sales'].rolling(window=window, min_periods=1).min()
                )
                featured_data.loc[store_mask, f'sales_rolling_max_{window}'] = (
                    store_data['sales'].rolling(window=window, min_periods=1).max()
                )
                
                # Rolling sum for promotions (how many promotions in window)
                featured_data.loc[store_mask, f'promotion_rolling_sum_{window}'] = (
                    store_data['promotion'].rolling(window=window, min_periods=1).sum()
                )
        
        rolling_cols_created = len([col for col in featured_data.columns if 'rolling' in col])
        print(f"✅ Created {rolling_cols_created} rolling window features")
        
        return featured_data
    
    def create_interaction_features(self, data):
        """Step 5: Create interaction features"""
        print("\n🔗 STEP 5: INTERACTION FEATURES")
        print("=" * 34)
        
        featured_data = data.copy()
        
        # Business logic interactions
        featured_data['promotion_weekend'] = featured_data['promotion'] * featured_data['is_weekend']
        featured_data['promotion_holiday'] = featured_data['promotion'] * featured_data['is_holiday']
        featured_data['holiday_weekend'] = featured_data['is_holiday'] * featured_data['is_weekend']
        
        # Seasonal interactions
        featured_data['promotion_month'] = featured_data['promotion'] * featured_data['month']
        featured_data['weekend_month'] = featured_data['is_weekend'] * featured_data['month']
        
        # Store-specific interactions
        featured_data['store_promotion'] = featured_data['store_id'] * featured_data['promotion']
        featured_data['store_weekend'] = featured_data['store_id'] * featured_data['is_weekend']
        
        interaction_cols = ['promotion_weekend', 'promotion_holiday', 'holiday_weekend',
                           'promotion_month', 'weekend_month', 'store_promotion', 'store_weekend']
        print(f"✅ Created {len(interaction_cols)} interaction features")
        
        return featured_data
    
    def create_target_encoding_features(self, data):
        """Step 6: Create target encoding features"""
        print("\n🎯 STEP 6: TARGET ENCODING FEATURES")
        print("=" * 38)
        
        featured_data = data.copy()
        
        # Mean encoding for categorical features (with smoothing to prevent overfitting)
        smoothing = 100  # Smoothing factor
        
        # Store average sales (excluding current row)
        store_means = featured_data.groupby('store_id')['sales'].transform('mean')
        global_mean = featured_data['sales'].mean()
        store_counts = featured_data.groupby('store_id')['sales'].transform('count')
        
        featured_data['store_mean_encoded'] = (
            (store_means * store_counts + global_mean * smoothing) / 
            (store_counts + smoothing)
        )
        
        # Day of week average sales
        dow_means = featured_data.groupby('day_of_week')['sales'].transform('mean')
        dow_counts = featured_data.groupby('day_of_week')['sales'].transform('count')
        
        featured_data['dow_mean_encoded'] = (
            (dow_means * dow_counts + global_mean * smoothing) / 
            (dow_counts + smoothing)
        )
        
        # Month average sales
        month_means = featured_data.groupby('month')['sales'].transform('mean')
        month_counts = featured_data.groupby('month')['sales'].transform('count')
        
        featured_data['month_mean_encoded'] = (
            (month_means * month_counts + global_mean * smoothing) / 
            (month_counts + smoothing)
        )
        
        print("✅ Created target encoding features:")
        print("   - Store mean encoded")
        print("   - Day of week mean encoded") 
        print("   - Month mean encoded")
        
        return featured_data
    
    def prepare_modeling_data(self, data, target_col='sales'):
        """Step 7: Final preparation for modeling"""
        print("\n🎯 STEP 7: MODELING PREPARATION")
        print("=" * 35)
        
        modeling_data = data.copy()
        
        # Remove rows with missing target values
        initial_rows = len(modeling_data)
        modeling_data = modeling_data.dropna(subset=[target_col])
        target_missing = initial_rows - len(modeling_data)
        if target_missing > 0:
            print(f"✅ Removed {target_missing} rows with missing target values")
        
        # Handle missing feature values
        feature_cols = [col for col in modeling_data.columns 
                       if col not in ['date', 'store_id', target_col]]
        
        missing_summary = {}
        for col in feature_cols:
            missing_count = modeling_data[col].isnull().sum()
            if missing_count > 0:
                if modeling_data[col].dtype in ['float64', 'int64']:
                    fill_value = modeling_data[col].median()
                    modeling_data[col] = modeling_data[col].fillna(fill_value)
                    missing_summary[col] = f"filled {missing_count} with median {fill_value:.2f}"
                else:
                    mode_value = modeling_data[col].mode().iloc[0] if len(modeling_data[col].mode()) > 0 else 0
                    modeling_data[col] = modeling_data[col].fillna(mode_value)
                    missing_summary[col] = f"filled {missing_count} with mode {mode_value}"
        
        if missing_summary:
            print("✅ Handled missing feature values:")
            for col, summary in missing_summary.items():
                print(f"   - {col}: {summary}")
        else:
            print("✅ No missing feature values found")
        
        # Store feature statistics
        self.feature_stats = {
            'total_features': len(feature_cols),
            'numeric_features': len([col for col in feature_cols if modeling_data[col].dtype in ['float64', 'int64']]),
            'categorical_features': len([col for col in feature_cols if modeling_data[col].dtype == 'object']),
            'feature_names': feature_cols
        }
        
        print(f"✅ Final modeling data ready!")
        print(f"   Shape: {modeling_data.shape}")
        print(f"   Features: {self.feature_stats['total_features']}")
        print(f"   Numeric: {self.feature_stats['numeric_features']}")
        print(f"   Missing values: {modeling_data.isnull().sum().sum()}")
        
        return modeling_data
    
    def save_processed_data(self, data, filename='processed_sales_data.csv'):
        """Save processed data with metadata"""
        filepath = os.path.join(PROCESSED_DATA_PATH, filename)
        data.to_csv(filepath, index=False)
        
        # Save feature metadata
        metadata_path = os.path.join(PROCESSED_DATA_PATH, 'feature_metadata.txt')
        with open(metadata_path, 'w') as f:
            f.write("FEATURE ENGINEERING SUMMARY\n")
            f.write("=" * 30 + "\n\n")
            f.write(f"Total features: {self.feature_stats['total_features']}\n")
            f.write(f"Numeric features: {self.feature_stats['numeric_features']}\n")
            f.write(f"Categorical features: {self.feature_stats['categorical_features']}\n\n")
            f.write("FEATURE LIST:\n")
            for i, feature in enumerate(self.feature_stats['feature_names'], 1):
                f.write(f"{i:3d}. {feature}\n")
        
        print(f"✅ Processed data saved to: {filepath}")
        print(f"✅ Feature metadata saved to: {metadata_path}")
        
    def full_preprocessing_pipeline(self, raw_data):
        """Execute complete preprocessing pipeline"""
        print("🚀 STARTING COMPLETE PREPROCESSING PIPELINE")
        print("=" * 60)
        
        # Execute all preprocessing steps
        cleaned_data = self.clean_data(raw_data)
        time_featured_data = self.create_time_features(cleaned_data)
        lag_featured_data = self.create_lag_features(time_featured_data)
        rolling_featured_data = self.create_rolling_features(lag_featured_data)
        interaction_featured_data = self.create_interaction_features(rolling_featured_data)
        encoded_data = self.create_target_encoding_features(interaction_featured_data)
        modeling_data = self.prepare_modeling_data(encoded_data)
        
        # Save processed data
        self.save_processed_data(modeling_data)
        
        print("\n🎉 PREPROCESSING PIPELINE COMPLETED!")
        print("=" * 45)
        print(f"📊 Input shape: {raw_data.shape}")
        print(f"📈 Output shape: {modeling_data.shape}")
        print(f"🔧 Features created: {modeling_data.shape[1] - raw_data.shape[1]}")
        
        # Feature breakdown
        feature_types = {
            'time_features': len([col for col in modeling_data.columns if any(x in col for x in ['year', 'month', 'day', 'week', 'sin', 'cos'])]),
            'lag_features': len([col for col in modeling_data.columns if 'lag' in col]),
            'rolling_features': len([col for col in modeling_data.columns if 'rolling' in col]),
            'interaction_features': len([col for col in modeling_data.columns if any(x in col for x in ['promotion_', 'holiday_', 'weekend_', 'store_'])]),
            'encoding_features': len([col for col in modeling_data.columns if 'encoded' in col])
        }
        
        print(f"\n🏷️  FEATURE BREAKDOWN:")
        for feature_type, count in feature_types.items():
            if count > 0:
                print(f"   {feature_type}: {count}")
        
        print(f"\n📁 Ready for modeling!")
        return modeling_data

if __name__ == "__main__":
    # Test preprocessing pipeline
    print("🧪 TESTING PREPROCESSING MODULE")
    
    # This would normally load real data
    from src.data_loader import DataLoader
    
    try:
        # Load data
        data_loader = DataLoader()
        raw_data = data_loader.load_data()
        
        # Run preprocessing
        preprocessor = SalesPreprocessor()
        processed_data = preprocessor.full_preprocessing_pipeline(raw_data)
        
        print("✅ Preprocessing module test completed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()