"""
📊 DATA LOADING & GENERATION MODULE
Generates realistic retail sales data and handles data loading
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys

# Add project root to path for config import
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from config import RAW_DATA_PATH

class SalesDataGenerator:
    """Generate realistic sample retail sales data"""
    
    def __init__(self, start_date='2020-01-01', end_date='2023-12-31', n_stores=5):
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.n_stores = n_stores
        
    def generate_data(self):
        """Generate comprehensive sales dataset with realistic patterns"""
        print("🏗️  Generating realistic retail sales data...")
        np.random.seed(42)
        
        dates = pd.date_range(start=self.start_date, end=self.end_date, freq='D')
        print(f"📅 Date range: {self.start_date.date()} to {self.end_date.date()}")
        print(f"📊 Total days: {len(dates)}")
        print(f"🏪 Number of stores: {self.n_stores}")
        
        all_data = []
        
        for store_id in range(1, self.n_stores + 1):
            print(f"   Generating data for Store {store_id}...")
            
            for date in dates:
                # Base sales with growth trend
                days_from_start = (date - self.start_date).days
                base_sales = 1000 + (days_from_start * 0.1)  # 0.1 per day growth
                
                # Seasonal patterns (yearly cycle)
                seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * date.dayofyear / 365.25)
                
                # Weekly patterns (higher sales on weekends)
                weekly_factor = 1.2 if date.weekday() >= 5 else 1.0
                
                # Store-specific performance factors
                store_factor = 0.8 + (store_id * 0.1)  # Stores perform differently
                
                # Holiday effects
                holiday_factor = self._get_holiday_factor(date)
                
                # Random promotions (15% chance)
                promotion = np.random.choice([0, 1], p=[0.85, 0.15])
                promotion_factor = 1.4 if promotion else 1.0
                
                # Calculate final sales with realistic noise
                sales = (base_sales * seasonal_factor * weekly_factor * 
                        store_factor * holiday_factor * promotion_factor)
                sales += np.random.normal(0, 50)  # Add noise
                sales = max(0, sales)  # Ensure non-negative
                
                # Create complete record
                record = {
                    'date': date,
                    'store_id': store_id,
                    'sales': round(sales, 2),
                    'is_holiday': holiday_factor > 1.0,
                    'promotion': promotion,
                    'day_of_week': date.weekday(),
                    'month': date.month,
                    'quarter': date.quarter,
                    'year': date.year,
                    'is_weekend': date.weekday() >= 5
                }
                all_data.append(record)
        
        df = pd.DataFrame(all_data)
        print(f"✅ Dataset generated! Shape: {df.shape}")
        return df
    
    def _get_holiday_factor(self, date):
        """Calculate holiday impact on sales"""
        # Major holidays with different impacts
        major_holidays = [
            (12, 25),  # Christmas - highest impact
            (12, 24),  # Christmas Eve
            (1, 1),    # New Year's Day
            (7, 4),    # Independence Day
            (11, 24),  # Black Friday (approximate)
            (11, 25),  # Day after Black Friday
        ]
        
        minor_holidays = [
            (2, 14),   # Valentine's Day
            (3, 17),   # St. Patrick's Day
            (10, 31),  # Halloween
        ]
        
        # Check for major holidays
        if (date.month, date.day) in major_holidays:
            return 1.8  # 80% increase
        
        # Check for minor holidays
        elif (date.month, date.day) in minor_holidays:
            return 1.3  # 30% increase
        
        # Holiday season (December)
        elif date.month == 12:
            return 1.2  # 20% increase during holiday season
        
        # Back-to-school season (August-September)
        elif date.month in [8, 9]:
            return 1.1  # 10% increase
        
        else:
            return 1.0  # Normal sales
    
    def save_data(self, filename='sales_data.csv'):
        """Generate and save the complete dataset"""
        print("🚀 GENERATING SAMPLE RETAIL SALES DATA")
        print("=" * 50)
        
        data = self.generate_data()
        filepath = os.path.join(RAW_DATA_PATH, filename)
        data.to_csv(filepath, index=False)
        
        # Data summary
        total_sales = data['sales'].sum()
        avg_daily_sales = data['sales'].mean()
        
        print("💰 BUSINESS SUMMARY:")
        print(f"   Total Sales: ${total_sales:,.2f}")
        print(f"   Average Daily Sales: ${avg_daily_sales:,.2f}")
        print(f"   Sales per Store per Day: ${avg_daily_sales/self.n_stores:,.2f}")
        
        print(f"\n✅ Dataset saved to: {filepath}")
        print("=" * 50)
        return data

class DataLoader:
    """Professional data loading and validation"""
    
    @staticmethod
    def load_data(filename='sales_data.csv'):
        """Load sales data with validation"""
        filepath = os.path.join(RAW_DATA_PATH, filename)
        
        if not os.path.exists(filepath):
            print("📊 No existing data found. Generating sample data...")
            generator = SalesDataGenerator()
            return generator.save_data(filename)
        
        print("📂 LOADING SALES DATA")
        print("=" * 30)
        
        # Load and validate data
        data = pd.read_csv(filepath)
        data['date'] = pd.to_datetime(data['date'])
        
        # Data validation
        DataLoader._validate_data(data)
        
        print(f"✅ Data loaded successfully!")
        print(f"📊 Records: {len(data):,}")
        print(f"📅 Date range: {data['date'].min().date()} to {data['date'].max().date()}")
        print(f"🏪 Stores: {sorted(data['store_id'].unique())}")
        print(f"💰 Total sales: ${data['sales'].sum():,.2f}")
        print("=" * 30)
        
        return data
    
    @staticmethod
    def _validate_data(data):
        """Validate data quality"""
        issues = []
        
        # Check for missing values
        missing_cols = data.isnull().sum()
        if missing_cols.any():
            issues.append(f"Missing values found: {missing_cols[missing_cols > 0].to_dict()}")
        
        # Check for negative sales
        if (data['sales'] < 0).any():
            issues.append("Negative sales values found")
        
        # Check date continuity
        date_gaps = data.groupby('store_id')['date'].apply(
            lambda x: (x.sort_values().diff() > pd.Timedelta(days=1)).any()
        )
        if date_gaps.any():
            issues.append("Date gaps found in time series")
        
        if issues:
            print("⚠️  DATA QUALITY ISSUES:")
            for issue in issues:
                print(f"   - {issue}")
        else:
            print("✅ Data validation passed!")
    
    @staticmethod
    def get_store_data(data, store_id):
        """Extract data for specific store"""
        store_data = data[data['store_id'] == store_id].copy()
        store_data = store_data.sort_values('date').reset_index(drop=True)
        
        print(f"🏪 Store {store_id} data: {len(store_data)} records")
        print(f"   Sales range: ${store_data['sales'].min():.2f} - ${store_data['sales'].max():.2f}")
        print(f"   Average daily sales: ${store_data['sales'].mean():.2f}")
        
        return store_data
    
    @staticmethod
    def get_summary_stats(data):
        """Get comprehensive data summary"""
        print("📊 COMPREHENSIVE DATA SUMMARY")
        print("=" * 40)
        
        # Overall statistics
        print("🔢 OVERALL STATISTICS:")
        print(f"   Total records: {len(data):,}")
        print(f"   Date range: {(data['date'].max() - data['date'].min()).days} days")
        print(f"   Stores: {data['store_id'].nunique()}")
        print(f"   Total sales: ${data['sales'].sum():,.2f}")
        
        # Sales statistics
        print(f"\n💰 SALES STATISTICS:")
        print(f"   Mean: ${data['sales'].mean():,.2f}")
        print(f"   Median: ${data['sales'].median():,.2f}")
        print(f"   Std Dev: ${data['sales'].std():,.2f}")
        print(f"   Min: ${data['sales'].min():,.2f}")
        print(f"   Max: ${data['sales'].max():,.2f}")
        
        # Store performance
        print(f"\n🏪 STORE PERFORMANCE:")
        store_stats = data.groupby('store_id')['sales'].agg(['mean', 'sum', 'count']).round(2)
        for store_id, stats in store_stats.iterrows():
            print(f"   Store {store_id}: Avg=${stats['mean']}, Total=${stats['sum']:,.0f}")
        
        # Promotion analysis
        promo_impact = data.groupby('promotion')['sales'].mean()
        promo_lift = ((promo_impact[1] - promo_impact[0]) / promo_impact[0]) * 100
        print(f"\n🎯 PROMOTION IMPACT:")
        print(f"   Regular sales: ${promo_impact[0]:.2f}")
        print(f"   Promotion sales: ${promo_impact[1]:.2f}")
        print(f"   Lift: {promo_lift:.1f}%")
        
        # Data quality
        print(f"\n✅ DATA QUALITY:")
        print(f"   Missing values: {data.isnull().sum().sum()}")
        print(f"   Duplicate rows: {data.duplicated().sum()}")
        print(f"   Data completeness: {((1 - data.isnull().sum().sum()/(len(data)*len(data.columns))) * 100):.1f}%")
        
        return store_stats

if __name__ == "__main__":
    # Test data generation and loading
    print("🧪 TESTING DATA LOADING MODULE")
    
    # Generate data
    generator = SalesDataGenerator()
    data = generator.save_data()
    
    # Load and validate data
    loader = DataLoader()
    loaded_data = loader.load_data()
    
    # Get summary
    loader.get_summary_stats(loaded_data)
    
    print("\n✅ Data loading module test completed!")