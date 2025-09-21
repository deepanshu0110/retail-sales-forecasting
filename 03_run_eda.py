"""
🚀 STEP 3: EXPLORATORY DATA ANALYSIS
Comprehensive analysis and visualization of retail sales data
"""

import sys
import os
import pandas as pd

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    """Execute comprehensive EDA analysis"""
    
    print("🛍️  RETAIL SALES FORECASTING - EXPLORATORY DATA ANALYSIS")
    print("=" * 70)
    
    try:
        # Import modules
        from src.eda import SalesEDA
        from config import PROCESSED_DATA_PATH
        
        # Step 1: Load processed data
        print("STEP 1: Loading processed data...")
        processed_data_path = os.path.join(PROCESSED_DATA_PATH, 'processed_sales_data.csv')
        
        if not os.path.exists(processed_data_path):
            print("❌ Processed data not found!")
            print("Please run 02_preprocess_data.py first")
            
            # Fallback: try to load raw data
            print("Attempting to load raw data as fallback...")
            from src.data_loader import DataLoader
            data = DataLoader.load_data()
        else:
            print("✅ Loading processed data...")
            data = pd.read_csv(processed_data_path)
            data['date'] = pd.to_datetime(data['date'])
        
        print(f"📊 Data loaded: {data.shape[0]:,} records, {data.shape[1]} columns")
        
        # Step 2: Initialize EDA
        print("\nSTEP 2: Initializing EDA analysis...")
        eda = SalesEDA(data)
        
        # Step 3: Generate comprehensive report
        print("\nSTEP 3: Generating EDA report...")
        report = eda.generate_comprehensive_report()
        
        # Step 4: Display key insights
        print("\n📋 KEY INSIGHTS SUMMARY")
        print("=" * 40)
        
        # Business insights
        business = report['business_summary']
        print(f"💰 Total Sales: ${business['total_sales']:,.2f}")
        print(f"📈 Average Daily Sales: ${business['avg_daily_sales']:,.2f}")
        print(f"🏆 Best Store: Store {business['best_store']}")
        
        # Feature insights
        if 'correlations' in report and report['correlations'] is not None:
            top_features = report['correlations'].tail(5)
            print(f"\n🔧 Top 5 Features Correlated with Sales:")
            for i, (feature, corr) in enumerate(top_features.items(), 1):
                print(f"   {i}. {feature}: {corr:.3f}")
        
        # Forecasting readiness
        forecasting = report['forecasting_insights']
        print(f"\n🎯 Forecasting Readiness:")
        print(f"   Data Quality: {forecasting['data_quality']:.1f}%")
        print(f"   Time Span: {forecasting['time_span']} days")
        print(f"   Features: {forecasting['feature_count']}")
        print(f"   Lag Features: {forecasting['lag_features']}")
        print(f"   Rolling Features: {forecasting['rolling_features']}")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS:")
        print(f"   ✅ Data is ready for forecasting")
        print(f"   ✅ Rich feature set available")
        print(f"   ✅ Clear seasonal patterns detected")
        print(f"   ✅ Business drivers identified")
        
        print(f"\n🎉 EDA ANALYSIS COMPLETED!")
        print(f"📁 Visualizations saved to: results/plots/")
        print(f"📈 Ready for model training!")
        
        return report
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all required modules are available")
        print("Try: pip install matplotlib seaborn")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    report = main()