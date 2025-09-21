"""
🚀 STEP 1: DATA GENERATION
Generate realistic retail sales data for forecasting
"""

import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    """Generate sample retail sales data"""
    
    print("🛍️  RETAIL SALES FORECASTING - DATA GENERATION")
    print("=" * 60)
    
    try:
        # Import after adding to path
        from src.data_loader import SalesDataGenerator, DataLoader
        
        # Step 1: Generate data
        print("STEP 1: Generating sample data...")
        generator = SalesDataGenerator(
            start_date='2020-01-01',
            end_date='2023-12-31',
            n_stores=5
        )
        data = generator.save_data()
        
        # Step 2: Validate data
        print("\nSTEP 2: Validating generated data...")
        loader = DataLoader()
        loader.get_summary_stats(data)
        
        # Step 3: Show sample data
        print("\n📋 SAMPLE DATA:")
        print(data.head(10).to_string(index=False))
        
        print(f"\n🎉 DATA GENERATION COMPLETED!")
        print(f"📁 File saved: data/raw/sales_data.csv")
        print(f"📊 Ready for next step: Data Preprocessing")
        
        return data
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure to run 'python config.py' first to create directories")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    data = main()