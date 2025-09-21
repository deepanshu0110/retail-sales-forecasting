"""
🚀 STEP 4: MODEL TRAINING & EVALUATION
Train multiple forecasting models and compare their performance
"""

import sys
import os
import pandas as pd

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    """Execute complete model training pipeline"""
    
    print("🛍️  RETAIL SALES FORECASTING - MODEL TRAINING")
    print("=" * 65)
    
    try:
        # Import modules
        from src.models import ForecastingPipeline
        from config import PROCESSED_DATA_PATH
        
        # Step 1: Load processed data
        print("STEP 1: Loading processed data...")
        processed_data_path = os.path.join(PROCESSED_DATA_PATH, 'processed_sales_data.csv')
        
        if not os.path.exists(processed_data_path):
            print("❌ Processed data not found!")
            print("Falling back to raw data...")
            
            from src.data_loader import DataLoader
            data = DataLoader.load_data()
        else:
            print("✅ Loading processed data...")
            data = pd.read_csv(processed_data_path)
            data['date'] = pd.to_datetime(data['date'])
        
        print(f"📊 Data loaded: {data.shape[0]:,} records, {data.shape[1]} columns")
        
        # Step 2: Initialize forecasting pipeline
        print("\nSTEP 2: Initializing forecasting pipeline...")
        pipeline = ForecastingPipeline(
            data=data,
            target_col='sales',
            forecast_horizon=30
        )
        
        # Step 3: Train models for Store 1
        print("\nSTEP 3: Training models for Store 1...")
        comparison_df, report = pipeline.train_all_models(store_id=1)
        
        # Step 4: Display comprehensive results
        print("\n📋 TRAINING RESULTS SUMMARY")
        print("=" * 45)
        
        if comparison_df is not None:
            print("🏆 MODEL PERFORMANCE RANKING:")
            for i, (_, row) in enumerate(comparison_df.iterrows(), 1):
                rank_emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f"{i}."
                print(f"   {rank_emoji} {row['Model']}")
                print(f"      RMSE: {row['RMSE']:.2f} | MAPE: {row['MAPE']:.1f}% | Cost: ${row['Total_Cost']:.2f}")
        
        # Business insights
        if report and report['best_model']:
            print(f"\n💡 BUSINESS INSIGHTS:")
            print(f"   🎯 Best Model: {report['best_model']}")
            
            best_metrics = comparison_df.iloc[0]
            accuracy = 100 - best_metrics['MAPE']
            print(f"   📈 Forecast Accuracy: {accuracy:.1f}%")
            
            if best_metrics['MAPE'] < 5:
                print(f"   ✅ Excellent accuracy - Ready for production!")
            elif best_metrics['MAPE'] < 10:
                print(f"   ✅ Good accuracy - Suitable for business use")
            else:
                print(f"   ⚠️  Consider additional feature engineering")
        
        # Feature importance (if Random Forest performed well)
        if 'Random_Forest' in pipeline.models:
            rf_model = pipeline.models['Random_Forest']
            if 'feature_importance' in rf_model:
                print(f"\n🔧 KEY SALES DRIVERS (Top 5):")
                for i, (_, row) in enumerate(rf_model['feature_importance'].head().iterrows(), 1):
                    print(f"   {i}. {row['feature']}: {row['importance']:.3f}")
        
        # Recommendations
        if report and 'recommendations' in report:
            print(f"\n💼 RECOMMENDATIONS:")
            for rec in report['recommendations']:
                print(f"   {rec}")
        
        # Multi-store analysis (if time permits)
        print(f"\n🏪 MULTI-STORE ANALYSIS")
        print("=" * 30)
        
        store_results = {}
        for store_id in [2, 3]:  # Test a couple more stores
            try:
                print(f"\nAnalyzing Store {store_id}...")
                store_pipeline = ForecastingPipeline(data, target_col='sales')
                store_comparison, _ = store_pipeline.train_all_models(store_id=store_id)
                
                if store_comparison is not None:
                    best_model = store_comparison.iloc[0]
                    store_results[store_id] = {
                        'best_model': best_model['Model'],
                        'rmse': best_model['RMSE'],
                        'mape': best_model['MAPE']
                    }
                    print(f"   Best: {best_model['Model']} (MAPE: {best_model['MAPE']:.1f}%)")
                
            except Exception as e:
                print(f"   ⚠️  Store {store_id} analysis failed: {str(e)}")
        
        # Summary of multi-store results
        if store_results:
            print(f"\n🏆 MULTI-STORE SUMMARY:")
            all_stores = {1: comparison_df.iloc[0] if comparison_df is not None else None}
            all_stores.update(store_results)
            
            for store_id, results in all_stores.items():
                if results:
                    if isinstance(results, dict):
                        print(f"   Store {store_id}: {results['best_model']} (MAPE: {results['mape']:.1f}%)")
                    else:
                        print(f"   Store {store_id}: {results['Model']} (MAPE: {results['MAPE']:.1f}%)")
        
        print(f"\n🎉 MODEL TRAINING COMPLETED!")
        print("=" * 35)
        print(f"📁 Results saved to: results/")
        print(f"📊 Visualizations: model_forecasts_comparison.png, residual_analysis.png")
        print(f"💾 Models saved to: models/")
        print(f"📈 Ready for deployment!")
        
        # Next steps
        print(f"\n📋 NEXT STEPS:")
        print(f"   1. 📊 Review model performance visualizations")
        print(f"   2. 🔍 Analyze residual patterns for improvements")
        print(f"   3. 🚀 Deploy best model to production")
        print(f"   4. 📡 Set up automated retraining pipeline")
        print(f"   5. 📱 Create forecasting dashboard")
        
        return pipeline, comparison_df, report
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all required modules are available")
        print("Try installing: pip install scikit-learn matplotlib")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

if __name__ == "__main__":
    pipeline, comparison_df, report = main()