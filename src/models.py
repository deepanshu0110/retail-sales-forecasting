"""
🤖 FORECASTING MODELS MODULE
Professional implementation of multiple forecasting approaches
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

import os
import sys

# Setup paths
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

try:
    from config import RESULTS_PATH, MODELS_PATH
    results_dir = RESULTS_PATH
    models_dir = MODELS_PATH
except ImportError:
    results_dir = os.path.join(project_root, 'results')
    models_dir = os.path.join(project_root, 'models')

os.makedirs(results_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

class ForecastingPipeline:
    """Professional forecasting pipeline with multiple models"""
    
    def __init__(self, data, target_col='sales', forecast_horizon=30):
        self.data = data.copy()
        self.target_col = target_col
        self.forecast_horizon = forecast_horizon
        self.models = {}
        self.forecasts = {}
        self.metrics = {}
        self.scaler = StandardScaler()
        
        print(f"🎯 Initialized Forecasting Pipeline")
        print(f"   Target: {target_col}")
        print(f"   Forecast Horizon: {forecast_horizon} days")
        
    def prepare_data(self, store_id=1, test_size=0.2):
        """Prepare data for modeling with proper validation"""
        print(f"🔧 PREPARING DATA FOR STORE {store_id}")
        print("=" * 40)
        
        # Filter and sort data
        store_data = self.data[self.data['store_id'] == store_id].copy()
        store_data = store_data.sort_values('date').reset_index(drop=True)
        
        if len(store_data) == 0:
            raise ValueError(f"No data found for store {store_id}")
        
        # Split data chronologically (important for time series)
        split_point = int(len(store_data) * (1 - test_size))
        
        if split_point < 50:  # Ensure minimum training data
            raise ValueError(f"Insufficient data: only {split_point} training samples")
        
        self.train_data = store_data[:split_point].copy()
        self.test_data = store_data[split_point:].copy()
        
        print(f"✅ Train data: {len(self.train_data)} records")
        print(f"   Date range: {self.train_data['date'].min().date()} to {self.train_data['date'].max().date()}")
        print(f"✅ Test data: {len(self.test_data)} records")
        print(f"   Date range: {self.test_data['date'].min().date()} to {self.test_data['date'].max().date()}")
        
        # Store reference values
        self.actual_values = self.test_data[self.target_col].values
        self.test_dates = self.test_data['date'].values
        
        return self.train_data, self.test_data
    
    def calculate_metrics(self, actual, predicted, model_name=""):
        """Calculate comprehensive forecasting metrics"""
        # Ensure arrays are the same length
        min_len = min(len(actual), len(predicted))
        actual = actual[:min_len]
        predicted = predicted[:min_len]
        
        # Basic metrics
        mae = mean_absolute_error(actual, predicted)
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        mape = np.mean(np.abs((actual - predicted) / np.maximum(actual, 1))) * 100  # Avoid division by zero
        
        # Additional metrics
        bias = np.mean(predicted - actual)
        
        # Business metrics
        inventory_cost_per_unit = 0.1  # Cost of holding excess inventory
        stockout_cost_per_unit = 2.0   # Cost of stockouts
        
        overforecast = np.sum(np.maximum(predicted - actual, 0))
        underforecast = np.sum(np.maximum(actual - predicted, 0))
        total_cost = overforecast * inventory_cost_per_unit + underforecast * stockout_cost_per_unit
        
        metrics = {
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
            'Bias': bias,
            'Total_Cost': total_cost,
            'Overforecast': overforecast,
            'Underforecast': underforecast
        }
        
        if model_name:
            print(f"📊 {model_name} Metrics:")
            print(f"   MAE: {mae:.2f}")
            print(f"   RMSE: {rmse:.2f}")
            print(f"   MAPE: {mape:.1f}%")
            print(f"   Total Cost: ${total_cost:.2f}")
        
        return metrics
    
    def baseline_models(self):
        """Implement baseline forecasting models"""
        print("\n🎯 TRAINING BASELINE MODELS")
        print("=" * 35)
        
        # 1. Naive Forecast (last observed value)
        naive_forecast = np.full(len(self.actual_values), self.train_data[self.target_col].iloc[-1])
        naive_metrics = self.calculate_metrics(self.actual_values, naive_forecast, "Naive")
        
        self.models['Naive'] = {
            'forecast': naive_forecast,
            'metrics': naive_metrics,
            'description': 'Last observed value'
        }
        
        # 2. Moving Average (multiple windows)
        windows = [3, 7, 14, 30]
        best_ma_rmse = float('inf')
        best_ma_window = 7
        
        for window in windows:
            if len(self.train_data) >= window:
                ma_value = self.train_data[self.target_col].tail(window).mean()
                ma_forecast = np.full(len(self.actual_values), ma_value)
                ma_metrics = self.calculate_metrics(self.actual_values, ma_forecast)
                
                if ma_metrics['RMSE'] < best_ma_rmse:
                    best_ma_rmse = ma_metrics['RMSE']
                    best_ma_window = window
                    best_ma_forecast = ma_forecast
                    best_ma_metrics = ma_metrics
        
        print(f"📊 Moving Average (Best: {best_ma_window}-day):")
        print(f"   MAE: {best_ma_metrics['MAE']:.2f}")
        print(f"   RMSE: {best_ma_metrics['RMSE']:.2f}")
        print(f"   MAPE: {best_ma_metrics['MAPE']:.1f}%")
        
        self.models['Moving_Average'] = {
            'forecast': best_ma_forecast,
            'metrics': best_ma_metrics,
            'description': f'{best_ma_window}-day moving average'
        }
        
        # 3. Seasonal Naive (same period last cycle)
        seasonal_naive = []
        seasonal_period = 7  # Weekly seasonality
        
        for i in range(len(self.actual_values)):
            if len(self.train_data) >= seasonal_period:
                # Use corresponding day from previous week
                historical_idx = len(self.train_data) - seasonal_period + (i % seasonal_period)
                if historical_idx >= 0 and historical_idx < len(self.train_data):
                    seasonal_naive.append(self.train_data[self.target_col].iloc[historical_idx])
                else:
                    seasonal_naive.append(self.train_data[self.target_col].iloc[-1])
            else:
                seasonal_naive.append(self.train_data[self.target_col].iloc[-1])
        
        seasonal_naive = np.array(seasonal_naive)
        seasonal_metrics = self.calculate_metrics(self.actual_values, seasonal_naive, "Seasonal Naive")
        
        self.models['Seasonal_Naive'] = {
            'forecast': seasonal_naive,
            'metrics': seasonal_metrics,
            'description': '7-day seasonal naive'
        }
        
        print("✅ Baseline models trained successfully!")
    
    def linear_trend_model(self):
        """Simple linear trend model"""
        print("\n📈 TRAINING LINEAR TREND MODEL")
        print("=" * 38)
        
        try:
            # Prepare features
            train_features = self.train_data.copy()
            test_features = self.test_data.copy()
            
            # Create time index
            train_features['time_idx'] = range(len(train_features))
            test_features['time_idx'] = range(len(train_features), len(train_features) + len(test_features))
            
            # Add basic features if available
            feature_cols = ['time_idx']
            
            # Add day of week if available
            if 'day_of_week' in train_features.columns:
                feature_cols.append('day_of_week')
            
            # Add month if available
            if 'month' in train_features.columns:
                feature_cols.append('month')
                
            # Add promotion if available
            if 'promotion' in train_features.columns:
                feature_cols.append('promotion')
            
            # Train model
            model = LinearRegression()
            X_train = train_features[feature_cols]
            y_train = train_features[self.target_col]
            X_test = test_features[feature_cols]
            
            model.fit(X_train, y_train)
            
            # Generate forecast
            trend_forecast = model.predict(X_test)
            trend_metrics = self.calculate_metrics(self.actual_values, trend_forecast, "Linear Trend")
            
            self.models['Linear_Trend'] = {
                'model': model,
                'forecast': trend_forecast,
                'metrics': trend_metrics,
                'features': feature_cols,
                'description': f'Linear regression with {len(feature_cols)} features'
            }
            
            print("✅ Linear trend model trained successfully!")
            
        except Exception as e:
            print(f"❌ Linear trend model failed: {str(e)}")
    
    def random_forest_model(self):
        """Random Forest model with feature engineering"""
        print("\n🌳 TRAINING RANDOM FOREST MODEL")
        print("=" * 40)
        
        try:
            # Prepare features (exclude non-predictive columns)
            exclude_cols = ['date', 'store_id', self.target_col]
            feature_cols = [col for col in self.train_data.columns if col not in exclude_cols]
            
            if len(feature_cols) == 0:
                print("❌ No features available for Random Forest")
                return
            
            print(f"   Using {len(feature_cols)} features")
            
            # Prepare data
            X_train = self.train_data[feature_cols].copy()
            y_train = self.train_data[self.target_col]
            X_test = self.test_data[feature_cols].copy()
            
            # Handle missing values
            for col in feature_cols:
                if X_train[col].isnull().sum() > 0:
                    fill_value = X_train[col].median() if X_train[col].dtype in ['float64', 'int64'] else X_train[col].mode().iloc[0]
                    X_train[col].fillna(fill_value, inplace=True)
                    X_test[col].fillna(fill_value, inplace=True)
            
            # Train Random Forest
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
            
            model.fit(X_train, y_train)
            
            # Generate forecast
            rf_forecast = model.predict(X_test)
            rf_metrics = self.calculate_metrics(self.actual_values, rf_forecast, "Random Forest")
            
            # Feature importance
            feature_importance = pd.DataFrame({
                'feature': feature_cols,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print(f"🔧 Top 5 Important Features:")
            for i, row in feature_importance.head().iterrows():
                print(f"   {row['feature']}: {row['importance']:.3f}")
            
            self.models['Random_Forest'] = {
                'model': model,
                'forecast': rf_forecast,
                'metrics': rf_metrics,
                'features': feature_cols,
                'feature_importance': feature_importance,
                'description': f'Random Forest with {len(feature_cols)} features'
            }
            
            print("✅ Random Forest model trained successfully!")
            
        except Exception as e:
            print(f"❌ Random Forest model failed: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def exponential_smoothing_model(self):
        """Simple exponential smoothing"""
        print("\n📊 TRAINING EXPONENTIAL SMOOTHING MODEL")
        print("=" * 45)
        
        try:
            # Simple exponential smoothing
            alpha = 0.3  # Smoothing parameter
            
            # Initialize with first value
            smoothed_values = [self.train_data[self.target_col].iloc[0]]
            
            # Calculate smoothed values for training data
            for i in range(1, len(self.train_data)):
                smoothed = alpha * self.train_data[self.target_col].iloc[i-1] + (1 - alpha) * smoothed_values[-1]
                smoothed_values.append(smoothed)
            
            # Forecast using last smoothed value
            es_forecast = np.full(len(self.actual_values), smoothed_values[-1])
            es_metrics = self.calculate_metrics(self.actual_values, es_forecast, "Exponential Smoothing")
            
            self.models['Exponential_Smoothing'] = {
                'forecast': es_forecast,
                'metrics': es_metrics,
                'alpha': alpha,
                'description': f'Exponential smoothing (α={alpha})'
            }
            
            print("✅ Exponential smoothing model trained successfully!")
            
        except Exception as e:
            print(f"❌ Exponential smoothing model failed: {str(e)}")
    
    def compare_models(self):
        """Compare all trained models"""
        print("\n🏆 MODEL PERFORMANCE COMPARISON")
        print("=" * 50)
        
        if not self.models:
            print("❌ No models trained yet!")
            return None
        
        # Create comparison dataframe
        comparison_data = []
        for model_name, model_info in self.models.items():
            metrics = model_info['metrics']
            comparison_data.append({
                'Model': model_name.replace('_', ' '),
                'MAE': metrics['MAE'],
                'RMSE': metrics['RMSE'],
                'MAPE': metrics['MAPE'],
                'Bias': metrics['Bias'],
                'Total_Cost': metrics['Total_Cost'],
                'Description': model_info.get('description', '')
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Sort by RMSE (lower is better)
        comparison_df = comparison_df.sort_values('RMSE')
        
        # Display results
        print(comparison_df[['Model', 'MAE', 'RMSE', 'MAPE', 'Total_Cost']].round(2).to_string(index=False))
        
        # Best model analysis
        best_model = comparison_df.iloc[0]
        print(f"\n🥇 BEST MODEL: {best_model['Model']}")
        print(f"   RMSE: {best_model['RMSE']:.2f}")
        print(f"   MAPE: {best_model['MAPE']:.1f}%")
        print(f"   Total Cost: ${best_model['Total_Cost']:.2f}")
        print(f"   Description: {best_model['Description']}")
        
        # Performance tiers
        print(f"\n📊 PERFORMANCE TIERS:")
        excellent_threshold = comparison_df['MAPE'].min() * 1.1
        good_threshold = comparison_df['MAPE'].min() * 1.3
        
        for _, row in comparison_df.iterrows():
            if row['MAPE'] <= excellent_threshold:
                tier = "🌟 Excellent"
            elif row['MAPE'] <= good_threshold:
                tier = "✅ Good"
            else:
                tier = "⚠️  Needs Improvement"
            print(f"   {row['Model']}: {tier} (MAPE: {row['MAPE']:.1f}%)")
        
        return comparison_df
    
    def plot_forecasts(self, figsize=(16, 12)):
        """Create comprehensive forecast visualization"""
        if not self.models:
            print("❌ No models to plot!")
            return
        
        n_models = len(self.models)
        n_cols = 2
        n_rows = (n_models + 1) // 2
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()
        
        fig.suptitle('📈 Model Forecasting Comparison', fontsize=16, y=0.98)
        
        # Plot each model
        plot_idx = 0
        for model_name, model_info in self.models.items():
            if plot_idx >= len(axes):
                break
                
            ax = axes[plot_idx]
            forecast = model_info['forecast']
            metrics = model_info['metrics']
            
            # Plot actual vs predicted
            ax.plot(range(len(self.actual_values)), self.actual_values, 
                   'o-', label='Actual', alpha=0.8, markersize=4)
            ax.plot(range(len(forecast)), forecast, 
                   's-', label='Forecast', alpha=0.8, markersize=4)
            
            # Add metrics to title
            title = f"{model_name.replace('_', ' ')}\n"
            title += f"RMSE: {metrics['RMSE']:.1f}, MAPE: {metrics['MAPE']:.1f}%"
            ax.set_title(title, fontsize=10)
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_xlabel('Test Period (Days)')
            ax.set_ylabel('Sales ($)')
            
            plot_idx += 1
        
        # Hide unused subplots
        for idx in range(plot_idx, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(results_dir, 'model_forecasts_comparison.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"📊 Forecast comparison plot saved: {plot_path}")
        plt.show()
        
    def plot_residual_analysis(self, figsize=(15, 10)):
        """Detailed residual analysis for all models"""
        if not self.models:
            print("❌ No models to analyze!")
            return
        
        n_models = len(self.models)
        fig, axes = plt.subplots(2, n_models, figsize=figsize)
        if n_models == 1:
            axes = axes.reshape(2, 1)
        
        fig.suptitle('🔍 Residual Analysis', fontsize=16, y=0.98)
        
        for idx, (model_name, model_info) in enumerate(self.models.items()):
            forecast = model_info['forecast']
            residuals = forecast - self.actual_values
            
            # Residuals over time
            axes[0, idx].plot(range(len(residuals)), residuals, 'o-', alpha=0.7, markersize=3)
            axes[0, idx].axhline(y=0, color='red', linestyle='--', alpha=0.8)
            axes[0, idx].set_title(f'{model_name.replace("_", " ")} Residuals')
            axes[0, idx].set_xlabel('Test Period')
            axes[0, idx].set_ylabel('Residuals')
            axes[0, idx].grid(True, alpha=0.3)
            
            # Residuals histogram
            axes[1, idx].hist(residuals, bins=20, alpha=0.7, edgecolor='black')
            axes[1, idx].axvline(x=0, color='red', linestyle='--', alpha=0.8)
            axes[1, idx].set_title('Residuals Distribution')
            axes[1, idx].set_xlabel('Residuals')
            axes[1, idx].set_ylabel('Frequency')
            
            # Add statistics
            mean_residual = np.mean(residuals)
            std_residual = np.std(residuals)
            axes[1, idx].text(0.05, 0.95, f'Mean: {mean_residual:.2f}\nStd: {std_residual:.2f}', 
                             transform=axes[1, idx].transAxes, verticalalignment='top',
                             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(results_dir, 'residual_analysis.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"📊 Residual analysis plot saved: {plot_path}")
        plt.show()
    
    def generate_forecast_report(self):
        """Generate comprehensive forecast report"""
        print("\n📄 GENERATING FORECAST REPORT")
        print("=" * 40)
        
        if not self.models:
            print("❌ No models available for report!")
            return
        
        # Create report
        report = {
            'timestamp': pd.Timestamp.now(),
            'data_summary': {
                'train_records': len(self.train_data),
                'test_records': len(self.test_data),
                'forecast_horizon': len(self.actual_values)
            },
            'model_comparison': self.compare_models(),
            'best_model': None,
            'recommendations': []
        }
        
        # Find best model
        if report['model_comparison'] is not None:
            best_model_name = report['model_comparison'].iloc[0]['Model']
            report['best_model'] = best_model_name
            
            # Generate recommendations
            best_mape = report['model_comparison'].iloc[0]['MAPE']
            
            if best_mape < 5:
                report['recommendations'].append("📊 Excellent forecast accuracy - ready for production")
            elif best_mape < 10:
                report['recommendations'].append("✅ Good forecast accuracy - suitable for business use")
            else:
                report['recommendations'].append("⚠️ Consider advanced models or more feature engineering")
            
            # Feature-based recommendations
            if 'Random_Forest' in self.models and self.models['Random_Forest']['metrics']['RMSE'] < best_mape * 10:
                report['recommendations'].append("🌳 Random Forest shows promise - consider ensemble methods")
            
            # Business recommendations
            report['recommendations'].append("💼 Monitor forecast performance weekly")
            report['recommendations'].append("🔄 Retrain models monthly with new data")
        
        return report
    
    def save_models(self):
        """Save trained models for deployment"""
        print("\n💾 SAVING MODELS")
        print("=" * 20)
        
        try:
            import pickle
            
            for model_name, model_info in self.models.items():
                if 'model' in model_info:  # Only save models with actual model objects
                    model_path = os.path.join(models_dir, f'{model_name.lower()}_model.pkl')
                    
                    with open(model_path, 'wb') as f:
                        pickle.dump({
                            'model': model_info['model'],
                            'features': model_info.get('features', []),
                            'metrics': model_info['metrics'],
                            'description': model_info['description']
                        }, f)
                    
                    print(f"✅ {model_name} model saved: {model_path}")
            
            # Save metadata
            metadata_path = os.path.join(models_dir, 'model_metadata.json')
            metadata = {
                'timestamp': pd.Timestamp.now().isoformat(),
                'models_trained': list(self.models.keys()),
                'target_column': self.target_col,
                'forecast_horizon': self.forecast_horizon
            }
            
            import json
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"✅ Model metadata saved: {metadata_path}")
            
        except Exception as e:
            print(f"❌ Error saving models: {str(e)}")
    
    def train_all_models(self, store_id=1):
        """Train all available models"""
        print("🚀 STARTING COMPREHENSIVE MODEL TRAINING")
        print("=" * 60)
        
        # Prepare data
        self.prepare_data(store_id=store_id)
        
        # Train all models
        print("\n🎯 Training Models...")
        self.baseline_models()
        self.linear_trend_model()
        self.random_forest_model()
        self.exponential_smoothing_model()
        
        # Compare and analyze
        print("\n📊 Analyzing Results...")
        comparison_df = self.compare_models()
        
        # Create visualizations
        print("\n📈 Creating Visualizations...")
        self.plot_forecasts()
        self.plot_residual_analysis()
        
        # Generate report
        print("\n📄 Generating Report...")
        report = self.generate_forecast_report()
        
        # Save models
        self.save_models()
        
        print("\n🎉 MODEL TRAINING COMPLETED!")
        print("=" * 35)
        print(f"✅ {len(self.models)} models trained successfully")
        print(f"📊 Best model: {report.get('best_model', 'Unknown')}")
        print(f"📁 Results saved to: {results_dir}")
        print(f"💾 Models saved to: {models_dir}")
        
        return comparison_df, report

if __name__ == "__main__":
    # Test the forecasting pipeline
    print("🧪 TESTING FORECASTING PIPELINE")
    
    try:
        # This would normally load processed data
        from data_loader import DataLoader
        
        # Load data
        loader = DataLoader()
        data = loader.load_data()
        
        # Initialize pipeline
        pipeline = ForecastingPipeline(data)
        
        # Train models
        comparison_df, report = pipeline.train_all_models()
        
        print("✅ Forecasting pipeline test completed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()