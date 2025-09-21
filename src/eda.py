"""
🔍 EXPLORATORY DATA ANALYSIS MODULE
Comprehensive analysis and visualization of retail sales data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import sys

# Setup paths and imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

try:
    from config import RESULTS_PATH
    plots_dir = os.path.join(RESULTS_PATH, 'plots')
except ImportError:
    plots_dir = os.path.join(project_root, 'results', 'plots')

os.makedirs(plots_dir, exist_ok=True)

class SalesEDA:
    """Professional EDA for retail sales forecasting"""
    
    def __init__(self, data):
        self.data = data.copy()
        self.plots_dir = plots_dir
        
        # Set style for better plots
        plt.style.use('default')
        sns.set_palette("husl")
        
    def generate_business_summary(self):
        """Generate executive business summary"""
        print("💼 BUSINESS SUMMARY")
        print("=" * 30)
        
        # Key business metrics
        total_sales = self.data['sales'].sum()
        avg_daily_sales = self.data['sales'].mean()
        n_stores = self.data['store_id'].nunique()
        date_range = (self.data['date'].max() - self.data['date'].min()).days
        
        print(f"💰 Total Sales: ${total_sales:,.2f}")
        print(f"📈 Average Daily Sales: ${avg_daily_sales:,.2f}")
        print(f"🏪 Number of Stores: {n_stores}")
        print(f"📅 Analysis Period: {date_range} days")
        print(f"📊 Daily Sales per Store: ${avg_daily_sales/n_stores:,.2f}")
        
        # Growth analysis
        if 'year' in self.data.columns:
            yearly_sales = self.data.groupby('year')['sales'].sum()
            if len(yearly_sales) > 1:
                growth_rate = ((yearly_sales.iloc[-1] / yearly_sales.iloc[0]) ** (1/(len(yearly_sales)-1)) - 1) * 100
                print(f"📈 Annual Growth Rate: {growth_rate:.1f}%")
        
        # Store performance
        store_performance = self.data.groupby('store_id')['sales'].agg(['mean', 'sum', 'count'])
        best_store = store_performance['mean'].idxmax()
        worst_store = store_performance['mean'].idxmin()
        
        print(f"🏆 Best Performing Store: Store {best_store} (${store_performance.loc[best_store, 'mean']:,.2f}/day)")
        print(f"📉 Lowest Performing Store: Store {worst_store} (${store_performance.loc[worst_store, 'mean']:,.2f}/day)")
        
        # Promotion & holiday impact
        if 'promotion' in self.data.columns:
            promo_impact = self._calculate_promotion_impact()
            print(f"🎯 Promotion Sales Lift: +{promo_impact:.1f}%")
        
        if 'is_holiday' in self.data.columns:
            holiday_impact = self._calculate_holiday_impact()
            print(f"🎄 Holiday Sales Lift: +{holiday_impact:.1f}%")
        
        return {
            'total_sales': total_sales,
            'avg_daily_sales': avg_daily_sales,
            'best_store': best_store,
            'n_stores': n_stores
        }
    
    def _calculate_promotion_impact(self):
        """Calculate promotion impact percentage"""
        if 'promotion' not in self.data.columns:
            return 0
        regular_sales = self.data[self.data['promotion'] == 0]['sales'].mean()
        promo_sales = self.data[self.data['promotion'] == 1]['sales'].mean()
        return ((promo_sales - regular_sales) / regular_sales) * 100
    
    def _calculate_holiday_impact(self):
        """Calculate holiday impact percentage"""
        if 'is_holiday' not in self.data.columns:
            return 0
        regular_sales = self.data[self.data['is_holiday'] == False]['sales'].mean()
        holiday_sales = self.data[self.data['is_holiday'] == True]['sales'].mean()
        return ((holiday_sales - regular_sales) / regular_sales) * 100
    
    def plot_sales_trends(self, figsize=(16, 12)):
        """Comprehensive sales trend analysis"""
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle('📈 Sales Trend Analysis', fontsize=16, y=0.98)
        
        # 1. Overall daily sales trend
        daily_sales = self.data.groupby('date')['sales'].sum()
        axes[0, 0].plot(daily_sales.index, daily_sales.values, alpha=0.8, linewidth=1)
        axes[0, 0].set_title('Overall Daily Sales Trend')
        axes[0, 0].set_ylabel('Total Sales ($)')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Add trend line
        x_numeric = np.arange(len(daily_sales))
        z = np.polyfit(x_numeric, daily_sales.values, 1)
        p = np.poly1d(z)
        axes[0, 0].plot(daily_sales.index, p(x_numeric), "r--", alpha=0.8, linewidth=2, label=f'Trend')
        axes[0, 0].legend()
        
        # 2. Monthly sales pattern
        if 'month' in self.data.columns:
            monthly_sales = self.data.groupby('month')['sales'].mean()
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            axes[0, 1].bar(range(1, 13), monthly_sales.values, color='skyblue', alpha=0.8)
            axes[0, 1].set_title('Average Sales by Month')
            axes[0, 1].set_xlabel('Month')
            axes[0, 1].set_ylabel('Average Sales ($)')
            axes[0, 1].set_xticks(range(1, 13))
            axes[0, 1].set_xticklabels(month_names, rotation=45)
        
        # 3. Weekly pattern
        if 'day_of_week' in self.data.columns:
            weekly_sales = self.data.groupby('day_of_week')['sales'].mean()
            days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
            colors = ['lightcoral' if i >= 5 else 'lightblue' for i in range(7)]
            axes[0, 2].bar(range(7), weekly_sales.values, color=colors, alpha=0.8)
            axes[0, 2].set_title('Average Sales by Day of Week')
            axes[0, 2].set_xlabel('Day of Week')
            axes[0, 2].set_ylabel('Average Sales ($)')
            axes[0, 2].set_xticks(range(7))
            axes[0, 2].set_xticklabels(days)
        
        # 4. Store comparison
        store_sales = self.data.groupby('store_id')['sales'].agg(['mean', 'std'])
        axes[1, 0].bar(store_sales.index, store_sales['mean'], 
                       yerr=store_sales['std'], capsize=5, alpha=0.8)
        axes[1, 0].set_title('Average Sales by Store (with Std Dev)')
        axes[1, 0].set_xlabel('Store ID')
        axes[1, 0].set_ylabel('Average Sales ($)')
        
        # 5. Sales distribution
        axes[1, 1].hist(self.data['sales'], bins=50, alpha=0.7, color='green', edgecolor='black')
        axes[1, 1].set_title('Sales Distribution')
        axes[1, 1].set_xlabel('Sales ($)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].axvline(self.data['sales'].mean(), color='red', linestyle='--', 
                          label=f'Mean: ${self.data["sales"].mean():.0f}')
        axes[1, 1].legend()
        
        # 6. Year-over-Year comparison (if multi-year data)
        if 'year' in self.data.columns and self.data['year'].nunique() > 1:
            yearly_monthly = self.data.groupby(['year', 'month'])['sales'].sum().unstack(level=0)
            yearly_monthly.plot(kind='line', ax=axes[1, 2], marker='o')
            axes[1, 2].set_title('Monthly Sales by Year')
            axes[1, 2].set_xlabel('Month')
            axes[1, 2].set_ylabel('Total Sales ($)')
            axes[1, 2].legend(title='Year')
        else:
            # Quarterly analysis instead
            if 'quarter' in self.data.columns:
                quarterly_sales = self.data.groupby('quarter')['sales'].mean()
                axes[1, 2].bar(quarterly_sales.index, quarterly_sales.values, alpha=0.8)
                axes[1, 2].set_title('Average Sales by Quarter')
                axes[1, 2].set_xlabel('Quarter')
                axes[1, 2].set_ylabel('Average Sales ($)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'sales_trends.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_seasonality_analysis(self, figsize=(15, 10)):
        """Detailed seasonality analysis"""
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('🌟 Seasonality Analysis', fontsize=16, y=0.98)
        
        # 1. Daily pattern throughout the year
        if 'day_of_year' in self.data.columns or 'date' in self.data.columns:
            if 'day_of_year' not in self.data.columns:
                self.data['day_of_year'] = self.data['date'].dt.dayofyear
            
            daily_pattern = self.data.groupby('day_of_year')['sales'].mean()
            axes[0, 0].plot(daily_pattern.index, daily_pattern.values, alpha=0.8)
            axes[0, 0].set_title('Average Sales Throughout the Year')
            axes[0, 0].set_xlabel('Day of Year')
            axes[0, 0].set_ylabel('Average Sales ($)')
            
            # Add seasonal markers
            seasons = [(80, 'Spring'), (172, 'Summer'), (266, 'Fall'), (355, 'Winter')]
            for day, season in seasons:
                axes[0, 0].axvline(day, alpha=0.3, linestyle='--')
                axes[0, 0].text(day, axes[0, 0].get_ylim()[1]*0.9, season, rotation=90)
        
        # 2. Heatmap: Day of Week vs Month
        if all(col in self.data.columns for col in ['day_of_week', 'month']):
            heatmap_data = self.data.groupby(['day_of_week', 'month'])['sales'].mean().unstack()
            sns.heatmap(heatmap_data, annot=True, fmt='.0f', cmap='YlOrRd', ax=axes[0, 1])
            axes[0, 1].set_title('Sales Heatmap: Day of Week vs Month')
            axes[0, 1].set_xlabel('Month')
            axes[0, 1].set_ylabel('Day of Week')
            
            # Set y-axis labels
            axes[0, 1].set_yticklabels(['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
        
        # 3. Weekly pattern by store
        if all(col in self.data.columns for col in ['day_of_week', 'store_id']):
            weekly_store = self.data.groupby(['store_id', 'day_of_week'])['sales'].mean().unstack()
            weekly_store.plot(kind='line', ax=axes[1, 0], marker='o')
            axes[1, 0].set_title('Weekly Pattern by Store')
            axes[1, 0].set_xlabel('Store ID')
            axes[1, 0].set_ylabel('Average Sales ($)')
            axes[1, 0].legend(title='Day of Week', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 4. Seasonal decomposition visualization (simplified)
        if 'date' in self.data.columns:
            monthly_sales = self.data.groupby(self.data['date'].dt.to_period('M'))['sales'].sum()
            
            # Simple trend removal
            trend = monthly_sales.rolling(window=12, center=True).mean()
            detrended = monthly_sales - trend
            
            axes[1, 1].plot(monthly_sales.index.to_timestamp(), monthly_sales.values, 
                           label='Original', alpha=0.8)
            axes[1, 1].plot(trend.index.to_timestamp(), trend.values, 
                           label='Trend', alpha=0.8, linewidth=2)
            axes[1, 1].set_title('Monthly Sales with Trend')
            axes[1, 1].set_xlabel('Date')
            axes[1, 1].set_ylabel('Monthly Sales ($)')
            axes[1, 1].legend()
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'seasonality_analysis.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_promotion_holiday_analysis(self, figsize=(15, 10)):
        """Analyze promotion and holiday effects"""
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle('🎯 Promotion & Holiday Analysis', fontsize=16, y=0.98)
        
        # 1. Promotion impact
        if 'promotion' in self.data.columns:
            promo_comparison = self.data.groupby('promotion')['sales'].agg(['mean', 'std', 'count'])
            
            axes[0, 0].bar(['No Promotion', 'Promotion'], promo_comparison['mean'], 
                          yerr=promo_comparison['std'], capsize=5, alpha=0.8)
            axes[0, 0].set_title('Sales: Promotion vs No Promotion')
            axes[0, 0].set_ylabel('Average Sales ($)')
            
            # Add counts as text
            for i, (idx, row) in enumerate(promo_comparison.iterrows()):
                axes[0, 0].text(i, row['mean'] + row['std'] + 20, f"n={row['count']}", 
                               ha='center', fontsize=10)
        
        # 2. Holiday impact
        if 'is_holiday' in self.data.columns:
            holiday_comparison = self.data.groupby('is_holiday')['sales'].agg(['mean', 'std', 'count'])
            
            axes[0, 1].bar(['Regular Day', 'Holiday'], holiday_comparison['mean'],
                          yerr=holiday_comparison['std'], capsize=5, alpha=0.8, color=['lightblue', 'orange'])
            axes[0, 1].set_title('Sales: Holiday vs Regular Day')
            axes[0, 1].set_ylabel('Average Sales ($)')
            
            # Add counts as text
            for i, (idx, row) in enumerate(holiday_comparison.iterrows()):
                axes[0, 1].text(i, row['mean'] + row['std'] + 20, f"n={row['count']}", 
                               ha='center', fontsize=10)
        
        # 3. Combined effect (Promotion + Holiday)
        if all(col in self.data.columns for col in ['promotion', 'is_holiday']):
            combined_effect = self.data.groupby(['promotion', 'is_holiday'])['sales'].mean().unstack()
            combined_effect.plot(kind='bar', ax=axes[0, 2], alpha=0.8)
            axes[0, 2].set_title('Combined Effect: Promotion & Holiday')
            axes[0, 2].set_xlabel('Promotion (0=No, 1=Yes)')
            axes[0, 2].set_ylabel('Average Sales ($)')
            axes[0, 2].legend(title='Holiday', labels=['Regular', 'Holiday'])
            axes[0, 2].tick_params(axis='x', rotation=0)
        
        # 4. Sales distribution by promotion
        if 'promotion' in self.data.columns:
            self.data[self.data['promotion'] == 0]['sales'].hist(
                alpha=0.6, bins=30, label='No Promotion', ax=axes[1, 0], color='blue')
            self.data[self.data['promotion'] == 1]['sales'].hist(
                alpha=0.6, bins=30, label='Promotion', ax=axes[1, 0], color='red')
            axes[1, 0].set_title('Sales Distribution by Promotion')
            axes[1, 0].set_xlabel('Sales ($)')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].legend()
        
        # 5. Promotion frequency by store
        if all(col in self.data.columns for col in ['promotion', 'store_id']):
            promo_by_store = self.data.groupby('store_id')['promotion'].agg(['sum', 'count'])
            promo_by_store['promo_rate'] = promo_by_store['sum'] / promo_by_store['count'] * 100
            
            axes[1, 1].bar(promo_by_store.index, promo_by_store['promo_rate'], alpha=0.8)
            axes[1, 1].set_title('Promotion Rate by Store (%)')
            axes[1, 1].set_xlabel('Store ID')
            axes[1, 1].set_ylabel('Promotion Rate (%)')
        
        # 6. Weekend vs Weekday analysis
        if 'is_weekend' in self.data.columns:
            weekend_comparison = self.data.groupby('is_weekend')['sales'].agg(['mean', 'std'])
            
            axes[1, 2].bar(['Weekday', 'Weekend'], weekend_comparison['mean'],
                          yerr=weekend_comparison['std'], capsize=5, alpha=0.8, 
                          color=['lightgreen', 'coral'])
            axes[1, 2].set_title('Sales: Weekday vs Weekend')
            axes[1, 2].set_ylabel('Average Sales ($)')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'promotion_holiday_analysis.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_correlation_analysis(self, figsize=(12, 10)):
        """Comprehensive correlation analysis"""
        # Select numeric columns for correlation
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove date-related columns that might be numeric but not meaningful for correlation
        exclude_cols = ['store_id', 'year'] if 'year' in numeric_cols else ['store_id']
        numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        if len(numeric_cols) < 3:
            print("⚠️ Not enough numeric columns for correlation analysis")
            return
        
        # Calculate correlation matrix
        corr_matrix = self.data[numeric_cols].corr()
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle('🔗 Correlation Analysis', fontsize=16, y=0.98)
        
        # 1. Full correlation heatmap
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', 
                   cmap='coolwarm', center=0, ax=axes[0],
                   square=True, cbar_kws={"shrink": .8})
        axes[0].set_title('Feature Correlation Matrix')
        
        # 2. Sales correlation bar plot
        sales_corr = corr_matrix['sales'].drop('sales').abs().sort_values(ascending=True)
        
        # Color bars based on correlation strength
        colors = ['red' if abs(x) > 0.5 else 'orange' if abs(x) > 0.3 else 'lightblue' for x in sales_corr]
        
        axes[1].barh(range(len(sales_corr)), sales_corr.values, color=colors, alpha=0.8)
        axes[1].set_yticks(range(len(sales_corr)))
        axes[1].set_yticklabels(sales_corr.index, fontsize=10)
        axes[1].set_xlabel('Absolute Correlation with Sales')
        axes[1].set_title('Features Most Correlated with Sales')
        axes[1].axvline(x=0.3, color='orange', linestyle='--', alpha=0.7, label='Moderate (0.3)')
        axes[1].axvline(x=0.5, color='red', linestyle='--', alpha=0.7, label='Strong (0.5)')
        axes[1].legend()
        
        # Add correlation values as text
        for i, v in enumerate(sales_corr.values):
            axes[1].text(v + 0.01, i, f'{v:.3f}', va='center', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.plots_dir, 'correlation_analysis.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        return sales_corr
    
    def generate_forecasting_insights(self):
        """Generate insights specifically for forecasting"""
        print("\n🎯 FORECASTING READINESS ASSESSMENT")
        print("=" * 45)
        
        # Data quality assessment
        total_records = len(self.data)
        missing_pct = (self.data.isnull().sum().sum() / (total_records * len(self.data.columns))) * 100
        
        print(f"✅ Data Quality:")
        print(f"   Records: {total_records:,}")
        print(f"   Completeness: {100-missing_pct:.1f}%")
        
        # Time series properties
        if 'date' in self.data.columns:
            date_range = (self.data['date'].max() - self.data['date'].min()).days
            print(f"   Time span: {date_range} days")
            
            # Check for gaps in time series
            for store_id in self.data['store_id'].unique():
                store_data = self.data[self.data['store_id'] == store_id]['date'].sort_values()
                expected_days = (store_data.max() - store_data.min()).days + 1
                actual_days = len(store_data)
                completeness = (actual_days / expected_days) * 100
                if completeness < 95:
                    print(f"   ⚠️  Store {store_id}: {completeness:.1f}% date completeness")
        
        # Feature availability
        feature_cols = [col for col in self.data.columns if col not in ['date', 'store_id', 'sales']]
        lag_features = [col for col in feature_cols if 'lag' in col]
        rolling_features = [col for col in feature_cols if 'rolling' in col]
        
        print(f"\n✅ Feature Engineering:")
        print(f"   Total features: {len(feature_cols)}")
        print(f"   Lag features: {len(lag_features)}")
        print(f"   Rolling features: {len(rolling_features)}")
        
        # Seasonality detection
        if 'day_of_week' in self.data.columns:
            weekly_var = self.data.groupby('day_of_week')['sales'].mean().var()
            weekly_strength = weekly_var / self.data['sales'].var()
            print(f"   Weekly seasonality strength: {weekly_strength:.3f}")
        
        if 'month' in self.data.columns:
            monthly_var = self.data.groupby('month')['sales'].mean().var()
            monthly_strength = monthly_var / self.data['sales'].var()
            print(f"   Monthly seasonality strength: {monthly_strength:.3f}")
        
        # Model recommendations
        print(f"\n🤖 Recommended Models:")
        print(f"   🎯 Baseline: Naive, Moving Average")
        print(f"   📊 Statistical: ARIMA (if stationary), Prophet (for seasonality)")
        
        if len(lag_features) > 3:
            print(f"   🌳 ML: Random Forest, XGBoost (rich features)")
        
        if date_range > 365:
            print(f"   🧠 Deep Learning: LSTM (long sequences)")
        
        # Business insights
        print(f"\n💼 Business Insights:")
        if 'promotion' in self.data.columns:
            promo_impact = self._calculate_promotion_impact()
            print(f"   Promotion lift: {promo_impact:.1f}%")
        
        if 'is_weekend' in self.data.columns:
            weekend_sales = self.data[self.data['is_weekend'] == 1]['sales'].mean()
            weekday_sales = self.data[self.data['is_weekend'] == 0]['sales'].mean()
            weekend_lift = ((weekend_sales - weekday_sales) / weekday_sales) * 100
            print(f"   Weekend lift: {weekend_lift:.1f}%")
        
        return {
            'data_quality': 100-missing_pct,
            'time_span': date_range if 'date' in self.data.columns else 0,
            'feature_count': len(feature_cols),
            'lag_features': len(lag_features),
            'rolling_features': len(rolling_features)
        }
    
    def generate_comprehensive_report(self):
        """Generate complete EDA report with all analyses"""
        print("🚀 GENERATING COMPREHENSIVE EDA REPORT")
        print("=" * 60)
        
        # Business summary
        business_summary = self.generate_business_summary()
        
        # Visual analyses
        print("\n📊 Creating visualizations...")
        self.plot_sales_trends()
        self.plot_seasonality_analysis()
        self.plot_promotion_holiday_analysis()
        correlations = self.plot_correlation_analysis()
        
        # Forecasting insights
        forecasting_insights = self.generate_forecasting_insights()
        
        print(f"\n🎉 EDA REPORT COMPLETED!")
        print(f"📁 Plots saved to: {self.plots_dir}")
        print(f"📈 Generated visualizations:")
        print(f"   • sales_trends.png")
        print(f"   • seasonality_analysis.png")
        print(f"   • promotion_holiday_analysis.png")
        print(f"   • correlation_analysis.png")
        
        return {
            'business_summary': business_summary,
            'correlations': correlations,
            'forecasting_insights': forecasting_insights
        }

if __name__ == "__main__":
    # Test EDA module
    print("🧪 TESTING EDA MODULE")
    
    try:
        # This would normally load processed data
        from data_loader import DataLoader
        
        # Load data
        loader = DataLoader()
        data = loader.load_data()
        
        # Run EDA
        eda = SalesEDA(data)
        report = eda.generate_comprehensive_report()
        
        print("✅ EDA module test completed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()