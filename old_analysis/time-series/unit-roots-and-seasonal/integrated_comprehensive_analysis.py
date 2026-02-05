#!/usr/bin/env python3
"""
Integrated Comprehensive Unit Root Testing and Seasonal Analysis for Time Series Variables
Combines robust data loading with advanced statistical testing and multi-frequency analysis
Tests for I(0), I(1), and I(2) integration orders with seasonal adjustments
"""

import pandas as pd
import numpy as np
import warnings
from datetime import datetime
import io
from contextlib import redirect_stdout
from typing import Dict, List, Tuple, Optional

# Import with comprehensive error handling for scipy/statsmodels issues
try:
    from statsmodels.tsa.stattools import adfuller, kpss
    from statsmodels.stats.diagnostic import acorr_ljungbox
    STATSMODELS_AVAILABLE = True
    print("✓ statsmodels successfully imported")
except ImportError as e:
    print(f"⚠ Warning: statsmodels import failed: {e}")
    STATSMODELS_AVAILABLE = False

    # Define robust dummy functions if statsmodels not available
    def adfuller(*args, **kwargs):
        return [0, 1, 0, 0, {}, None]
    def kpss(*args, **kwargs):
        return [0, 1, 0, {}]
    def acorr_ljungbox(*args, **kwargs):
        import pandas as pd
        return pd.DataFrame({'lb_pvalue': [1.0]})

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
    print("✓ scipy successfully imported")
except ImportError as e:
    print(f"⚠ Warning: scipy import failed: {e}")
    SCIPY_AVAILABLE = False

    # Define robust dummy stats module
    class DummyStats:
        @staticmethod
        def skew(x):
            return 0.0
        @staticmethod
        def kurtosis(x):
            return 0.0
    stats = DummyStats()

try:
    from tabulate import tabulate
    TABULATE_AVAILABLE = True
except ImportError:
    TABULATE_AVAILABLE = False
    def tabulate(data, headers=None, tablefmt='simple'):
        # Simple fallback table formatter
        if headers:
            result = '\t'.join(headers) + '\n'
            result += '-' * (len('\t'.join(headers))) + '\n'
        else:
            result = ''
        for row in data:
            result += '\t'.join(str(x) for x in row) + '\n'
        return result

warnings.filterwarnings('ignore')

class IntegratedTimeSeriesTester:
    """
    Integrated comprehensive time series analysis combining robust data handling
    with advanced statistical testing capabilities
    """

    def __init__(self, data_path='../data/'):
        self.data_path = data_path
        self.raw_data = {}
        self.half_hourly_data = None
        self.daily_data = None
        self.weekly_data = None
        self.seasonally_adjusted_data = {}
        self.seasonal_components = {}
        self.test_results = {}
        self.integration_orders = {}
        self.seasonal_analysis = {}
        self.time_features = None

    def load_data(self):
        """Load all available half-hourly data files with robust error handling"""
        print("="*100)
        print("DATA LOADING AND INITIALIZATION")
        print("="*100)
        print(f"Analysis started: {datetime.now()}")
        print(f"Data path: {self.data_path}")
        print(f"Statistical libraries available: statsmodels={STATSMODELS_AVAILABLE}, scipy={SCIPY_AVAILABLE}")

        # Try to load each data type with comprehensive error handling
        data_files = [
            ('purchases', 'half_hourly_purchases_2025-03-01_to_2025-09-30.parquet'),
            ('clicks', 'half_hourly_clicks_2025-03-01_to_2025-09-30.parquet'),
            ('impressions', 'half_hourly_impressions_2025-03-01_to_2025-09-30.parquet'),
            ('auctions', 'half_hourly_auctions_2025-03-01_to_2025-09-30.parquet'),
            ('auction_users', 'half_hourly_auction_users_2025-03-01_to_2025-09-30.parquet')
        ]

        loaded_count = 0
        for name, filename in data_files:
            try:
                full_path = f'{self.data_path}{filename}'
                df = pd.read_parquet(full_path)
                self.raw_data[name] = df
                loaded_count += 1
                print(f"  ✓ {name:15s}: {df.shape[0]:,} rows × {df.shape[1]:2d} cols")
                print(f"    File: {filename}")
                print(f"    Columns: {list(df.columns[:3])}{'...' if len(df.columns) > 3 else ''}")
                print(f"    Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

                # Basic data quality check
                missing_pct = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
                print(f"    Missing data: {missing_pct:.2f}%")

            except FileNotFoundError:
                print(f"  ✗ {name:15s}: file not found - {filename}")
            except Exception as e:
                print(f"  ✗ {name:15s}: error loading - {str(e)[:60]}")

        if not self.raw_data:
            raise ValueError("CRITICAL ERROR: No data files could be loaded")

        print(f"\nData loading summary:")
        print(f"  Successfully loaded: {loaded_count}/{len(data_files)} datasets")
        print(f"  Total raw data points: {sum(df.shape[0] * df.shape[1] for df in self.raw_data.values()):,}")

        return self.raw_data

    def prepare_half_hourly_data(self):
        """Prepare merged half-hourly data with comprehensive time features and validation"""
        print("\n" + "="*100)
        print("HALF-HOURLY DATA PREPARATION AND FEATURE ENGINEERING")
        print("="*100)

        if not self.raw_data:
            print("CRITICAL ERROR: No raw data available for preparation")
            return None, None

        # Find the time column with robust detection
        time_cols = ['ACTIVITY_HALF_HOUR', 'ACTIVITY_HOUR', 'timestamp', 'time']
        time_col = None

        print("Detecting time column:")
        for name, df in self.raw_data.items():
            for col in time_cols:
                if col in df.columns:
                    time_col = col
                    print(f"  Found time column '{col}' in dataset '{name}'")
                    break
            if time_col:
                break

        if not time_col:
            print("CRITICAL ERROR: No time column found in any dataset")
            available_cols = set()
            for df in self.raw_data.values():
                available_cols.update(df.columns)
            print(f"Available columns across all datasets: {sorted(available_cols)}")
            return None, None

        # Merge all datasets with detailed logging
        print(f"\nMerging datasets on time column: {time_col}")
        merged = None
        merge_log = []

        for name, df in self.raw_data.items():
            df_copy = df.copy()
            if time_col not in df_copy.columns:
                print(f"  Skipping {name}: time column '{time_col}' not found")
                continue

            # Convert time column to datetime
            df_copy[time_col] = pd.to_datetime(df_copy[time_col])

            if merged is None:
                merged = df_copy
                merge_log.append(f"  Starting with {name}: {df_copy.shape}")
                print(f"  Starting with {name}: {df_copy.shape}")
            else:
                before_shape = merged.shape
                merged = merged.merge(df_copy, on=time_col, how='outer', suffixes=('', f'_{name}'))
                after_shape = merged.shape
                merge_log.append(f"  Merged {name}: {before_shape} → {after_shape}")
                print(f"  Merged {name}: {before_shape} → {after_shape}")

        if merged is None:
            print("CRITICAL ERROR: No datasets could be merged")
            return None, None

        # Data preparation and cleaning
        print(f"\nData preparation and cleaning:")
        print(f"  Pre-cleaning shape: {merged.shape}")

        # Convert time column and set as index
        merged[time_col] = pd.to_datetime(merged[time_col])
        merged = merged.set_index(time_col)

        # Handle missing values with logging
        missing_before = merged.isnull().sum().sum()
        merged = merged.fillna(0)
        missing_after = merged.isnull().sum().sum()
        print(f"  Missing values filled: {missing_before:,} → {missing_after:,}")

        # Keep only numeric columns with detailed analysis
        all_cols = merged.columns.tolist()
        numeric_cols = merged.select_dtypes(include=[np.number]).columns.tolist()
        non_numeric_cols = [c for c in all_cols if c not in numeric_cols]

        print(f"  Total columns: {len(all_cols)}")
        print(f"  Numeric columns: {len(numeric_cols)}")
        print(f"  Non-numeric columns dropped: {len(non_numeric_cols)}")
        if non_numeric_cols:
            print(f"    Dropped: {non_numeric_cols[:5]}{'...' if len(non_numeric_cols) > 5 else ''}")

        self.half_hourly_data = merged[numeric_cols]

        # Create comprehensive time features for seasonal analysis
        print(f"\nCreating time features for seasonal analysis:")
        time_features = pd.DataFrame(index=self.half_hourly_data.index)

        # Basic time components
        time_features['year'] = self.half_hourly_data.index.year
        time_features['month'] = self.half_hourly_data.index.month
        time_features['day'] = self.half_hourly_data.index.day
        time_features['hour'] = self.half_hourly_data.index.hour
        time_features['minute'] = self.half_hourly_data.index.minute
        time_features['dayofweek'] = self.half_hourly_data.index.dayofweek
        time_features['dayofyear'] = self.half_hourly_data.index.dayofyear

        # Fractional time components for precise seasonal analysis
        time_features['hour_of_day'] = (self.half_hourly_data.index.hour +
                                       self.half_hourly_data.index.minute/60)
        time_features['day_of_week'] = self.half_hourly_data.index.dayofweek
        time_features['hour_of_week'] = (self.half_hourly_data.index.dayofweek * 24 +
                                        self.half_hourly_data.index.hour +
                                        self.half_hourly_data.index.minute/60)
        time_features['half_hour_of_week'] = (self.half_hourly_data.index.dayofweek * 48 +
                                             self.half_hourly_data.index.hour * 2 +
                                             self.half_hourly_data.index.minute // 30)

        # Additional seasonal indicators
        time_features['is_weekend'] = (self.half_hourly_data.index.dayofweek >= 5).astype(int)
        time_features['is_business_hours'] = ((self.half_hourly_data.index.hour >= 9) &
                                            (self.half_hourly_data.index.hour < 17) &
                                            (self.half_hourly_data.index.dayofweek < 5)).astype(int)

        self.time_features = time_features

        # Comprehensive data summary
        print(f"\nFinal half-hourly dataset summary:")
        print(f"  Shape: {self.half_hourly_data.shape[0]:,} observations × {self.half_hourly_data.shape[1]} variables")
        print(f"  Date range: {self.half_hourly_data.index.min()} to {self.half_hourly_data.index.max()}")
        print(f"  Time span: {(self.half_hourly_data.index.max() - self.half_hourly_data.index.min()).days} days")
        print(f"  Frequency: {pd.infer_freq(self.half_hourly_data.index) or 'Irregular'}")
        print(f"  Total data points: {self.half_hourly_data.shape[0] * self.half_hourly_data.shape[1]:,}")
        print(f"  Memory usage: {self.half_hourly_data.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

        # Data quality metrics
        zero_counts = (self.half_hourly_data == 0).sum()
        non_zero_vars = (zero_counts < len(self.half_hourly_data) * 0.95).sum()
        print(f"  Variables with >5% non-zero values: {non_zero_vars}/{len(self.half_hourly_data.columns)}")

        # Sample of variable names by category
        var_categories = {}
        for col in self.half_hourly_data.columns:
            col_upper = col.upper()
            if any(x in col_upper for x in ['COUNT', 'UNITS', 'LINES']):
                var_categories.setdefault('Counts', []).append(col)
            elif any(x in col_upper for x in ['GMV', 'REVENUE', 'PRICE']):
                var_categories.setdefault('Financial', []).append(col)
            elif any(x in col_upper for x in ['USER', 'VENDOR', 'CAMPAIGN']):
                var_categories.setdefault('Entities', []).append(col)
            elif any(x in col_upper for x in ['AVG', 'MEAN', 'MEDIAN']):
                var_categories.setdefault('Averages', []).append(col)
            else:
                var_categories.setdefault('Other', []).append(col)

        print(f"\nVariable categories:")
        for category, vars in var_categories.items():
            print(f"  {category}: {len(vars)} variables")
            if vars:
                print(f"    Examples: {vars[:2]}")

        return self.half_hourly_data, time_features

    def comprehensive_seasonal_analysis(self, series, name, time_features):
        """Comprehensive seasonal analysis with detailed pattern detection"""
        print(f"\n{'='*80}")
        print(f"SEASONAL ANALYSIS: {name}")
        print(f"{'='*80}")

        analysis = {
            'variable': name,
            'length': len(series),
            'missing': series.isnull().sum(),
            'mean': series.mean(),
            'std': series.std(),
            'min': series.min(),
            'max': series.max(),
            'sum': series.sum(),
            'non_zero_count': (series != 0).sum(),
            'non_zero_pct': ((series != 0).sum() / len(series)) * 100
        }

        print(f"Basic statistics:")
        print(f"  Length: {analysis['length']:,} observations")
        print(f"  Missing: {analysis['missing']:,} ({analysis['missing']/analysis['length']*100:.2f}%)")
        print(f"  Mean: {analysis['mean']:.6f}")
        print(f"  Std: {analysis['std']:.6f}")
        print(f"  Range: [{analysis['min']:.6f}, {analysis['max']:.6f}]")
        print(f"  Sum: {analysis['sum']:.2f}")
        print(f"  Non-zero: {analysis['non_zero_count']:,} ({analysis['non_zero_pct']:.1f}%)")

        if len(series.dropna()) < 100:
            print(f"  WARNING: Insufficient data for seasonal analysis ({len(series.dropna())} clean obs)")
            return analysis

        series_clean = series.dropna()
        print(f"  Clean observations for analysis: {len(series_clean):,}")

        # Hour of day pattern analysis
        try:
            print(f"\nHour-of-day patterns:")
            hourly_pattern = series_clean.groupby(series_clean.index.hour).agg(['mean', 'std', 'count'])
            hourly_means = hourly_pattern['mean'] if isinstance(hourly_pattern.columns, pd.MultiIndex) else hourly_pattern

            if len(hourly_means) > 1 and hourly_means.mean() > 0:
                analysis['hourly_variation'] = hourly_means.std() / hourly_means.mean()
                analysis['peak_hour'] = hourly_means.idxmax()
                analysis['trough_hour'] = hourly_means.idxmin()
                analysis['hour_pattern_strength'] = (hourly_means.max() - hourly_means.min()) / hourly_means.mean()

                print(f"  Hourly coefficient of variation: {analysis['hourly_variation']:.4f}")
                print(f"  Peak hour: {analysis['peak_hour']:02d}:00 (value: {hourly_means.iloc[analysis['peak_hour']]:.4f})")
                print(f"  Trough hour: {analysis['trough_hour']:02d}:00 (value: {hourly_means.iloc[analysis['trough_hour']]:.4f})")
                print(f"  Pattern strength: {analysis['hour_pattern_strength']:.4f}")

                # Detailed hourly breakdown
                print(f"  Hourly means by period:")
                print(f"    Night (00-05): {hourly_means.iloc[0:6].mean():.4f}")
                print(f"    Morning (06-11): {hourly_means.iloc[6:12].mean():.4f}")
                print(f"    Afternoon (12-17): {hourly_means.iloc[12:18].mean():.4f}")
                print(f"    Evening (18-23): {hourly_means.iloc[18:24].mean():.4f}")
            else:
                analysis['hourly_variation'] = 0
                print(f"  No significant hourly pattern detected")
        except Exception as e:
            analysis['hourly_variation'] = 0
            print(f"  Error in hourly analysis: {str(e)}")

        # Day of week pattern analysis
        try:
            print(f"\nDay-of-week patterns:")
            daily_pattern = series_clean.groupby(series_clean.index.dayofweek).agg(['mean', 'std', 'count'])
            daily_means = daily_pattern['mean'] if isinstance(daily_pattern.columns, pd.MultiIndex) else daily_pattern

            if len(daily_means) > 1 and daily_means.mean() > 0:
                analysis['daily_variation'] = daily_means.std() / daily_means.mean()
                analysis['peak_day'] = daily_means.idxmax()
                analysis['trough_day'] = daily_means.idxmin()
                analysis['weekday_weekend_ratio'] = daily_means.iloc[0:5].mean() / daily_means.iloc[5:7].mean() if daily_means.iloc[5:7].mean() > 0 else np.inf

                day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
                print(f"  Daily coefficient of variation: {analysis['daily_variation']:.4f}")
                print(f"  Peak day: {day_names[analysis['peak_day']]} (value: {daily_means.iloc[analysis['peak_day']]:.4f})")
                print(f"  Trough day: {day_names[analysis['trough_day']]} (value: {daily_means.iloc[analysis['trough_day']]:.4f})")
                print(f"  Weekday/Weekend ratio: {analysis['weekday_weekend_ratio']:.4f}")

                # Detailed daily breakdown
                print(f"  Daily means:")
                for i, day_name in enumerate(day_names):
                    if i < len(daily_means):
                        print(f"    {day_name}: {daily_means.iloc[i]:.4f}")
            else:
                analysis['daily_variation'] = 0
                print(f"  No significant daily pattern detected")
        except Exception as e:
            analysis['daily_variation'] = 0
            print(f"  Error in daily analysis: {str(e)}")

        # Hour of week pattern (336 half-hour periods)
        try:
            print(f"\nHour-of-week patterns (336 half-hour periods):")
            if 'half_hour_of_week' in time_features.columns:
                time_index = time_features.loc[series_clean.index, 'half_hour_of_week']
                weekly_pattern = series_clean.groupby(time_index).agg(['mean', 'std', 'count'])
                weekly_means = weekly_pattern['mean'] if isinstance(weekly_pattern.columns, pd.MultiIndex) else weekly_pattern

                if len(weekly_means) > 10 and weekly_means.mean() > 0:
                    analysis['weekly_variation'] = weekly_means.std() / weekly_means.mean()
                    analysis['peak_half_hour_week'] = weekly_means.idxmax()
                    analysis['trough_half_hour_week'] = weekly_means.idxmin()
                    analysis['weekly_pattern_coverage'] = len(weekly_means) / 336  # How many of 336 periods have data

                    print(f"  Weekly coefficient of variation: {analysis['weekly_variation']:.4f}")
                    print(f"  Peak half-hour of week: {analysis['peak_half_hour_week']} (value: {weekly_means.loc[analysis['peak_half_hour_week']]:.4f})")
                    print(f"  Trough half-hour of week: {analysis['trough_half_hour_week']} (value: {weekly_means.loc[analysis['trough_half_hour_week']]:.4f})")
                    print(f"  Pattern coverage: {analysis['weekly_pattern_coverage']:.1%} of weekly periods")

                    # Convert peak/trough back to readable format
                    peak_day = analysis['peak_half_hour_week'] // 48
                    peak_hour = (analysis['peak_half_hour_week'] % 48) // 2
                    peak_minute = ((analysis['peak_half_hour_week'] % 48) % 2) * 30
                    day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                    print(f"  Peak occurs: {day_names[peak_day]} {peak_hour:02d}:{peak_minute:02d}")
                else:
                    analysis['weekly_variation'] = 0
                    print(f"  Insufficient data for weekly pattern analysis")
            else:
                analysis['weekly_variation'] = 0
                print(f"  No weekly time features available")
        except Exception as e:
            analysis['weekly_variation'] = 0
            print(f"  Error in weekly analysis: {str(e)}")

        # Autocorrelation analysis with multiple lags
        try:
            print(f"\nAutocorrelation analysis:")
            lags_to_test = [1, 2, 48, 96, 336, 672]  # 30min, 1h, 1day, 2days, 1week, 2weeks
            lag_names = ['30min', '1hour', '1day', '2days', '1week', '2weeks']
            analysis['autocorrelations'] = {}

            for lag, lag_name in zip(lags_to_test, lag_names):
                if len(series_clean) > lag + 10:
                    try:
                        if STATSMODELS_AVAILABLE:
                            # Use pandas autocorr which is more robust
                            corr = series_clean.autocorr(lag=lag)
                        else:
                            # Manual calculation as fallback
                            s1 = series_clean.iloc[:-lag]
                            s2 = series_clean.iloc[lag:]
                            corr = np.corrcoef(s1, s2)[0, 1] if len(s1) > 1 else 0

                        analysis['autocorrelations'][f'lag_{lag}'] = corr
                        significance = "***" if abs(corr) > 0.2 else "**" if abs(corr) > 0.1 else "*" if abs(corr) > 0.05 else ""
                        print(f"  Lag {lag:3d} ({lag_name:6s}): {corr:8.4f} {significance}")
                    except:
                        analysis['autocorrelations'][f'lag_{lag}'] = 0
                        print(f"  Lag {lag:3d} ({lag_name:6s}): calculation failed")

            # Ljung-Box test for serial correlation if available
            if STATSMODELS_AVAILABLE and len(series_clean) > 48:
                try:
                    lb_test = acorr_ljungbox(series_clean, lags=min(48, len(series_clean)//4), return_df=True)
                    analysis['ljung_box_pvalue'] = lb_test['lb_pvalue'].iloc[-1]
                    analysis['ljung_box_statistic'] = lb_test['lb_stat'].iloc[-1]
                    print(f"  Ljung-Box test:")
                    print(f"    Statistic: {analysis['ljung_box_statistic']:.4f}")
                    print(f"    P-value: {analysis['ljung_box_pvalue']:.6f}")
                    print(f"    Conclusion: {'Serial correlation detected' if analysis['ljung_box_pvalue'] < 0.05 else 'No significant serial correlation'}")
                except Exception as e:
                    print(f"  Ljung-Box test failed: {str(e)}")
        except Exception as e:
            print(f"  Autocorrelation analysis failed: {str(e)}")
            analysis['autocorrelations'] = {}

        # Overall seasonality assessment
        seasonal_strength = 0
        if 'hourly_variation' in analysis and analysis['hourly_variation'] > 0:
            seasonal_strength += analysis['hourly_variation']
        if 'daily_variation' in analysis and analysis['daily_variation'] > 0:
            seasonal_strength += analysis['daily_variation']
        if 'weekly_variation' in analysis and analysis['weekly_variation'] > 0:
            seasonal_strength += analysis['weekly_variation']

        analysis['overall_seasonal_strength'] = seasonal_strength

        if seasonal_strength > 0.5:
            seasonality_level = "Strong"
        elif seasonal_strength > 0.2:
            seasonality_level = "Moderate"
        elif seasonal_strength > 0.05:
            seasonality_level = "Weak"
        else:
            seasonality_level = "None"

        analysis['seasonality_assessment'] = seasonality_level
        print(f"\nOverall seasonality assessment:")
        print(f"  Combined seasonal strength: {seasonal_strength:.4f}")
        print(f"  Assessment: {seasonality_level} seasonality")

        return analysis

    def create_seasonal_adjustments(self):
        """Create seasonally adjusted series using comprehensive seasonal patterns"""
        print("\n" + "="*100)
        print("SEASONAL ADJUSTMENT AND DESEASONALIZATION")
        print("="*100)

        if self.half_hourly_data is None or self.time_features is None:
            print("ERROR: Half-hourly data or time features not available")
            return None

        adjusted_data = {}
        seasonal_components = {}
        adjustment_log = []

        print(f"Creating seasonal adjustments for {len(self.half_hourly_data.columns)} variables...")

        for i, col in enumerate(self.half_hourly_data.columns, 1):
            print(f"\n{'-'*80}")
            print(f"SEASONAL ADJUSTMENT {i}/{len(self.half_hourly_data.columns)}: {col}")
            print(f"{'-'*80}")

            series = self.half_hourly_data[col]
            series_clean = series.dropna()

            # Skip if insufficient data or variation
            if len(series_clean) < 336:  # Less than 1 week of half-hourly data
                print(f"  Skipping: insufficient data ({len(series_clean)} observations)")
                adjustment_log.append(f"{col}: Skipped - insufficient data")
                continue

            if series_clean.std() == 0:
                print(f"  Skipping: no variation in data")
                adjustment_log.append(f"{col}: Skipped - no variation")
                continue

            try:
                # Use half-hour of week pattern (0-335) for seasonal adjustment
                if 'half_hour_of_week' in self.time_features.columns:
                    print(f"  Using half-hour-of-week seasonal pattern...")

                    # Get time index for clean series
                    time_index = self.time_features.loc[series_clean.index, 'half_hour_of_week']

                    # Calculate seasonal means for each half-hour period
                    seasonal_means = series_clean.groupby(time_index).mean()
                    seasonal_counts = series_clean.groupby(time_index).count()
                    overall_mean = series_clean.mean()

                    print(f"    Overall mean: {overall_mean:.6f}")
                    print(f"    Seasonal periods with data: {len(seasonal_means)}/336")
                    print(f"    Min observations per period: {seasonal_counts.min()}")
                    print(f"    Max observations per period: {seasonal_counts.max()}")

                    # Create seasonal factors
                    seasonal_factors = seasonal_means / overall_mean

                    # Handle missing periods and extreme values
                    seasonal_factors = seasonal_factors.fillna(1.0)
                    # Cap extreme seasonal factors to prevent over-adjustment
                    seasonal_factors = seasonal_factors.clip(0.1, 10.0)

                    print(f"    Seasonal factor range: [{seasonal_factors.min():.3f}, {seasonal_factors.max():.3f}]")
                    print(f"    Seasonal factor std: {seasonal_factors.std():.3f}")

                    # Map seasonal factors back to full series
                    time_index_full = self.time_features.loc[series.index, 'half_hour_of_week']
                    series_seasonal_factors = time_index_full.map(seasonal_factors).fillna(1.0)

                    # Apply seasonal adjustment
                    adjusted_series = series / series_seasonal_factors

                    # Create seasonal component
                    seasonal_component = series_seasonal_factors * overall_mean

                    # Store results
                    adjusted_data[f"{col}_sa"] = adjusted_series
                    seasonal_components[f"{col}_seasonal"] = seasonal_component

                    # Quality assessment
                    original_std = series.std()
                    adjusted_std = adjusted_series.std()
                    adjustment_ratio = adjusted_std / original_std if original_std > 0 else 1

                    print(f"    ✓ Seasonal adjustment completed")
                    print(f"    Original std: {original_std:.6f}")
                    print(f"    Adjusted std: {adjusted_std:.6f}")
                    print(f"    Adjustment ratio: {adjustment_ratio:.3f}")

                    if adjustment_ratio < 0.5:
                        print(f"    Strong seasonal adjustment applied")
                    elif adjustment_ratio < 0.8:
                        print(f"    Moderate seasonal adjustment applied")
                    else:
                        print(f"    Weak seasonal adjustment applied")

                    adjustment_log.append(f"{col}: Success - adjustment ratio {adjustment_ratio:.3f}")

                else:
                    print(f"  ERROR: No half_hour_of_week feature available")
                    adjustment_log.append(f"{col}: Failed - no time features")

            except Exception as e:
                print(f"  ERROR: Seasonal adjustment failed - {str(e)}")
                adjustment_log.append(f"{col}: Failed - {str(e)[:50]}")

        # Combine adjusted data
        if adjusted_data:
            self.seasonally_adjusted_data = pd.DataFrame(adjusted_data, index=self.half_hourly_data.index)
            self.seasonal_components = pd.DataFrame(seasonal_components, index=self.half_hourly_data.index)

            adjusted_count = len([k for k in adjusted_data.keys() if k.endswith('_sa')])
            print(f"\nSeasonal adjustment summary:")
            print(f"  Successfully adjusted: {adjusted_count}/{len(self.half_hourly_data.columns)} variables")
            print(f"  Total adjusted series created: {len(adjusted_data)}")
            print(f"  Seasonal components created: {len(seasonal_components)}")

            print(f"\nAdjustment log:")
            for log_entry in adjustment_log:
                print(f"    {log_entry}")
        else:
            print(f"\nWARNING: No seasonal adjustments could be created")

        return self.seasonally_adjusted_data

    def prepare_daily_data(self):
        """Aggregate to daily frequency with intelligent aggregation rules"""
        print("\n" + "="*100)
        print("DAILY DATA AGGREGATION")
        print("="*100)

        if self.half_hourly_data is None:
            print("ERROR: No half-hourly data available for daily aggregation")
            return None

        print(f"Aggregating {self.half_hourly_data.shape[1]} variables from half-hourly to daily frequency...")

        data = self.half_hourly_data.copy()
        data.reset_index(inplace=True)
        data['date'] = data.iloc[:, 0].dt.date  # First column is the time index

        # Intelligent aggregation rules based on variable semantics
        agg_rules = {}
        rule_explanations = {}

        for col in self.half_hourly_data.columns:
            col_upper = col.upper()

            if any(x in col_upper for x in ['COUNT', 'COUNTS', 'LINES', 'BIDS', 'CLICKS', 'IMPRESSIONS', 'PURCHASES']):
                agg_rules[col] = 'sum'
                rule_explanations[col] = 'Sum (count variable)'
            elif any(x in col_upper for x in ['GMV', 'REVENUE', 'VALUE', 'AMOUNT', 'UNITS']):
                agg_rules[col] = 'sum'
                rule_explanations[col] = 'Sum (monetary/quantity variable)'
            elif any(x in col_upper for x in ['USERS', 'USER']):
                agg_rules[col] = 'sum'
                rule_explanations[col] = 'Sum (user counts across periods)'
            elif any(x in col_upper for x in ['VENDORS', 'CAMPAIGNS', 'PRODUCTS', 'AUCTIONS', 'PAIRS']):
                agg_rules[col] = 'max'
                rule_explanations[col] = 'Max (diversity/cardinality metric)'
            elif any(x in col_upper for x in ['AVG', 'MEAN', 'AVERAGE']):
                agg_rules[col] = 'mean'
                rule_explanations[col] = 'Mean (average variable)'
            elif any(x in col_upper for x in ['MIN', 'MINIMUM']):
                agg_rules[col] = 'min'
                rule_explanations[col] = 'Min (minimum variable)'
            elif any(x in col_upper for x in ['MAX', 'MAXIMUM']):
                agg_rules[col] = 'max'
                rule_explanations[col] = 'Max (maximum variable)'
            elif any(x in col_upper for x in ['STDDEV', 'STD', 'DEVIATION']):
                agg_rules[col] = 'mean'
                rule_explanations[col] = 'Mean (standard deviation variable)'
            elif any(x in col_upper for x in ['RATE', 'RATIO', 'PCT', 'PERCENT']):
                agg_rules[col] = 'mean'
                rule_explanations[col] = 'Mean (rate/ratio variable)'
            else:
                agg_rules[col] = 'sum'
                rule_explanations[col] = 'Sum (default)'

        # Apply aggregation
        print(f"\nAggregation rules applied:")
        rule_counts = {}
        for rule in agg_rules.values():
            rule_counts[rule] = rule_counts.get(rule, 0) + 1

        for rule, count in sorted(rule_counts.items()):
            print(f"  {rule}: {count} variables")

        # Show sample of rules
        print(f"\nSample aggregation assignments:")
        sample_rules = list(rule_explanations.items())[:10]
        for var, explanation in sample_rules:
            print(f"  {var[:30]:30s} → {explanation}")
        if len(rule_explanations) > 10:
            print(f"  ... and {len(rule_explanations) - 10} more")

        # Perform aggregation
        self.daily_data = data.groupby('date').agg(agg_rules).reset_index()
        self.daily_data['date'] = pd.to_datetime(self.daily_data['date'])
        self.daily_data.set_index('date', inplace=True)

        print(f"\nDaily aggregation results:")
        print(f"  Shape: {self.daily_data.shape[0]} days × {self.daily_data.shape[1]} variables")
        print(f"  Date range: {self.daily_data.index.min()} to {self.daily_data.index.max()}")
        print(f"  Days of data: {len(self.daily_data)}")
        print(f"  Expected days (range): {(self.daily_data.index.max() - self.daily_data.index.min()).days + 1}")

        # Data quality check
        missing_days = pd.date_range(start=self.daily_data.index.min(),
                                    end=self.daily_data.index.max(),
                                    freq='D').difference(self.daily_data.index)
        if len(missing_days) > 0:
            print(f"  Missing days: {len(missing_days)}")
            if len(missing_days) <= 5:
                print(f"    Missing dates: {missing_days.tolist()}")

        return self.daily_data

    def prepare_weekly_data(self):
        """Aggregate to weekly frequency"""
        print("\n" + "="*100)
        print("WEEKLY DATA AGGREGATION")
        print("="*100)

        if self.daily_data is None:
            print("Daily data not available, creating it first...")
            self.prepare_daily_data()

        if self.daily_data is None:
            print("ERROR: Cannot create weekly data without daily data")
            return None

        print(f"Aggregating {self.daily_data.shape[1]} variables from daily to weekly frequency...")

        daily_with_week = self.daily_data.copy()
        daily_with_week['week'] = pd.to_datetime(daily_with_week.index).to_period('W-SUN')  # Week ending on Sunday

        # Weekly aggregation rules
        agg_rules = {}
        for col in self.daily_data.columns:
            col_upper = col.upper()
            if any(x in col_upper for x in ['VENDORS', 'CAMPAIGNS', 'PRODUCTS', 'AUCTIONS', 'PAIRS']):
                agg_rules[col] = 'max'  # Max for diversity metrics
            elif any(x in col_upper for x in ['AVG', 'MEAN', 'RATE', 'RATIO', 'PCT']):
                agg_rules[col] = 'mean'  # Mean for rates and averages
            else:
                agg_rules[col] = 'sum'  # Sum for counts and values

        self.weekly_data = daily_with_week.groupby('week').agg(agg_rules)
        self.weekly_data.index = self.weekly_data.index.to_timestamp()

        print(f"\nWeekly aggregation results:")
        print(f"  Shape: {self.weekly_data.shape[0]} weeks × {self.weekly_data.shape[1]} variables")
        print(f"  Date range: {self.weekly_data.index.min()} to {self.weekly_data.index.max()}")
        print(f"  Weeks of data: {len(self.weekly_data)}")

        # Weekly aggregation rules summary
        weekly_rule_counts = {}
        for rule in agg_rules.values():
            weekly_rule_counts[rule] = weekly_rule_counts.get(rule, 0) + 1

        print(f"  Weekly aggregation rules:")
        for rule, count in sorted(weekly_rule_counts.items()):
            print(f"    {rule}: {count} variables")

        if len(self.weekly_data) < 10:
            print(f"  ⚠ WARNING: Only {len(self.weekly_data)} weekly observations - may be insufficient for robust testing")

        return self.weekly_data

    def advanced_unit_root_test(self, series, name, regression_types=['c', 'ct'], max_lags=None):
        """Advanced unit root testing with comprehensive statistical analysis"""
        print(f"\n{'='*80}")
        print(f"UNIT ROOT TESTING: {name}")
        print(f"{'='*80}")

        results = {
            'name': name,
            'series_info': {},
            'adf_tests': {},
            'kpss_tests': {},
            'stationarity_conclusion': {}
        }

        # Basic series information with comprehensive statistics
        series_clean = series.dropna()
        if len(series_clean) < 30:
            print(f"INSUFFICIENT DATA: Only {len(series_clean)} clean observations")
            results['series_info'] = {'length': len(series_clean), 'status': 'insufficient_data'}
            return results

        results['series_info'] = {
            'total_length': len(series),
            'clean_length': len(series_clean),
            'missing_count': series.isnull().sum(),
            'missing_pct': (series.isnull().sum() / len(series)) * 100,
            'mean': series_clean.mean(),
            'std': series_clean.std(),
            'min': series_clean.min(),
            'max': series_clean.max(),
            'range': series_clean.max() - series_clean.min(),
            'skewness': stats.skew(series_clean) if SCIPY_AVAILABLE and len(series_clean) > 3 else 0,
            'kurtosis': stats.kurtosis(series_clean) if SCIPY_AVAILABLE and len(series_clean) > 3 else 0,
            'is_positive': (series_clean >= 0).all(),
            'zero_count': (series_clean == 0).sum(),
            'zero_pct': ((series_clean == 0).sum() / len(series_clean)) * 100
        }

        print(f"Series characteristics:")
        print(f"  Total observations: {results['series_info']['total_length']:,}")
        print(f"  Clean observations: {results['series_info']['clean_length']:,}")
        print(f"  Missing: {results['series_info']['missing_count']:,} ({results['series_info']['missing_pct']:.2f}%)")
        print(f"  Mean: {results['series_info']['mean']:.6f}")
        print(f"  Std: {results['series_info']['std']:.6f}")
        print(f"  Range: [{results['series_info']['min']:.6f}, {results['series_info']['max']:.6f}]")
        print(f"  Skewness: {results['series_info']['skewness']:.4f}")
        print(f"  Kurtosis: {results['series_info']['kurtosis']:.4f}")
        print(f"  Zero values: {results['series_info']['zero_count']:,} ({results['series_info']['zero_pct']:.1f}%)")

        # Test at multiple differencing orders
        test_orders = [0, 1, 2]
        order_labels = ['Level', 'First Difference', 'Second Difference']

        for diff_order, order_label in zip(test_orders, order_labels):
            print(f"\n{'-'*60}")
            print(f"TESTING: {order_label.upper()}")
            print(f"{'-'*60}")

            # Create test series
            if diff_order == 0:
                test_series = series_clean
            else:
                test_series = series_clean.diff(diff_order).dropna()

            if len(test_series) < 20:
                print(f"  Insufficient observations after differencing: {len(test_series)}")
                continue

            print(f"  Test series length: {len(test_series):,}")
            print(f"  Test series mean: {test_series.mean():.6f}")
            print(f"  Test series std: {test_series.std():.6f}")

            # ADF Tests
            print(f"\n  AUGMENTED DICKEY-FULLER TESTS:")
            for reg_type in regression_types:
                try:
                    if STATSMODELS_AVAILABLE:
                        adf_result = adfuller(test_series, regression=reg_type, autolag='AIC', maxlag=max_lags)
                        adf_data = {
                            'statistic': adf_result[0],
                            'pvalue': adf_result[1],
                            'usedlag': adf_result[2],
                            'nobs': adf_result[3],
                            'critical_values': adf_result[4],
                            'icbest': adf_result[5] if len(adf_result) > 5 else None,
                            'regression': reg_type,
                            'decision': 'Stationary' if adf_result[1] < 0.05 else 'Unit Root'
                        }

                        results['adf_tests'][f'{order_label.lower()}_{reg_type}'] = adf_data

                        print(f"    ADF ({reg_type:2s}): stat={adf_data['statistic']:8.4f}, p={adf_data['pvalue']:7.4f}, lags={adf_data['usedlag']:2d}")
                        print(f"              nobs={adf_data['nobs']:4d}, decision={adf_data['decision']}")

                        # Critical values
                        cv = adf_data['critical_values']
                        print(f"              Critical values: 1%={cv['1%']:.3f}, 5%={cv['5%']:.3f}, 10%={cv['10%']:.3f}")

                        # Significance assessment
                        if adf_data['statistic'] < cv['1%']:
                            significance = "1% level (***)"
                        elif adf_data['statistic'] < cv['5%']:
                            significance = "5% level (**)"
                        elif adf_data['statistic'] < cv['10%']:
                            significance = "10% level (*)"
                        else:
                            significance = "Not significant"
                        print(f"              Significance: {significance}")

                    else:
                        print(f"    ADF ({reg_type:2s}): UNAVAILABLE (statsmodels not installed)")
                        results['adf_tests'][f'{order_label.lower()}_{reg_type}'] = {'error': 'statsmodels_unavailable'}

                except Exception as e:
                    print(f"    ADF ({reg_type:2s}): ERROR - {str(e)}")
                    results['adf_tests'][f'{order_label.lower()}_{reg_type}'] = {'error': str(e)}

            # KPSS Tests
            print(f"\n  KPSS STATIONARITY TESTS:")
            for reg_type in regression_types:
                try:
                    if STATSMODELS_AVAILABLE:
                        kpss_result = kpss(test_series, regression=reg_type, nlags='auto')
                        kpss_data = {
                            'statistic': kpss_result[0],
                            'pvalue': kpss_result[1],
                            'usedlag': kpss_result[2],
                            'critical_values': kpss_result[3],
                            'regression': reg_type,
                            'decision': 'Non-stationary' if kpss_result[1] < 0.05 else 'Stationary'
                        }

                        results['kpss_tests'][f'{order_label.lower()}_{reg_type}'] = kpss_data

                        print(f"    KPSS({reg_type:2s}): stat={kpss_data['statistic']:8.4f}, p={kpss_data['pvalue']:7.4f}, lags={kpss_data['usedlag']:2d}")
                        print(f"              decision={kpss_data['decision']}")

                        # Critical values
                        cv = kpss_data['critical_values']
                        print(f"              Critical values: 10%={cv['10%']:.3f}, 5%={cv['5%']:.3f}, 2.5%={cv['2.5%']:.3f}, 1%={cv['1%']:.3f}")

                    else:
                        print(f"    KPSS({reg_type:2s}): UNAVAILABLE (statsmodels not installed)")
                        results['kpss_tests'][f'{order_label.lower()}_{reg_type}'] = {'error': 'statsmodels_unavailable'}

                except Exception as e:
                    print(f"    KPSS({reg_type:2s}): ERROR - {str(e)}")
                    results['kpss_tests'][f'{order_label.lower()}_{reg_type}'] = {'error': str(e)}

            # Integration order determination for this level
            self.determine_integration_order_for_level(results, diff_order, order_label)

        return results

    def determine_integration_order_for_level(self, test_results, diff_order, order_label):
        """Determine integration order based on test results"""
        print(f"\n  INTEGRATION ORDER ASSESSMENT FOR {order_label.upper()}:")

        # Collect test outcomes
        adf_outcomes = []
        kpss_outcomes = []

        for test_name, test_result in test_results['adf_tests'].items():
            if test_name.startswith(order_label.lower()) and 'error' not in test_result:
                adf_outcomes.append(test_result['decision'] == 'Stationary')

        for test_name, test_result in test_results['kpss_tests'].items():
            if test_name.startswith(order_label.lower()) and 'error' not in test_result:
                kpss_outcomes.append(test_result['decision'] == 'Stationary')

        # Decision logic
        adf_stationary_count = sum(adf_outcomes)
        kpss_stationary_count = sum(kpss_outcomes)
        total_adf_tests = len(adf_outcomes)
        total_kpss_tests = len(kpss_outcomes)

        print(f"    ADF tests indicating stationarity: {adf_stationary_count}/{total_adf_tests}")
        print(f"    KPSS tests indicating stationarity: {kpss_stationary_count}/{total_kpss_tests}")

        if total_adf_tests == 0 and total_kpss_tests == 0:
            conclusion = "Cannot determine - no valid tests"
            confidence = "None"
        elif adf_stationary_count == total_adf_tests and kpss_stationary_count == total_kpss_tests:
            conclusion = f"I({diff_order}) - Strong evidence for stationarity"
            confidence = "High"
        elif adf_stationary_count > 0 and kpss_stationary_count > 0:
            conclusion = f"I({diff_order}) - Moderate evidence for stationarity"
            confidence = "Medium"
        elif adf_stationary_count > 0 or kpss_stationary_count > 0:
            conclusion = f"I({diff_order}) - Weak evidence for stationarity"
            confidence = "Low"
        else:
            conclusion = f"Not I({diff_order}) - Evidence suggests non-stationarity"
            confidence = "Medium"

        print(f"    CONCLUSION: {conclusion}")
        print(f"    CONFIDENCE: {confidence}")

        test_results['stationarity_conclusion'][order_label.lower()] = {
            'order': diff_order,
            'conclusion': conclusion,
            'confidence': confidence,
            'adf_support': f"{adf_stationary_count}/{total_adf_tests}",
            'kpss_support': f"{kpss_stationary_count}/{total_kpss_tests}"
        }

    def test_log_transformation(self, series, name):
        """Test log transformation for positive series"""
        if not (series > 0).all():
            return None

        print(f"\n{'='*60}")
        print(f"LOG TRANSFORMATION ANALYSIS: {name}")
        print(f"{'='*60}")

        try:
            log_series = np.log(series)
            log_results = self.advanced_unit_root_test(log_series, f"log_{name}")

            # Compare original vs log transformation
            orig_std = series.std()
            log_std = log_series.std()

            print(f"\nTransformation comparison:")
            print(f"  Original std: {orig_std:.6f}")
            print(f"  Log std: {log_std:.6f}")
            print(f"  Transformation effect: {'Stabilizing' if log_std < orig_std else 'Destabilizing'}")

            return log_results

        except Exception as e:
            print(f"Log transformation failed: {str(e)}")
            return None

    def run_comprehensive_testing(self, frequencies=['half_hourly', 'daily']):
        """Run comprehensive unit root testing across multiple frequencies"""
        print("\n" + "="*100)
        print("COMPREHENSIVE UNIT ROOT TESTING ACROSS FREQUENCIES")
        print("="*100)
        print(f"Testing frequencies: {', '.join(frequencies)}")
        print(f"Statistical testing available: {'Yes' if STATSMODELS_AVAILABLE else 'No (using fallback methods)'}")

        all_results = {}

        for frequency in frequencies:
            print(f"\n{'='*100}")
            print(f"FREQUENCY: {frequency.upper()}")
            print(f"{'='*100}")

            # Select appropriate dataset
            if frequency == 'half_hourly':
                data = self.half_hourly_data
            elif frequency == 'daily':
                data = self.daily_data
            elif frequency == 'weekly':
                data = self.weekly_data
            else:
                print(f"Unknown frequency: {frequency}")
                continue

            if data is None:
                print(f"No {frequency} data available")
                continue

            freq_results = {}
            print(f"Testing {len(data.columns)} variables at {frequency} frequency...")

            # Test each variable
            for i, col in enumerate(data.columns, 1):
                print(f"\n{'#'*100}")
                print(f"VARIABLE {i}/{len(data.columns)} ({frequency.upper()}): {col}")
                print(f"{'#'*100}")

                series = data[col]

                # Main unit root testing
                test_results = self.advanced_unit_root_test(series, f"{col}_{frequency}")
                freq_results[col] = test_results

                # Test log transformation if applicable
                if (series > 0).all() and len(series.dropna()) > 30:
                    log_results = self.test_log_transformation(series, f"{col}_{frequency}")
                    if log_results:
                        freq_results[f"log_{col}"] = log_results

            # Store results for this frequency
            all_results[frequency] = freq_results
            self.test_results[frequency] = freq_results

            # Frequency summary
            self.summarize_frequency_results(frequency, freq_results)

        return all_results

    def summarize_frequency_results(self, frequency, results):
        """Summarize results for a specific frequency"""
        print(f"\n{'='*80}")
        print(f"SUMMARY FOR {frequency.upper()} FREQUENCY")
        print(f"{'='*80}")

        if not results:
            print("No results to summarize")
            return

        # Count integration orders
        integration_counts = {}
        confidence_counts = {}

        for var_name, test_result in results.items():
            if 'stationarity_conclusion' in test_result:
                conclusions = test_result['stationarity_conclusion']

                # Find the best conclusion (lowest order with high confidence)
                best_conclusion = None
                best_order = float('inf')

                for level, conclusion_data in conclusions.items():
                    if conclusion_data['confidence'] in ['High', 'Medium']:
                        order = conclusion_data['order']
                        if order < best_order:
                            best_order = order
                            best_conclusion = conclusion_data

                if best_conclusion:
                    order_key = f"I({best_conclusion['order']})"
                    integration_counts[order_key] = integration_counts.get(order_key, 0) + 1
                    confidence_counts[best_conclusion['confidence']] = confidence_counts.get(best_conclusion['confidence'], 0) + 1
                else:
                    integration_counts['Undetermined'] = integration_counts.get('Undetermined', 0) + 1

        print(f"Integration order distribution:")
        for order, count in sorted(integration_counts.items()):
            print(f"  {order}: {count} variables")

        print(f"\nConfidence distribution:")
        for conf, count in sorted(confidence_counts.items()):
            print(f"  {conf}: {count} variables")

        print(f"\nTotal variables tested: {len(results)}")

    def save_comprehensive_results(self, filename='integrated_comprehensive_analysis_results.txt'):
        """Save all comprehensive results to file"""
        print(f"\n{'='*100}")
        print(f"SAVING COMPREHENSIVE RESULTS")
        print(f"{'='*100}")
        print(f"Output file: {filename}")

        try:
            with open(filename, 'w') as f:
                # Header
                f.write("="*120 + "\n")
                f.write("INTEGRATED COMPREHENSIVE TIME SERIES ANALYSIS RESULTS\n")
                f.write(f"Generated: {datetime.now()}\n")
                f.write(f"Statistical Libraries: statsmodels={STATSMODELS_AVAILABLE}, scipy={SCIPY_AVAILABLE}\n")
                f.write("="*120 + "\n\n")

                # Data summary
                f.write("DATA SUMMARY\n")
                f.write("-"*40 + "\n")
                if self.half_hourly_data is not None:
                    f.write(f"Half-hourly data: {self.half_hourly_data.shape[0]:,} obs × {self.half_hourly_data.shape[1]} vars\n")
                    f.write(f"Date range: {self.half_hourly_data.index.min()} to {self.half_hourly_data.index.max()}\n")
                if self.daily_data is not None:
                    f.write(f"Daily data: {self.daily_data.shape[0]:,} obs × {self.daily_data.shape[1]} vars\n")
                if self.weekly_data is not None:
                    f.write(f"Weekly data: {self.weekly_data.shape[0]:,} obs × {self.weekly_data.shape[1]} vars\n")

                # Seasonal analysis results
                if self.seasonal_analysis:
                    f.write(f"\n\nSEASONAL ANALYSIS RESULTS\n")
                    f.write("="*60 + "\n")

                    for var_name, analysis in self.seasonal_analysis.items():
                        f.write(f"\nVariable: {var_name}\n")
                        f.write(f"Length: {analysis.get('length', 'N/A')}, Missing: {analysis.get('missing', 'N/A')}\n")
                        f.write(f"Mean: {analysis.get('mean', 0):.6f}, Std: {analysis.get('std', 0):.6f}\n")
                        f.write(f"Range: [{analysis.get('min', 0):.6f}, {analysis.get('max', 0):.6f}]\n")
                        f.write(f"Non-zero: {analysis.get('non_zero_count', 0)} ({analysis.get('non_zero_pct', 0):.1f}%)\n")
                        f.write(f"Hourly variation: {analysis.get('hourly_variation', 0):.4f}\n")
                        f.write(f"Daily variation: {analysis.get('daily_variation', 0):.4f}\n")
                        f.write(f"Weekly variation: {analysis.get('weekly_variation', 0):.4f}\n")
                        f.write(f"Overall seasonality: {analysis.get('seasonality_assessment', 'Unknown')}\n")
                        if 'peak_hour' in analysis:
                            f.write(f"Peak hour: {analysis['peak_hour']}, Peak day: {analysis.get('peak_day', 'N/A')}\n")
                        if 'autocorrelations' in analysis:
                            f.write(f"Autocorrelations: {analysis['autocorrelations']}\n")

                # Unit root testing results
                for frequency in ['half_hourly', 'daily', 'weekly']:
                    if frequency not in self.test_results:
                        continue

                    f.write(f"\n\n{'='*120}\n")
                    f.write(f"UNIT ROOT TESTING RESULTS - {frequency.upper()} FREQUENCY\n")
                    f.write("="*120 + "\n")

                    results = self.test_results[frequency]

                    for var_name, test_result in results.items():
                        f.write(f"\n{'-'*100}\n")
                        f.write(f"Variable: {var_name}\n")
                        f.write(f"{'-'*100}\n")

                        # Series info
                        if 'series_info' in test_result:
                            info = test_result['series_info']
                            f.write(f"Series Info:\n")
                            f.write(f"  Total length: {info.get('total_length', 'N/A'):,}\n")
                            f.write(f"  Clean length: {info.get('clean_length', 'N/A'):,}\n")
                            f.write(f"  Missing: {info.get('missing_count', 'N/A'):,} ({info.get('missing_pct', 0):.2f}%)\n")
                            f.write(f"  Mean: {info.get('mean', 0):.6f}\n")
                            f.write(f"  Std: {info.get('std', 0):.6f}\n")
                            f.write(f"  Range: [{info.get('min', 0):.6f}, {info.get('max', 0):.6f}]\n")
                            f.write(f"  Skewness: {info.get('skewness', 0):.4f}\n")
                            f.write(f"  Kurtosis: {info.get('kurtosis', 0):.4f}\n")
                            f.write(f"  Zero values: {info.get('zero_count', 0):,} ({info.get('zero_pct', 0):.1f}%)\n\n")

                        # ADF test results
                        if 'adf_tests' in test_result:
                            f.write(f"ADF Test Results:\n")
                            for test_name, adf_result in test_result['adf_tests'].items():
                                if 'error' not in adf_result:
                                    f.write(f"  {test_name}: stat={adf_result.get('statistic', 0):.4f}, ")
                                    f.write(f"p={adf_result.get('pvalue', 1):.6f}, ")
                                    f.write(f"lags={adf_result.get('usedlag', 0)}, ")
                                    f.write(f"decision={adf_result.get('decision', 'N/A')}\n")
                                else:
                                    f.write(f"  {test_name}: ERROR - {adf_result['error']}\n")
                            f.write("\n")

                        # KPSS test results
                        if 'kpss_tests' in test_result:
                            f.write(f"KPSS Test Results:\n")
                            for test_name, kpss_result in test_result['kpss_tests'].items():
                                if 'error' not in kpss_result:
                                    f.write(f"  {test_name}: stat={kpss_result.get('statistic', 0):.4f}, ")
                                    f.write(f"p={kpss_result.get('pvalue', 1):.6f}, ")
                                    f.write(f"lags={kpss_result.get('usedlag', 0)}, ")
                                    f.write(f"decision={kpss_result.get('decision', 'N/A')}\n")
                                else:
                                    f.write(f"  {test_name}: ERROR - {kpss_result['error']}\n")
                            f.write("\n")

                        # Stationarity conclusions
                        if 'stationarity_conclusion' in test_result:
                            f.write(f"Stationarity Conclusions:\n")
                            for level, conclusion in test_result['stationarity_conclusion'].items():
                                f.write(f"  {level}: {conclusion.get('conclusion', 'N/A')} ")
                                f.write(f"(confidence: {conclusion.get('confidence', 'N/A')})\n")
                                f.write(f"    ADF support: {conclusion.get('adf_support', 'N/A')}, ")
                                f.write(f"KPSS support: {conclusion.get('kpss_support', 'N/A')}\n")

            print(f"✓ Comprehensive results saved successfully to {filename}")
            print(f"  File size: {open(filename).seek(0, 2)} bytes" if open(filename, 'r').readable() else "")

        except Exception as e:
            print(f"✗ Error saving results: {str(e)}")

    def run_complete_integrated_analysis(self):
        """Run the complete integrated analysis pipeline"""
        print("="*120)
        print("INTEGRATED COMPREHENSIVE TIME SERIES ANALYSIS PIPELINE")
        print("="*120)
        print(f"Analysis initiated: {datetime.now()}")
        print(f"Python environment: pandas={pd.__version__}")
        print(f"Statistical capabilities: statsmodels={STATSMODELS_AVAILABLE}, scipy={SCIPY_AVAILABLE}")
        print("="*120)

        try:
            # Step 1: Load and prepare data
            print(f"\nSTEP 1: DATA LOADING AND PREPARATION")
            print("-" * 50)
            self.load_data()
            half_hourly_data, time_features = self.prepare_half_hourly_data()

            if half_hourly_data is None:
                raise ValueError("Failed to prepare half-hourly data")

            # Step 2: Seasonal analysis
            print(f"\nSTEP 2: COMPREHENSIVE SEASONAL ANALYSIS")
            print("-" * 50)
            for i, col in enumerate(self.half_hourly_data.columns, 1):
                print(f"\nAnalyzing seasonality for variable {i}/{len(self.half_hourly_data.columns)}: {col}")
                seasonal_info = self.comprehensive_seasonal_analysis(
                    self.half_hourly_data[col], col, time_features
                )
                self.seasonal_analysis[col] = seasonal_info

            # Step 3: Seasonal adjustments
            print(f"\nSTEP 3: SEASONAL ADJUSTMENT")
            print("-" * 50)
            self.create_seasonal_adjustments()

            # Step 4: Multi-frequency data preparation
            print(f"\nSTEP 4: MULTI-FREQUENCY DATA PREPARATION")
            print("-" * 50)
            self.prepare_daily_data()
            self.prepare_weekly_data()

            # Step 5: Comprehensive unit root testing
            print(f"\nSTEP 5: COMPREHENSIVE UNIT ROOT TESTING")
            print("-" * 50)
            test_frequencies = ['half_hourly', 'daily']
            if self.weekly_data is not None and len(self.weekly_data) >= 10:
                test_frequencies.append('weekly')

            self.run_comprehensive_testing(test_frequencies)

            # Step 6: Save comprehensive results
            print(f"\nSTEP 6: SAVING RESULTS")
            print("-" * 50)
            self.save_comprehensive_results()

            print(f"\n{'='*120}")
            print("INTEGRATED ANALYSIS COMPLETED SUCCESSFULLY")
            print(f"Completion time: {datetime.now()}")

            # Final summary
            total_vars_analyzed = 0
            for freq in ['half_hourly', 'daily', 'weekly']:
                if freq in self.test_results:
                    total_vars_analyzed += len(self.test_results[freq])

            print(f"Total variables analyzed: {total_vars_analyzed}")
            print(f"Seasonal analyses completed: {len(self.seasonal_analysis)}")
            print(f"Frequencies tested: {', '.join(test_frequencies)}")
            print("="*120)

            return {
                'seasonal_analysis': self.seasonal_analysis,
                'test_results': self.test_results,
                'frequencies_tested': test_frequencies
            }

        except Exception as e:
            print(f"\nCRITICAL ERROR in integrated analysis: {str(e)}")
            import traceback
            traceback.print_exc()
            return {}


def main():
    """Main execution function with comprehensive output capture"""
    print("="*120)
    print("INTEGRATED COMPREHENSIVE TIME SERIES ANALYSIS")
    print("STARTING EXECUTION WITH FULL OUTPUT CAPTURE")
    print("="*120)

    # Create output buffer for complete capture
    output_buffer = io.StringIO()

    # Run analysis with complete output capture
    try:
        with redirect_stdout(output_buffer):
            tester = IntegratedTimeSeriesTester()
            results = tester.run_complete_integrated_analysis()
    except Exception as e:
        print(f"CRITICAL ANALYSIS FAILURE: {str(e)}")
        import traceback
        traceback.print_exc()
        results = {}

    # Get complete captured output
    complete_output = output_buffer.getvalue()

    # Print to console for immediate viewing
    print(complete_output)

    # Save complete output to file
    output_filename = 'integrated_comprehensive_analysis_output.txt'
    try:
        with open(output_filename, 'w') as f:
            f.write("="*120 + "\n")
            f.write("INTEGRATED COMPREHENSIVE TIME SERIES ANALYSIS - COMPLETE OUTPUT\n")
            f.write(f"Generated: {datetime.now()}\n")
            f.write(f"Statistical Libraries Available: statsmodels={STATSMODELS_AVAILABLE}, scipy={SCIPY_AVAILABLE}\n")
            f.write("="*120 + "\n\n")
            f.write(complete_output)

        print("\n" + "="*120)
        print("OUTPUT CAPTURE AND STORAGE COMPLETED")
        print("="*120)
        print(f"✓ COMPLETE VERBOSE OUTPUT SAVED TO: {output_filename}")
        print(f"✓ STRUCTURED RESULTS SAVED TO: integrated_comprehensive_analysis_results.txt")
        print(f"✓ ANALYSIS STATUS: {'SUCCESS' if results else 'FAILED'}")

        if results:
            total_analyses = sum(len(freq_results) for freq_results in results.get('test_results', {}).values())
            print(f"✓ TOTAL STATISTICAL ANALYSES COMPLETED: {total_analyses}")

        print("="*120)

    except Exception as e:
        print(f"✗ ERROR SAVING OUTPUT: {str(e)}")

    return results


if __name__ == "__main__":
    main()