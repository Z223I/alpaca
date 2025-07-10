"""
Statistical analysis utilities for alert performance evaluation.

This atom provides advanced statistical functions for analyzing alert patterns,
correlations, and performance distributions.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union
from scipy import stats
from scipy.stats import chi2_contingency, pearsonr, spearmanr
import warnings


def calculate_alert_performance_statistics(
    alerts_df: pd.DataFrame,
    return_col: str = "return_pct",
    status_col: str = "status",
    priority_col: str = "priority"
) -> Dict[str, Any]:
    """
    Calculate comprehensive statistical analysis of alert performance.
    
    Args:
        alerts_df: DataFrame containing alert results
        return_col: Column name for returns
        status_col: Column name for trade status
        priority_col: Column name for priority levels
        
    Returns:
        Dictionary with statistical analysis results
    """
    if alerts_df.empty:
        return {'error': 'No data provided for analysis'}
    
    results = {
        'descriptive_stats': {},
        'distribution_tests': {},
        'priority_analysis': {},
        'performance_patterns': {}
    }
    
    # Descriptive statistics
    if return_col in alerts_df.columns:
        returns = alerts_df[return_col].dropna()
        
        results['descriptive_stats'] = {
            'count': len(returns),
            'mean': returns.mean(),
            'median': returns.median(),
            'std': returns.std(),
            'min': returns.min(),
            'max': returns.max(),
            'q25': returns.quantile(0.25),
            'q75': returns.quantile(0.75),
            'iqr': returns.quantile(0.75) - returns.quantile(0.25)
        }
        
        # Distribution tests
        results['distribution_tests'] = analyze_return_distribution(returns)
        
        # Performance patterns
        results['performance_patterns'] = identify_performance_patterns(alerts_df, return_col)
    
    # Priority analysis
    if priority_col in alerts_df.columns:
        results['priority_analysis'] = analyze_priority_performance(
            alerts_df, return_col, status_col, priority_col
        )
    
    return results


def analyze_return_distribution(returns: pd.Series) -> Dict[str, Any]:
    """
    Analyze the statistical distribution of returns.
    
    Args:
        returns: Series of return values
        
    Returns:
        Dictionary with distribution analysis
    """
    if returns.empty:
        return {'error': 'No returns data provided'}
    
    # Remove any infinite or NaN values
    clean_returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
    
    if clean_returns.empty:
        return {'error': 'No valid returns after cleaning'}
    
    distribution_analysis = {}
    
    # Basic distribution metrics
    distribution_analysis['skewness'] = stats.skew(clean_returns)
    distribution_analysis['kurtosis'] = stats.kurtosis(clean_returns, fisher=True)
    
    # Normality tests
    try:
        # Shapiro-Wilk test (best for n < 5000)
        if len(clean_returns) <= 5000:
            shapiro_stat, shapiro_p = stats.shapiro(clean_returns)
            distribution_analysis['shapiro_wilk'] = {
                'statistic': shapiro_stat,
                'p_value': shapiro_p,
                'is_normal': shapiro_p > 0.05
            }
        
        # Jarque-Bera test
        jb_stat, jb_p = stats.jarque_bera(clean_returns)
        distribution_analysis['jarque_bera'] = {
            'statistic': jb_stat,
            'p_value': jb_p,
            'is_normal': jb_p > 0.05
        }
        
        # Kolmogorov-Smirnov test against normal distribution
        ks_stat, ks_p = stats.kstest(
            clean_returns, 
            lambda x: stats.norm.cdf(x, clean_returns.mean(), clean_returns.std())
        )
        distribution_analysis['kolmogorov_smirnov'] = {
            'statistic': ks_stat,
            'p_value': ks_p,
            'is_normal': ks_p > 0.05
        }
        
    except Exception as e:
        distribution_analysis['test_error'] = str(e)
    
    # Distribution fit analysis
    distribution_analysis['best_fit'] = find_best_distribution_fit(clean_returns)
    
    return distribution_analysis


def find_best_distribution_fit(data: pd.Series) -> Dict[str, Any]:
    """
    Find the best statistical distribution fit for the data.
    
    Args:
        data: Series of data points
        
    Returns:
        Dictionary with best fit information
    """
    if data.empty:
        return {'error': 'No data provided'}
    
    # List of distributions to test
    distributions = [
        stats.norm, stats.lognorm, stats.expon, stats.gamma, 
        stats.beta, stats.uniform, stats.t, stats.laplace
    ]
    
    best_fit = {
        'distribution': None,
        'aic': np.inf,
        'bic': np.inf,
        'parameters': None,
        'ks_statistic': np.inf,
        'ks_p_value': 0.0
    }
    
    for distribution in distributions:
        try:
            # Fit distribution
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                params = distribution.fit(data)
            
            # Calculate goodness of fit
            ks_stat, ks_p = stats.kstest(data, lambda x: distribution.cdf(x, *params))
            
            # Calculate AIC and BIC
            log_likelihood = np.sum(distribution.logpdf(data, *params))
            k = len(params)  # number of parameters
            n = len(data)    # sample size
            
            aic = 2 * k - 2 * log_likelihood
            bic = k * np.log(n) - 2 * log_likelihood
            
            # Update best fit if this is better
            if aic < best_fit['aic']:
                best_fit.update({
                    'distribution': distribution.name,
                    'aic': aic,
                    'bic': bic,
                    'parameters': params,
                    'ks_statistic': ks_stat,
                    'ks_p_value': ks_p
                })
                
        except Exception:
            continue  # Skip distributions that fail to fit
    
    return best_fit


def analyze_priority_performance(
    alerts_df: pd.DataFrame,
    return_col: str,
    status_col: str,
    priority_col: str
) -> Dict[str, Any]:
    """
    Analyze performance differences across priority levels.
    
    Args:
        alerts_df: DataFrame containing alert data
        return_col: Column name for returns
        status_col: Column name for status
        priority_col: Column name for priority
        
    Returns:
        Dictionary with priority analysis results
    """
    if alerts_df.empty:
        return {'error': 'No data provided'}
    
    priority_analysis = {
        'summary_stats': {},
        'statistical_tests': {},
        'effect_sizes': {}
    }
    
    # Group by priority
    priority_groups = alerts_df.groupby(priority_col)
    
    # Summary statistics by priority
    for priority, group in priority_groups:
        if return_col in group.columns:
            returns = group[return_col].dropna()
            
            priority_analysis['summary_stats'][priority] = {
                'count': len(returns),
                'mean_return': returns.mean(),
                'median_return': returns.median(),
                'std_return': returns.std(),
                'success_rate': (group[status_col] == 'SUCCESS').mean() * 100 if status_col in group.columns else None
            }
    
    # Statistical tests between priority groups
    priority_analysis['statistical_tests'] = perform_priority_tests(
        alerts_df, return_col, status_col, priority_col
    )
    
    return priority_analysis


def perform_priority_tests(
    alerts_df: pd.DataFrame,
    return_col: str,
    status_col: str,
    priority_col: str
) -> Dict[str, Any]:
    """
    Perform statistical tests between priority groups.
    
    Args:
        alerts_df: DataFrame containing alert data
        return_col: Column name for returns
        status_col: Column name for status
        priority_col: Column name for priority
        
    Returns:
        Dictionary with test results
    """
    test_results = {}
    
    # Get unique priorities
    priorities = alerts_df[priority_col].unique()
    
    if len(priorities) < 2:
        return {'error': 'Need at least 2 priority levels for comparison'}
    
    # ANOVA test for returns across priorities
    if return_col in alerts_df.columns:
        priority_returns = [
            alerts_df[alerts_df[priority_col] == priority][return_col].dropna()
            for priority in priorities
        ]
        
        # Filter out empty groups
        priority_returns = [group for group in priority_returns if len(group) > 0]
        
        if len(priority_returns) >= 2:
            try:
                f_stat, f_p = stats.f_oneway(*priority_returns)
                test_results['anova_returns'] = {
                    'f_statistic': f_stat,
                    'p_value': f_p,
                    'significant': f_p < 0.05
                }
            except Exception as e:
                test_results['anova_error'] = str(e)
    
    # Chi-square test for success rates across priorities
    if status_col in alerts_df.columns:
        # Create contingency table
        contingency_table = pd.crosstab(
            alerts_df[priority_col],
            alerts_df[status_col] == 'SUCCESS'
        )
        
        if contingency_table.shape[0] >= 2 and contingency_table.shape[1] >= 2:
            try:
                chi2_stat, chi2_p, dof, expected = chi2_contingency(contingency_table)
                test_results['chi_square_success'] = {
                    'chi2_statistic': chi2_stat,
                    'p_value': chi2_p,
                    'degrees_of_freedom': dof,
                    'significant': chi2_p < 0.05
                }
            except Exception as e:
                test_results['chi_square_error'] = str(e)
    
    return test_results


def identify_performance_patterns(
    alerts_df: pd.DataFrame,
    return_col: str,
    min_pattern_size: int = 5
) -> Dict[str, Any]:
    """
    Identify patterns in alert performance data.
    
    Args:
        alerts_df: DataFrame containing alert data
        return_col: Column name for returns
        min_pattern_size: Minimum size for pattern identification
        
    Returns:
        Dictionary with identified patterns
    """
    if alerts_df.empty or return_col not in alerts_df.columns:
        return {'error': 'Insufficient data for pattern analysis'}
    
    patterns = {
        'temporal_patterns': {},
        'streak_analysis': {},
        'volatility_clusters': {}
    }
    
    # Temporal patterns (if timestamp available)
    if 'timestamp' in alerts_df.columns:
        patterns['temporal_patterns'] = analyze_temporal_patterns(alerts_df, return_col)
    
    # Streak analysis
    patterns['streak_analysis'] = analyze_performance_streaks(alerts_df, return_col)
    
    # Volatility clustering
    patterns['volatility_clusters'] = analyze_volatility_clustering(alerts_df[return_col])
    
    return patterns


def analyze_temporal_patterns(
    alerts_df: pd.DataFrame,
    return_col: str
) -> Dict[str, Any]:
    """
    Analyze temporal patterns in alert performance.
    
    Args:
        alerts_df: DataFrame with timestamp and return data
        return_col: Column name for returns
        
    Returns:
        Dictionary with temporal analysis
    """
    temporal_analysis = {}
    
    # Convert timestamp to datetime if needed
    df = alerts_df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Hour of day analysis
    df['hour'] = df['timestamp'].dt.hour
    hour_performance = df.groupby('hour')[return_col].agg(['mean', 'std', 'count'])
    temporal_analysis['hourly_performance'] = hour_performance.to_dict('index')
    
    # Day of week analysis
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    day_performance = df.groupby('day_of_week')[return_col].agg(['mean', 'std', 'count'])
    temporal_analysis['daily_performance'] = day_performance.to_dict('index')
    
    # Time-based correlation test
    df['time_numeric'] = df['timestamp'].astype(np.int64)
    if len(df) > 3:
        corr_coef, corr_p = pearsonr(df['time_numeric'], df[return_col])
        temporal_analysis['time_correlation'] = {
            'correlation': corr_coef,
            'p_value': corr_p,
            'significant': corr_p < 0.05
        }
    
    return temporal_analysis


def analyze_performance_streaks(
    alerts_df: pd.DataFrame,
    return_col: str
) -> Dict[str, Any]:
    """
    Analyze winning and losing streaks in performance.
    
    Args:
        alerts_df: DataFrame containing alert data
        return_col: Column name for returns
        
    Returns:
        Dictionary with streak analysis
    """
    if alerts_df.empty or return_col not in alerts_df.columns:
        return {'error': 'Insufficient data for streak analysis'}
    
    returns = alerts_df[return_col].dropna()
    
    if returns.empty:
        return {'error': 'No valid returns for streak analysis'}
    
    # Classify as wins/losses
    wins = (returns > 0).astype(int)
    
    # Find streaks
    streaks = []
    current_streak = 1
    current_type = wins.iloc[0]
    
    for i in range(1, len(wins)):
        if wins.iloc[i] == current_type:
            current_streak += 1
        else:
            streaks.append((current_type, current_streak))
            current_streak = 1
            current_type = wins.iloc[i]
    
    # Add the last streak
    streaks.append((current_type, current_streak))
    
    # Analyze streaks
    win_streaks = [length for streak_type, length in streaks if streak_type == 1]
    loss_streaks = [length for streak_type, length in streaks if streak_type == 0]
    
    streak_analysis = {
        'total_streaks': len(streaks),
        'win_streaks': {
            'count': len(win_streaks),
            'max_length': max(win_streaks) if win_streaks else 0,
            'avg_length': np.mean(win_streaks) if win_streaks else 0
        },
        'loss_streaks': {
            'count': len(loss_streaks),
            'max_length': max(loss_streaks) if loss_streaks else 0,
            'avg_length': np.mean(loss_streaks) if loss_streaks else 0
        },
        'streak_distribution': pd.Series(streaks).value_counts().to_dict()
    }
    
    return streak_analysis


def analyze_volatility_clustering(returns: pd.Series) -> Dict[str, Any]:
    """
    Analyze volatility clustering in returns.
    
    Args:
        returns: Series of return values
        
    Returns:
        Dictionary with volatility clustering analysis
    """
    if returns.empty or len(returns) < 10:
        return {'error': 'Insufficient data for volatility analysis'}
    
    # Calculate rolling volatility
    rolling_vol = returns.rolling(window=5).std()
    rolling_vol = rolling_vol.dropna()
    
    if len(rolling_vol) < 5:
        return {'error': 'Insufficient data after volatility calculation'}
    
    volatility_analysis = {}
    
    # Autocorrelation in volatility
    if len(rolling_vol) > 2:
        # Lag-1 autocorrelation
        vol_lag1 = rolling_vol.shift(1).dropna()
        vol_current = rolling_vol[1:]
        
        if len(vol_lag1) > 0 and len(vol_current) > 0:
            vol_corr, vol_p = pearsonr(vol_lag1, vol_current)
            volatility_analysis['volatility_autocorr'] = {
                'correlation': vol_corr,
                'p_value': vol_p,
                'clustering_detected': vol_corr > 0.3 and vol_p < 0.05
            }
    
    # High/low volatility periods
    vol_threshold = rolling_vol.quantile(0.75)
    high_vol_periods = rolling_vol > vol_threshold
    
    volatility_analysis['volatility_regime'] = {
        'high_vol_percentage': high_vol_periods.mean() * 100,
        'avg_high_vol': rolling_vol[high_vol_periods].mean(),
        'avg_low_vol': rolling_vol[~high_vol_periods].mean()
    }
    
    return volatility_analysis


def calculate_correlation_matrix(
    alerts_df: pd.DataFrame,
    numeric_columns: Optional[List[str]] = None,
    method: str = "pearson"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Calculate correlation matrix and significance tests.
    
    Args:
        alerts_df: DataFrame containing alert data
        numeric_columns: List of numeric columns to include
        method: Correlation method ('pearson', 'spearman', 'kendall')
        
    Returns:
        Tuple of (correlation_matrix, p_value_matrix)
    """
    if alerts_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    
    if numeric_columns is None:
        # Auto-detect numeric columns
        numeric_columns = alerts_df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Filter to available columns
    available_columns = [col for col in numeric_columns if col in alerts_df.columns]
    
    if len(available_columns) < 2:
        return pd.DataFrame(), pd.DataFrame()
    
    data = alerts_df[available_columns].dropna()
    
    if data.empty:
        return pd.DataFrame(), pd.DataFrame()
    
    # Calculate correlation matrix
    if method == "pearson":
        corr_matrix = data.corr(method='pearson')
    elif method == "spearman":
        corr_matrix = data.corr(method='spearman')
    elif method == "kendall":
        corr_matrix = data.corr(method='kendall')
    else:
        raise ValueError(f"Unknown correlation method: {method}")
    
    # Calculate p-values
    n = len(data)
    p_matrix = pd.DataFrame(index=corr_matrix.index, columns=corr_matrix.columns)
    
    for i, col1 in enumerate(corr_matrix.columns):
        for j, col2 in enumerate(corr_matrix.columns):
            if i <= j:
                if i == j:
                    p_matrix.loc[col1, col2] = 0.0
                else:
                    if method == "pearson":
                        _, p_val = pearsonr(data[col1], data[col2])
                    elif method == "spearman":
                        _, p_val = spearmanr(data[col1], data[col2])
                    else:
                        # For Kendall, use a simple approximation
                        tau = corr_matrix.loc[col1, col2]
                        p_val = 2 * (1 - stats.norm.cdf(abs(tau) * np.sqrt(n * (n - 1) / (2 * (2 * n + 5)))))
                    
                    p_matrix.loc[col1, col2] = p_val
                    p_matrix.loc[col2, col1] = p_val
    
    return corr_matrix, p_matrix.astype(float)