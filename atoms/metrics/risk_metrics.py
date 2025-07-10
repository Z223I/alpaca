"""
Advanced risk metrics for trading performance analysis.

This atom provides comprehensive risk assessment functions including
VaR, CVaR, Sharpe variations, and advanced drawdown metrics.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from scipy import stats


def calculate_value_at_risk(
    returns: pd.Series,
    confidence_level: float = 0.05,
    method: str = "historical"
) -> Dict[str, float]:
    """
    Calculate Value at Risk (VaR) using different methods.
    
    Args:
        returns: Series of returns (in percentage)
        confidence_level: Confidence level (default 5% for 95% VaR)
        method: Calculation method ('historical', 'parametric', 'monte_carlo')
        
    Returns:
        Dictionary with VaR metrics
    """
    if returns.empty:
        return {'var': 0.0, 'cvar': 0.0, 'method': method}
    
    # Convert to decimal for calculations
    returns_decimal = returns / 100
    
    if method == "historical":
        var = np.percentile(returns_decimal, confidence_level * 100) * 100
        
    elif method == "parametric":
        mean_return = returns_decimal.mean()
        std_return = returns_decimal.std()
        var = (mean_return + std_return * stats.norm.ppf(confidence_level)) * 100
        
    elif method == "monte_carlo":
        # Simple Monte Carlo simulation
        n_simulations = 10000
        simulated_returns = np.random.normal(
            returns_decimal.mean(), 
            returns_decimal.std(), 
            n_simulations
        )
        var = np.percentile(simulated_returns, confidence_level * 100) * 100
        
    else:
        raise ValueError(f"Unknown VaR method: {method}")
    
    # Calculate Conditional VaR (Expected Shortfall)
    returns_below_var = returns[returns <= var]
    cvar = returns_below_var.mean() if not returns_below_var.empty else var
    
    return {
        'var': var,
        'cvar': cvar,
        'confidence_level': confidence_level,
        'method': method,
        'observations': len(returns)
    }


def calculate_advanced_sharpe_metrics(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> Dict[str, float]:
    """
    Calculate advanced Sharpe ratio variations.
    
    Args:
        returns: Series of returns (percentage)
        risk_free_rate: Risk-free rate (annual percentage)
        periods_per_year: Number of periods per year
        
    Returns:
        Dictionary with Sharpe variations
    """
    if returns.empty:
        return {
            'sharpe_ratio': 0.0,
            'information_ratio': 0.0,
            'calmar_ratio': 0.0,
            'sterling_ratio': 0.0
        }
    
    # Convert to decimal
    returns_decimal = returns / 100
    rf_decimal = risk_free_rate / 100
    
    # Excess returns
    excess_returns = returns_decimal - (rf_decimal / periods_per_year)
    
    # Standard Sharpe Ratio
    if returns_decimal.std() != 0:
        sharpe_ratio = (excess_returns.mean() * periods_per_year) / (returns_decimal.std() * np.sqrt(periods_per_year))
    else:
        sharpe_ratio = 0.0
    
    # Information Ratio (excess return / tracking error)
    benchmark_returns = pd.Series([rf_decimal / periods_per_year] * len(returns_decimal))
    tracking_error = (returns_decimal - benchmark_returns).std() * np.sqrt(periods_per_year)
    information_ratio = (excess_returns.mean() * periods_per_year) / tracking_error if tracking_error != 0 else 0.0
    
    # Calmar Ratio (annual return / max drawdown)
    from atoms.metrics.calculate_returns import calculate_cumulative_returns, calculate_maximum_drawdown
    cumulative_returns = calculate_cumulative_returns(returns, compound=True)
    max_drawdown = abs(calculate_maximum_drawdown(cumulative_returns))
    annual_return = excess_returns.mean() * periods_per_year * 100
    calmar_ratio = annual_return / max_drawdown if max_drawdown != 0 else 0.0
    
    # Sterling Ratio (similar to Calmar but uses average drawdown)
    drawdowns = calculate_rolling_drawdowns(cumulative_returns)
    avg_drawdown = abs(drawdowns.mean()) if not drawdowns.empty else 1.0
    sterling_ratio = annual_return / avg_drawdown if avg_drawdown != 0 else 0.0
    
    return {
        'sharpe_ratio': sharpe_ratio,
        'information_ratio': information_ratio,
        'calmar_ratio': calmar_ratio,
        'sterling_ratio': sterling_ratio,
        'annual_return': annual_return,
        'annual_volatility': returns_decimal.std() * np.sqrt(periods_per_year) * 100
    }


def calculate_rolling_drawdowns(cumulative_returns: pd.Series) -> pd.Series:
    """
    Calculate rolling drawdowns from cumulative returns.
    
    Args:
        cumulative_returns: Series of cumulative returns (percentage)
        
    Returns:
        Series of drawdown values
    """
    if cumulative_returns.empty:
        return pd.Series(dtype=float)
    
    # Convert to wealth index
    wealth_index = 100 * (1 + cumulative_returns / 100)
    
    # Calculate running maximum
    running_max = wealth_index.expanding().max()
    
    # Calculate drawdown
    drawdown = (wealth_index - running_max) / running_max * 100
    
    return drawdown


def calculate_downside_risk_metrics(
    returns: pd.Series,
    target_return: float = 0.0,
    periods_per_year: int = 252
) -> Dict[str, float]:
    """
    Calculate downside risk metrics.
    
    Args:
        returns: Series of returns (percentage)
        target_return: Target return threshold (percentage)
        periods_per_year: Number of periods per year
        
    Returns:
        Dictionary with downside risk metrics
    """
    if returns.empty:
        return {
            'downside_deviation': 0.0,
            'sortino_ratio': 0.0,
            'downside_frequency': 0.0,
            'pain_index': 0.0
        }
    
    # Convert to decimal
    returns_decimal = returns / 100
    target_decimal = target_return / 100
    
    # Downside returns (returns below target)
    downside_returns = returns_decimal[returns_decimal < target_decimal]
    
    # Downside deviation
    if len(downside_returns) > 0:
        downside_deviation = np.sqrt(((downside_returns - target_decimal) ** 2).mean()) * np.sqrt(periods_per_year) * 100
    else:
        downside_deviation = 0.0
    
    # Sortino Ratio
    excess_return = (returns_decimal.mean() - target_decimal) * periods_per_year * 100
    sortino_ratio = excess_return / downside_deviation if downside_deviation != 0 else 0.0
    
    # Downside frequency
    downside_frequency = len(downside_returns) / len(returns_decimal) * 100
    
    # Pain Index (average drawdown)
    from atoms.metrics.calculate_returns import calculate_cumulative_returns
    cumulative_returns = calculate_cumulative_returns(returns, compound=True)
    drawdowns = calculate_rolling_drawdowns(cumulative_returns)
    pain_index = abs(drawdowns.mean()) if not drawdowns.empty else 0.0
    
    return {
        'downside_deviation': downside_deviation,
        'sortino_ratio': sortino_ratio,
        'downside_frequency': downside_frequency,
        'pain_index': pain_index,
        'worst_return': returns.min(),
        'worst_drawdown': drawdowns.min() if not drawdowns.empty else 0.0
    }


def calculate_tail_risk_metrics(
    returns: pd.Series,
    confidence_levels: Optional[List[float]] = None
) -> Dict[str, Any]:
    """
    Calculate tail risk metrics including skewness, kurtosis, and tail ratios.
    
    Args:
        returns: Series of returns (percentage)
        confidence_levels: List of confidence levels for tail analysis
        
    Returns:
        Dictionary with tail risk metrics
    """
    if returns.empty:
        return {
            'skewness': 0.0,
            'kurtosis': 0.0,
            'tail_ratio': 0.0,
            'var_metrics': {}
        }
    
    if confidence_levels is None:
        confidence_levels = [0.01, 0.05, 0.10]
    
    # Convert to decimal for calculations
    returns_decimal = returns / 100
    
    # Skewness and Kurtosis
    skewness = stats.skew(returns_decimal)
    kurtosis = stats.kurtosis(returns_decimal, fisher=True)  # Excess kurtosis
    
    # Tail Ratio (ratio of gains to losses in tail regions)
    upper_tail = returns[returns >= returns.quantile(0.95)]
    lower_tail = returns[returns <= returns.quantile(0.05)]
    
    if len(lower_tail) > 0 and lower_tail.mean() != 0:
        tail_ratio = abs(upper_tail.mean()) / abs(lower_tail.mean())
    else:
        tail_ratio = 0.0
    
    # VaR at different confidence levels
    var_metrics = {}
    for confidence in confidence_levels:
        var_result = calculate_value_at_risk(returns, confidence, method="historical")
        var_metrics[f'var_{int(confidence*100)}'] = var_result['var']
        var_metrics[f'cvar_{int(confidence*100)}'] = var_result['cvar']
    
    # Expected Tail Loss (average of worst 5% returns)
    worst_returns = returns[returns <= returns.quantile(0.05)]
    expected_tail_loss = worst_returns.mean() if not worst_returns.empty else 0.0
    
    return {
        'skewness': skewness,
        'kurtosis': kurtosis,
        'tail_ratio': tail_ratio,
        'expected_tail_loss': expected_tail_loss,
        'var_metrics': var_metrics,
        'distribution_test': {
            'jarque_bera_stat': stats.jarque_bera(returns_decimal)[0],
            'jarque_bera_pvalue': stats.jarque_bera(returns_decimal)[1],
            'is_normal': stats.jarque_bera(returns_decimal)[1] > 0.05
        }
    }


def calculate_rolling_risk_metrics(
    returns: pd.Series,
    window: int = 30,
    metrics: Optional[List[str]] = None
) -> pd.DataFrame:
    """
    Calculate rolling risk metrics over time.
    
    Args:
        returns: Series of returns (percentage)
        window: Rolling window size
        metrics: List of metrics to calculate
        
    Returns:
        DataFrame with rolling risk metrics
    """
    if returns.empty or len(returns) < window:
        return pd.DataFrame()
    
    if metrics is None:
        metrics = ['volatility', 'sharpe_ratio', 'max_drawdown', 'var_5']
    
    results = {}
    
    for i in range(window - 1, len(returns)):
        window_returns = returns.iloc[i - window + 1:i + 1]
        window_date = returns.index[i] if hasattr(returns, 'index') else i
        
        window_metrics = {}
        
        if 'volatility' in metrics:
            window_metrics['volatility'] = window_returns.std()
        
        if 'sharpe_ratio' in metrics:
            sharpe_metrics = calculate_advanced_sharpe_metrics(window_returns)
            window_metrics['sharpe_ratio'] = sharpe_metrics['sharpe_ratio']
        
        if 'max_drawdown' in metrics:
            from atoms.metrics.calculate_returns import calculate_cumulative_returns, calculate_maximum_drawdown
            cum_returns = calculate_cumulative_returns(window_returns, compound=True)
            window_metrics['max_drawdown'] = calculate_maximum_drawdown(cum_returns)
        
        if 'var_5' in metrics:
            var_result = calculate_value_at_risk(window_returns, confidence_level=0.05)
            window_metrics['var_5'] = var_result['var']
        
        results[window_date] = window_metrics
    
    return pd.DataFrame.from_dict(results, orient='index')


def calculate_regime_adjusted_metrics(
    returns: pd.Series,
    regime_indicator: pd.Series,
    regimes: Optional[List[str]] = None
) -> Dict[str, Dict[str, float]]:
    """
    Calculate risk metrics adjusted for different market regimes.
    
    Args:
        returns: Series of returns (percentage)
        regime_indicator: Series indicating market regime for each period
        regimes: List of regime names to analyze
        
    Returns:
        Dictionary with metrics by regime
    """
    if returns.empty or regime_indicator.empty:
        return {}
    
    if regimes is None:
        regimes = regime_indicator.unique().tolist()
    
    regime_metrics = {}
    
    for regime in regimes:
        regime_mask = regime_indicator == regime
        regime_returns = returns[regime_mask]
        
        if regime_returns.empty:
            continue
        
        # Calculate comprehensive metrics for this regime
        sharpe_metrics = calculate_advanced_sharpe_metrics(regime_returns)
        downside_metrics = calculate_downside_risk_metrics(regime_returns)
        var_metrics = calculate_value_at_risk(regime_returns)
        tail_metrics = calculate_tail_risk_metrics(regime_returns)
        
        regime_metrics[regime] = {
            'count': len(regime_returns),
            'mean_return': regime_returns.mean(),
            'volatility': regime_returns.std(),
            'sharpe_ratio': sharpe_metrics['sharpe_ratio'],
            'sortino_ratio': downside_metrics['sortino_ratio'],
            'max_return': regime_returns.max(),
            'min_return': regime_returns.min(),
            'var_5': var_metrics['var'],
            'cvar_5': var_metrics['cvar'],
            'skewness': tail_metrics['skewness'],
            'kurtosis': tail_metrics['kurtosis']
        }
    
    return regime_metrics