# -*- coding: utf-8 -*-
"""
Statistical Analysis for Social Science Research
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Any, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

class SocialScienceStats:
    """Statistical analysis tools for social science research"""
    
    def __init__(self):
        self.results = {}
    
    def descriptive_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Comprehensive descriptive statistics"""
        
        results = {
            'summary_stats': data.describe(),
            'missing_data': data.isnull().sum(),
            'data_types': data.dtypes,
            'correlations': data.corr() if len(data.select_dtypes(include=[np.number]).columns) > 1 else None
        }
        
        return results
    
    def reliability_analysis(self, data: pd.DataFrame, items: List[str]) -> Dict[str, float]:
        """Calculate Cronbach's alpha for scale reliability"""
        
        if len(items) < 2:
            return {'error': 'Need at least 2 items for reliability analysis'}
        
        # Get item data
        item_data = data[items].dropna()
        
        if item_data.empty:
            return {'error': 'No valid data for reliability analysis'}
        
        # Calculate Cronbach's alpha
        n_items = len(items)
        item_variances = item_data.var(axis=0, ddof=1)
        total_variance = item_data.sum(axis=1).var(ddof=1)
        
        alpha = (n_items / (n_items - 1)) * (1 - (item_variances.sum() / total_variance))
        
        return {
            'cronbach_alpha': alpha,
            'n_items': n_items,
            'n_cases': len(item_data),
            'interpretation': self._interpret_alpha(alpha)
        }
    
    def _interpret_alpha(self, alpha: float) -> str:
        """Interpret Cronbach's alpha value"""
        if alpha >= 0.9:
            return "Excellent reliability"
        elif alpha >= 0.8:
            return "Good reliability"
        elif alpha >= 0.7:
            return "Acceptable reliability"
        elif alpha >= 0.6:
            return "Questionable reliability"
        else:
            return "Poor reliability"
    
    def correlation_analysis(self, data: pd.DataFrame, variables: List[str]) -> Dict[str, Any]:
        """Correlation analysis with significance tests"""
        
        var_data = data[variables].dropna()
        
        # Pearson correlations
        corr_matrix = var_data.corr()
        
        # Calculate p-values
        n = len(var_data)
        p_values = np.zeros((len(variables), len(variables)))
        
        for i, var1 in enumerate(variables):
            for j, var2 in enumerate(variables):
                if i != j:
                    r = corr_matrix.iloc[i, j]
                    t_stat = r * np.sqrt((n - 2) / (1 - r**2))
                    p_val = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
                    p_values[i, j] = p_val
        
        p_matrix = pd.DataFrame(p_values, index=variables, columns=variables)
        
        return {
            'correlation_matrix': corr_matrix,
            'p_values': p_matrix,
            'sample_size': n,
            'significant_correlations': self._identify_significant_correlations(corr_matrix, p_matrix)
        }
    
    def _identify_significant_correlations(self, corr_matrix: pd.DataFrame, 
                                         p_matrix: pd.DataFrame, alpha: float = 0.05) -> List[Dict]:
        """Identify statistically significant correlations"""
        
        significant = []
        
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                var1 = corr_matrix.columns[i]
                var2 = corr_matrix.columns[j]
                r = corr_matrix.iloc[i, j]
                p = p_matrix.iloc[i, j]
                
                if p < alpha:
                    significant.append({
                        'variable1': var1,
                        'variable2': var2,
                        'correlation': r,
                        'p_value': p,
                        'strength': self._interpret_correlation_strength(abs(r))
                    })
        
        return significant
    
    def _interpret_correlation_strength(self, r: float) -> str:
        """Interpret correlation strength"""
        if r >= 0.7:
            return "Strong"
        elif r >= 0.5:
            return "Moderate"
        elif r >= 0.3:
            return "Weak"
        else:
            return "Very weak"
    
    def regression_analysis(self, data: pd.DataFrame, dependent: str, 
                          independent: List[str]) -> Dict[str, Any]:
        """Multiple regression analysis"""
        
        # Prepare data
        reg_data = data[[dependent] + independent].dropna()
        
        if reg_data.empty:
            return {'error': 'No valid data for regression analysis'}
        
        y = reg_data[dependent]
        X = reg_data[independent]
        
        # Add constant for intercept
        X_with_const = np.column_stack([np.ones(len(X)), X])
        
        # Calculate regression coefficients
        try:
            beta = np.linalg.inv(X_with_const.T @ X_with_const) @ X_with_const.T @ y
        except np.linalg.LinAlgError:
            return {'error': 'Singular matrix - multicollinearity issue'}
        
        # Predictions and residuals
        y_pred = X_with_const @ beta
        residuals = y - y_pred
        
        # R-squared
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r_squared = 1 - (ss_res / ss_tot)
        
        # Adjusted R-squared
        n = len(y)
        p = len(independent)
        adj_r_squared = 1 - ((1 - r_squared) * (n - 1) / (n - p - 1))
        
        return {
            'coefficients': dict(zip(['intercept'] + independent, beta)),
            'r_squared': r_squared,
            'adjusted_r_squared': adj_r_squared,
            'sample_size': n,
            'residuals': residuals,
            'predictions': y_pred
        }
    
    def t_test_analysis(self, data: pd.DataFrame, variable: str, 
                       group_var: str = None, test_value: float = None) -> Dict[str, Any]:
        """T-test analysis (one-sample, independent samples)"""
        
        if group_var is None and test_value is None:
            return {'error': 'Must specify either group variable or test value'}
        
        if test_value is not None:
            # One-sample t-test
            sample_data = data[variable].dropna()
            t_stat, p_value = stats.ttest_1samp(sample_data, test_value)
            
            return {
                'test_type': 'one_sample',
                'test_value': test_value,
                'sample_mean': sample_data.mean(),
                'sample_std': sample_data.std(),
                'sample_size': len(sample_data),
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
        
        else:
            # Independent samples t-test
            groups = data[group_var].unique()
            if len(groups) != 2:
                return {'error': 'Group variable must have exactly 2 groups'}
            
            group1_data = data[data[group_var] == groups[0]][variable].dropna()
            group2_data = data[data[group_var] == groups[1]][variable].dropna()
            
            t_stat, p_value = stats.ttest_ind(group1_data, group2_data)
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt(((len(group1_data) - 1) * group1_data.var() + 
                                 (len(group2_data) - 1) * group2_data.var()) / 
                                (len(group1_data) + len(group2_data) - 2))
            cohens_d = (group1_data.mean() - group2_data.mean()) / pooled_std
            
            return {
                'test_type': 'independent_samples',
                'group1': {'name': groups[0], 'mean': group1_data.mean(), 'n': len(group1_data)},
                'group2': {'name': groups[1], 'mean': group2_data.mean(), 'n': len(group2_data)},
                't_statistic': t_stat,
                'p_value': p_value,
                'cohens_d': cohens_d,
                'effect_size_interpretation': self._interpret_effect_size(abs(cohens_d)),
                'significant': p_value < 0.05
            }
    
    def _interpret_effect_size(self, d: float) -> str:
        """Interpret Cohen's d effect size"""
        if d >= 0.8:
            return "Large effect"
        elif d >= 0.5:
            return "Medium effect"
        elif d >= 0.2:
            return "Small effect"
        else:
            return "Negligible effect"
    
    def generate_analysis_report(self, data: pd.DataFrame, 
                               constructs: Dict[str, List[str]]) -> Dict[str, Any]:
        """Generate comprehensive analysis report"""
        
        report = {
            'sample_description': self.descriptive_analysis(data),
            'reliability_analysis': {},
            'correlation_analysis': {},
            'summary': {}
        }
        
        # Reliability analysis for each construct
        for construct_name, items in constructs.items():
            if len(items) >= 2:
                reliability = self.reliability_analysis(data, items)
                report['reliability_analysis'][construct_name] = reliability
        
        # Correlation analysis between constructs
        construct_scores = {}
        for construct_name, items in constructs.items():
            if len(items) >= 2:
                construct_scores[construct_name] = data[items].mean(axis=1)
        
        if len(construct_scores) >= 2:
            construct_df = pd.DataFrame(construct_scores)
            correlation_results = self.correlation_analysis(construct_df, list(construct_scores.keys()))
            report['correlation_analysis'] = correlation_results
        
        # Summary
        report['summary'] = {
            'total_participants': len(data),
            'constructs_analyzed': len(constructs),
            'reliable_constructs': sum(1 for r in report['reliability_analysis'].values() 
                                     if isinstance(r, dict) and r.get('cronbach_alpha', 0) >= 0.7),
            'significant_correlations': len(report['correlation_analysis'].get('significant_correlations', []))
        }
        
        return report