# -*- coding: utf-8 -*-
"""
Mixed Methods Integration for Social Science Research
Advanced integration of quantitative and qualitative methodologies following established social science practices
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from enum import Enum
import json
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns

class MixedMethodsDesign(Enum):
    """Mixed methods research designs (Creswell & Plano Clark, 2017)"""
    CONVERGENT_PARALLEL = "convergent_parallel"
    EXPLANATORY_SEQUENTIAL = "explanatory_sequential"
    EXPLORATORY_SEQUENTIAL = "exploratory_sequential"
    EMBEDDED = "embedded"
    TRANSFORMATIVE = "transformative"
    MULTIPHASE = "multiphase"

class IntegrationLevel(Enum):
    """Levels of integration in mixed methods research"""
    MINIMAL = "minimal"  # Separate analysis, comparison at interpretation
    MODERATE = "moderate"  # Some integration during analysis
    MAXIMAL = "maximal"  # Full integration throughout

class DataTransformationType(Enum):
    """Types of data transformation for integration"""
    QUANTITIZING = "quantitizing"  # Qualitative to quantitative
    QUALITIZING = "qualitizing"  # Quantitative to qualitative
    JOINT_DISPLAYS = "joint_displays"  # Side-by-side comparison
    NARRATIVE_WEAVING = "narrative_weaving"  # Integrated narrative

@dataclass
class QuantitativeData:
    """Structure for quantitative data"""
    data: pd.DataFrame
    variables: Dict[str, str]  # variable_name: variable_type
    sample_size: int
    reliability_scores: Dict[str, float]
    validity_evidence: Dict[str, Any]
    statistical_assumptions: Dict[str, bool]

@dataclass
class QualitativeData:
    """Structure for qualitative data"""
    raw_data: List[Dict[str, Any]]  # interviews, observations, etc.
    codes: Dict[str, List[str]]  # code: [data_segments]
    themes: Dict[str, Dict[str, Any]]  # theme: {description, codes, quotes}
    categories: Dict[str, List[str]]  # category: [themes]
    theoretical_model: Dict[str, Any]
    trustworthiness_criteria: Dict[str, Any]

@dataclass
class IntegrationResult:
    """Result of mixed methods integration"""
    design_type: MixedMethodsDesign
    integration_level: IntegrationLevel
    convergent_findings: List[str]
    divergent_findings: List[str]
    complementary_findings: List[str]
    meta_inferences: List[str]
    joint_displays: Dict[str, Any]
    integrated_model: Dict[str, Any]
    quality_indicators: Dict[str, float]

class MixedMethodsIntegrator:
    """
    Advanced mixed methods integration following social science best practices
    Based on Creswell & Plano Clark (2017), Tashakkori & Teddlie (2010)
    """
    
    def __init__(self):
        """Initialize mixed methods integrator"""
        self.quantitative_data = None
        self.qualitative_data = None
        self.integration_results = {}
        
    def load_quantitative_data(self, data: pd.DataFrame, 
                             variables: Dict[str, str],
                             reliability_scores: Dict[str, float] = None,
                             validity_evidence: Dict[str, Any] = None) -> None:
        """Load and validate quantitative data"""
        
        # Basic validation
        if data.empty:
            raise ValueError("Quantitative data cannot be empty")
        
        # Check for missing data patterns
        missing_patterns = data.isnull().sum()
        
        # Test statistical assumptions
        assumptions = self._test_statistical_assumptions(data)
        
        self.quantitative_data = QuantitativeData(
            data=data,
            variables=variables or {},
            sample_size=len(data),
            reliability_scores=reliability_scores or {},
            validity_evidence=validity_evidence or {},
            statistical_assumptions=assumptions
        )
        
    def load_qualitative_data(self, raw_data: List[Dict[str, Any]],
                            codes: Dict[str, List[str]] = None,
                            themes: Dict[str, Dict[str, Any]] = None) -> None:
        """Load and structure qualitative data"""
        
        if not raw_data:
            raise ValueError("Qualitative data cannot be empty")
        
        # If codes and themes not provided, perform basic analysis
        if codes is None:
            codes = self._extract_basic_codes(raw_data)
        
        if themes is None:
            themes = self._identify_basic_themes(codes)
        
        # Assess trustworthiness
        trustworthiness = self._assess_trustworthiness(raw_data, codes, themes)
        
        self.qualitative_data = QualitativeData(
            raw_data=raw_data,
            codes=codes,
            themes=themes,
            categories=self._organize_categories(themes),
            theoretical_model=self._build_theoretical_model(themes),
            trustworthiness_criteria=trustworthiness
        )
    
    def integrate_convergent_parallel(self, integration_level: IntegrationLevel = IntegrationLevel.MODERATE) -> IntegrationResult:
        """
        Integrate data using convergent parallel design
        Both datasets collected simultaneously and analyzed separately, then integrated
        """
        
        if not self.quantitative_data or not self.qualitative_data:
            raise ValueError("Both quantitative and qualitative data required")
        
        # Separate analysis of each dataset
        quant_results = self._analyze_quantitative_data()
        qual_results = self._analyze_qualitative_data()
        
        # Integration strategies
        convergent_findings = self._identify_convergent_findings(quant_results, qual_results)
        divergent_findings = self._identify_divergent_findings(quant_results, qual_results)
        complementary_findings = self._identify_complementary_findings(quant_results, qual_results)
        
        # Create joint displays
        joint_displays = self._create_joint_displays(quant_results, qual_results)
        
        # Generate meta-inferences
        meta_inferences = self._generate_meta_inferences(
            convergent_findings, divergent_findings, complementary_findings
        )
        
        # Build integrated model
        integrated_model = self._build_integrated_model(quant_results, qual_results)
        
        # Assess integration quality
        quality_indicators = self._assess_integration_quality(
            convergent_findings, divergent_findings, complementary_findings
        )
        
        result = IntegrationResult(
            design_type=MixedMethodsDesign.CONVERGENT_PARALLEL,
            integration_level=integration_level,
            convergent_findings=convergent_findings,
            divergent_findings=divergent_findings,
            complementary_findings=complementary_findings,
            meta_inferences=meta_inferences,
            joint_displays=joint_displays,
            integrated_model=integrated_model,
            quality_indicators=quality_indicators
        )
        
        self.integration_results["convergent_parallel"] = result
        return result
    
    def integrate_explanatory_sequential(self) -> IntegrationResult:
        """
        Integrate using explanatory sequential design
        Quantitative first, then qualitative to explain quantitative results
        """
        
        if not self.quantitative_data or not self.qualitative_data:
            raise ValueError("Both quantitative and qualitative data required")
        
        # Use the complete implementation from mixed_methods_complete.py
        from .mixed_methods_complete import CompleteMixedMethodsIntegrator
        complete_integrator = CompleteMixedMethodsIntegrator()
        return complete_integrator.integrate_explanatory_sequential()