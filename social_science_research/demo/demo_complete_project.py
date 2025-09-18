# -*- coding: utf-8 -*-
"""
Complete Social Science Research Project Demo
Demonstrates the full workflow of the social science research framework
"""

import sys
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Any

# Add project paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.social_science_framework import SocialScienceFramework, ResearchParadigm, ResearchMethod
from methodologies.mixed_methods_complete import CompleteMixedMethodsIntegrator, MixedMethodsDesign
from data_collection.survey_design import SurveyDesigner
from analysis.stats import SocialScienceStats

class CompleteSocialScienceProject:
    """Complete social science research project demonstration"""
    
    def __init__(self):
        """Initialize the complete project"""
        self.framework = SocialScienceFramework()
        self.survey_designer = SurveyDesigner()
        self.mixed_methods = CompleteMixedMethodsIntegrator()
        self.stats = SocialScienceStats()
        
    def run_complete_demo(self):
        """Run complete social science research demo"""
        
        print("🔬 COMPLETE SOCIAL SCIENCE RESEARCH PROJECT DEMO")
        print("=" * 60)
        
        # Step 1: Design Research Study
        print("\n📋 STEP 1: RESEARCH DESIGN")
        research_design = self.design_research_study()
        print(f"✅ Created research design: {research_design.title}")
        print(f"   Paradigm: {research_design.paradigm.value}")
        print(f"   Method: {research_design.method.value}")
        print(f"   Sample size: {research_design.sample_size}")
        
        # Step 2: Design Survey Instrument
        print("\n📝 STEP 2: SURVEY DESIGN")
        survey_results = self.design_survey_instrument()
        print(f"✅ Created survey with {survey_results['total_questions']} questions")
        print(f"   Constructs: {', '.join(survey_results['constructs'])}")
        
        # Step 3: Generate Simulated Data
        print("\n📊 STEP 3: DATA GENERATION")
        data = self.generate_simulated_data()
        print(f"✅ Generated data: {len(data)} participants")
        print(f"   Variables: {len(data.columns)} total")
        
        # Step 4: Statistical Analysis
        print("\n📈 STEP 4: STATISTICAL ANALYSIS")
        stats_results = self.conduct_statistical_analysis(data)
        print(f"✅ Completed statistical analysis")
        print(f"   Reliable constructs: {stats_results['summary']['reliable_constructs']}")
        print(f"   Significant correlations: {stats_results['summary']['significant_correlations']}")
        
        # Step 5: Mixed Methods Integration
        print("\n🔄 STEP 5: MIXED METHODS INTEGRATION")
        integration_results = self.conduct_mixed_methods_integration()
        print(f"✅ Completed mixed methods integration")
        print(f"   Design: {integration_results.design_type.value}")
        print(f"   Quality score: {integration_results.quality_indicators['overall_quality']:.2f}")
        
        # Step 6: Cross-Cultural Analysis
        print("\n🌍 STEP 6: CROSS-CULTURAL ANALYSIS")
        cultural_results = self.conduct_cross_cultural_analysis(research_design.title)
        print(f"✅ Completed cross-cultural analysis")
        print(f"   Cultural contexts: {len(cultural_results['cultural_contexts'])}")
        print(f"   Universal theories: {len(cultural_results['cultural_universals'])}")
        
        # Step 7: Generate Final Report
        print("\n📄 STEP 7: FINAL REPORT")
        final_report = self.generate_final_report(
            research_design, stats_results, integration_results, cultural_results
        )
        print(f"✅ Generated comprehensive research report")
        print(f"   Sections: {len(final_report.keys())}")
        
        print("\n🎉 COMPLETE SOCIAL SCIENCE PROJECT DEMO FINISHED!")
        print("=" * 60)
        
        return {
            'research_design': research_design,
            'survey_results': survey_results,
            'data': data,
            'statistical_analysis': stats_results,
            'mixed_methods': integration_results,
            'cross_cultural': cultural_results,
            'final_report': final_report
        }
    
    def design_research_study(self):
        """Design comprehensive research study"""
        
        # Create research design
        design = self.framework.create_research_design(
            title="Cross-Cultural Technology Adaptation Study",
            paradigm=ResearchParadigm.PRAGMATIST,
            method=ResearchMethod.MIXED_METHODS,
            theory_names=["social_identity_theory", "cultural_dimensions_theory", "social_capital_theory"],
            research_questions=[
                "How does cultural background influence technology adaptation?",
                "What role does social identity play in technology acceptance?",
                "How do social networks facilitate cultural adaptation?"
            ],
            cultural_contexts=["western_individualistic", "east_asian_collectivistic", "latin_american"]
        )
        
        return design
    
    def design_survey_instrument(self):
        """Design comprehensive survey instrument"""
        
        # Design social identity scale
        si_questions = self.survey_designer.design_social_identity_scale()
        
        # Design cultural adaptation scale
        ca_questions = self.survey_designer.design_cultural_adaptation_scale()
        
        # Design demographics
        demo_questions = self.survey_designer.design_demographics_section()
        
        # Validate design
        validation = self.survey_designer.validate_survey_design()
        
        return {
            'total_questions': len(self.survey_designer.questions),
            'constructs': list(self.survey_designer.constructs.keys()),
            'validation': validation,
            'social_identity_questions': len(si_questions),
            'cultural_adaptation_questions': len(ca_questions),
            'demographic_questions': len(demo_questions)
        }
    
    def generate_simulated_data(self, n_participants: int = 300) -> pd.DataFrame:
        """Generate realistic simulated data"""
        
        np.random.seed(42)  # For reproducibility
        
        # Generate demographic data
        ages = np.random.choice(['18-25', '26-35', '36-45', '46-55', '56-65'], n_participants)
        genders = np.random.choice(['Male', 'Female', 'Non-binary'], n_participants, p=[0.45, 0.50, 0.05])
        cultures = np.random.choice(['Western', 'East Asian', 'Latin American'], n_participants, p=[0.4, 0.35, 0.25])
        
        # Generate scale responses (1-5 Likert scale)
        # Social Identity items (4 items)
        si_base = np.random.normal(3.5, 0.8, n_participants)
        si_01 = np.clip(np.round(si_base + np.random.normal(0, 0.3, n_participants)), 1, 5)
        si_02 = np.clip(np.round(si_base + np.random.normal(0, 0.3, n_participants)), 1, 5)
        si_03 = np.clip(np.round(si_base + np.random.normal(0, 0.3, n_participants)), 1, 5)
        si_04 = np.clip(np.round(si_base + np.random.normal(0, 0.3, n_participants)), 1, 5)
        
        # Cultural Adaptation items (4 items) - correlated with social identity
        ca_base = si_base * 0.6 + np.random.normal(0, 0.6, n_participants)
        ca_01 = np.clip(np.round(ca_base + np.random.normal(0, 0.3, n_participants)), 1, 5)
        ca_02 = np.clip(np.round(ca_base + np.random.normal(0, 0.3, n_participants)), 1, 5)
        ca_03 = np.clip(np.round(ca_base + np.random.normal(0, 0.3, n_participants)), 1, 5)
        ca_04 = np.clip(np.round(ca_base + np.random.normal(0, 0.3, n_participants)), 1, 5)
        
        # Create DataFrame
        data = pd.DataFrame({
            'demo_age': ages,
            'demo_gender': genders,
            'demo_culture': cultures,
            'si_01': si_01.astype(int),
            'si_02': si_02.astype(int),
            'si_03': si_03.astype(int),
            'si_04': si_04.astype(int),
            'ca_01': ca_01.astype(int),
            'ca_02': ca_02.astype(int),
            'ca_03': ca_03.astype(int),
            'ca_04': ca_04.astype(int)
        })
        
        return data
    
    def conduct_statistical_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Conduct comprehensive statistical analysis"""
        
        # Define constructs
        constructs = {
            'social_identity': ['si_01', 'si_02', 'si_03', 'si_04'],
            'cultural_adaptation': ['ca_01', 'ca_02', 'ca_03', 'ca_04']
        }
        
        # Generate comprehensive analysis report
        analysis_report = self.stats.generate_analysis_report(data, constructs)
        
        # Additional analyses
        # T-test comparing cultural groups
        if 'demo_culture' in data.columns:
            # Create binary cultural variable for t-test
            data['western_vs_other'] = (data['demo_culture'] == 'Western').astype(int)
            
            # Calculate construct scores
            data['social_identity_score'] = data[constructs['social_identity']].mean(axis=1)
            data['cultural_adaptation_score'] = data[constructs['cultural_adaptation']].mean(axis=1)
            
            # T-test for social identity
            si_ttest = self.stats.t_test_analysis(data, 'social_identity_score', 'western_vs_other')
            analysis_report['t_test_social_identity'] = si_ttest
            
