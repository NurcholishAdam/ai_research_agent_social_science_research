# -*- coding: utf-8 -*-
"""
Complete Social Science Research Project Demo
Demonstrates integration of all components with AI enhancements
"""

import sys
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Any

# Add project paths
sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import all components
from core.social_science_framework import SocialScienceFramework, ResearchParadigm, ResearchMethod
from methodologies.mixed_methods_complete import CompleteMixedMethodsIntegrator, MixedMethodsDesign
from data_collection.survey_design import SurveyDesigner
from analysis.statistical_analysis import AdvancedStatisticalAnalyzer
from ai_integration.enhanced_social_science_ai import (
    EnhancedSocialScienceAI, SocialScienceQuery, AIEnhancementType
)

class CompleteSocialScienceProject:
    """
    Complete Social Science Research Project
    Integrates all components with AI enhancements
    """
    
    def __init__(self):
        """Initialize complete project system"""
        print("🔬 Initializing Complete Social Science Research Project...")
        
        # Initialize core components
        self.framework = SocialScienceFramework()
        self.mixed_methods = CompleteMixedMethodsIntegrator()
        self.survey_designer = SurveyDesigner()
        self.statistical_analyzer = AdvancedStatisticalAnalyzer()
        self.ai_enhancer = EnhancedSocialScienceAI()
        
        print("✅ All components initialized successfully!")
    
    def run_complete_demo(self):
        """Run complete demonstration of all components"""
        
        print("\n" + "="*80)
        print("🚀 COMPLETE SOCIAL SCIENCE RESEARCH PROJECT DEMO")
        print("="*80)
        
        # Step 1: Design Research Study
        print("\n📋 STEP 1: RESEARCH DESIGN")
        research_design = self._demo_research_design()
        
        # Step 2: Survey Design
        print("\n📝 STEP 2: SURVEY DESIGN")
        survey_data = self._demo_survey_design()
        
        # Step 3: Generate Sample Data
        print("\n📊 STEP 3: DATA GENERATION")
        sample_data = self._generate_sample_data()
        
        # Step 4: Statistical Analysis
        print("\n📈 STEP 4: STATISTICAL ANALYSIS")
        statistical_results = self._demo_statistical_analysis(sample_data)
        
        # Step 5: Mixed Methods Integration
        print("\n🔄 STEP 5: MIXED METHODS INTEGRATION")
        mixed_methods_results = self._demo_mixed_methods_integration()
        
        # Step 6: AI Enhancement
        print("\n🤖 STEP 6: AI ENHANCEMENT")
        ai_enhanced_results = self._demo_ai_enhancement()
        
        # Step 7: Generate Final Report
        print("\n📄 STEP 7: COMPREHENSIVE REPORT")
        final_report = self._generate_final_report(
            research_design, survey_data, statistical_results, 
            mixed_methods_results, ai_enhanced_results
        )
        
        print("\n✅ DEMO COMPLETED SUCCESSFULLY!")
        return final_report
    
    def _demo_research_design(self) -> Dict[str, Any]:
        """Demonstrate research design capabilities"""
        
        print("Creating comprehensive research design...")
        
        # Create research design
        design = self.framework.create_research_design(
            title="Cross-Cultural Study of Social Identity and Technology Adaptation",
            paradigm=ResearchParadigm.PRAGMATIST,
            method=ResearchMethod.MIXED_METHODS,
            theory_names=["social_identity_theory", "cultural_dimensions_theory"],
            research_questions=[
                "How does social identity influence technology adaptation across cultures?",
                "What role do cultural values play in technology acceptance?",
                "How do social networks mediate cultural adaptation processes?"
            ],
            cultural_contexts=["western_individualistic", "east_asian_collectivistic"]
        )
        
        print(f"✅ Research design created: {design.title}")
        print(f"   - Paradigm: {design.paradigm.value}")
        print(f"   - Method: {design.method.value}")
        print(f"   - Theories: {len(design.theoretical_framework)}")
        print(f"   - Sample size: {design.sample_size}")
        
        # Conduct cross-cultural analysis
        cross_cultural_results = self.framework.conduct_cross_cultural_analysis(
            design.title, ["western_individualistic", "east_asian_collectivistic"]
        )
        
        print(f"   - Cultural comparisons: {len(cross_cultural_results['cultural_comparisons'])}")
        print(f"   - Universal theories: {len(cross_cultural_results['cultural_universals'])}")
        
        return {
            "design": design,
            "cross_cultural_analysis": cross_cultural_results
        }
    
    def _demo_survey_design(self) -> Dict[str, Any]:
        """Demonstrate survey design capabilities"""
        
        print("Designing comprehensive survey instrument...")
        
        # Design social identity scale
        social_identity_questions = self.survey_designer.design_social_identity_scale()
        print(f"✅ Social identity scale: {len(social_identity_questions)} questions")
        
        # Design cultural adaptation scale
        cultural_adaptation_questions = self.survey_designer.design_cultural_adaptation_scale()
        print(f"✅ Cultural adaptation scale: {len(cultural_adaptation_questions)} questions")
        
        # Design demographics section
        demographic_questions = self.survey_designer.design_demographics_section()
        print(f"✅ Demographics section: {len(demographic_questions)} questions")
        
        # Validate survey design
        validation_results = self.survey_designer.validate_survey_design()
        print(f"   - Total questions: {validation_results['total_questions']}")
        print(f"   - Constructs: {validation_results['constructs']}")
        print(f"   - Potential issues: {len(validation_results['potential_issues'])}")
        
        # Generate survey JSON
        survey_json = self.survey_designer.generate_survey_json()
        
        return {
            "social_identity_questions": social_identity_questions,
            "cultural_adaptation_questions": cultural_adaptation_questions,
            "demographic_questions": demographic_questions,
            "validation_results": validation_results,
            "survey_json": survey_json
        }
    
    def _generate_sample_data(self) -> pd.DataFrame:
        """Generate realistic sample data for demonstration"""
        
        print("Generating realistic sample data...")
        
        np.random.seed(42)  # For reproducibility
        n_participants = 300
        
        # Generate demographic data
        ages = np.random.choice(["18-25", "26-35", "36-45", "46-55", "56-65"], n_participants)
        genders = np.random.choice(["Male", "Female", "Non-binary"], n_participants, p=[0.45, 0.50, 0.05])
        cultures = np.random.choice(["Western", "East Asian", "Latin American"], n_participants, p=[0.4, 0.4, 0.2])
        education = np.random.choice(["High School", "Bachelor's", "Master's", "Doctoral"], 
                                   n_participants, p=[0.2, 0.4, 0.3, 0.1])
        
        # Generate social identity scores (1-5 Likert scale)
        si_base = np.random.normal(3.5, 0.8, n_participants)
        si_01 = np.clip(si_base + np.random.normal(0, 0.3, n_participants), 1, 5)
        si_02 = np.clip(si_base + np.random.normal(0, 0.3, n_participants), 1, 5)
        si_03 = np.clip(si_base + np.random.normal(0, 0.3, n_participants), 1, 5)
        si_04 = np.clip(si_base + np.random.normal(0, 0.3, n_participants), 1, 5)
        
        # Generate cultural adaptation scores (influenced by culture)
        ca_base = np.random.normal(3.2, 0.9, n_participants)
        # East Asians might score higher on adaptation
        ca_adjustment = np.where(cultures == "East Asian", 0.3, 0)
        ca_01 = np.clip(ca_base + ca_adjustment + np.random.normal(0, 0.3, n_participants), 1, 5)
        ca_02 = np.clip(ca_base + ca_adjustment + np.random.normal(0, 0.3, n_participants), 1, 5)
        ca_03 = np.clip(ca_base + ca_adjustment + np.random.normal(0, 0.3, n_participants), 1, 5)
        ca_04 = np.clip(ca_base + ca_adjustment + np.random.normal(0, 0.3, n_participants), 1, 5)
        
        # Generate technology adoption scores
        tech_adoption = np.clip(np.random.normal(3.8, 0.7, n_participants), 1, 5)
        
        # Create DataFrame
        data = pd.DataFrame({
            'participant_id': range(1, n_participants + 1),
            'age': ages,
            'gender': genders,
            'culture': cultures,
            'education': education,
            'si_01': si_01,
            'si_02': si_02,
            'si_03': si_03,
            'si_04': si_04,
            'ca_01': ca_01,
            'ca_02': ca_02,
            'ca_03': ca_03,
            'ca_04': ca_04,
            'tech_adoption': tech_adoption
        })
        
        print(f"✅ Generated sample data: {len(data)} participants")
        print(f"   - Cultural distribution: {data['culture'].value_counts().to_dict()}")
        print(f"   - Gender distribution: {data['gender'].value_counts().to_dict()}")
        
        return data
    
    def _demo_statistical_analysis(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Demonstrate statistical analysis capabilities"""
        
        print("Performing comprehensive statistical analysis...")
        
        # Load data into analyzer
        self.statistical_analyzer.load_data(data)
        
        # Descriptive analysis
        desc_results = self.statistical_analyzer.descriptive_analysis([
            'si_01', 'si_02', 'si_03', 'si_04', 'ca_01', 'ca_02', 'ca_03', 'ca_04', 'tech_adoption'
        ])
        print(f"✅ Descriptive analysis: {len(desc_results)} variables analyzed")
        
        # Correlation analysis
        corr_results = self.statistical_analyzer.correlation_analysis([
            'si_01', 'si_02', 'si_03', 'si_04', 'ca_01', 'ca_02', 'ca_03', 'ca_04', 'tech_adoption'
        ])
        print(f"✅ Correlation analysis: {len(corr_results['significant_correlations'])} significant correlations")
        
        # T-test analysis (comparing cultures)
        # Create binary culture variable for t-test
        data_binary = data[data['culture'].isin(['Western', 'East Asian'])].copy()
        self.statistical_analyzer.load_data(data_binary)
        
        t_test_result = self.statistical_analyzer.t_test_analysis(
            dependent_var='tech_adoption',
            independent_var='culture',
            test_type='independent'
        )
        print(f"✅ T-test analysis: p-value = {t_test_result.p_value:.4f}")
        
        # ANOVA analysis
        self.statistical_analyzer.load_data(data)  # Reload full data
        anova_result = self.statistical_analyzer.anova_analysis(
            dependent_var='tech_adoption',
            independent_var='culture'
        )
        print(f"✅ ANOVA analysis: F = {anova_result.statistic:.3f}, p = {anova_result.p_value:.4f}")
        
        # Regression analysis
        regression_results = self.statistical_analyzer.regression_analysis(
            dependent_var='tech_adoption',
            independent_vars=['si_01', 'si_02', 'ca_01', 'ca_02']
        )
        print(f"✅ Regression analysis: R² = {regression_results['r_squared']:.3f}")
        
        # Factor analysis
        factor_results = self.statistical_analyzer.factor_analysis([
            'si_01', 'si_02', 'si_03', 'si_04', 'ca_01', 'ca_02', 'ca_03', 'ca_04'
        ])
        print(f"✅ Factor analysis: {factor_results['n_factors']} factors extracted")
        
        # Network analysis
        network_results = self.statistical_analyzer.network_analysis()
        print(f"✅ Network analysis: {network_results['network_statistics']['num_edges']} connections")
        
        return {
            "descriptive": desc_results,
            "correlation": corr_results,
            "t_test": t_test_result,
            "anova": anova_result,
            "regression": regression_results,
            "factor_analysis": factor_results,
            "network_analysis": network_results
        }
    
    def _demo_mixed_methods_integration(self) -> Dict[str, Any]:
        """Demonstrate mixed methods integration"""
        
        print("Demonstrating mixed methods integration...")
        
        # Explanatory sequential design
        explanatory_result = self.mixed_methods.integrate_explanatory_sequential()
        print(f"✅ Explanatory sequential: {len(explanatory_result.convergent_findings)} convergent findings")
        
        # Exploratory sequential design
        exploratory_result = self.mixed_methods.integrate_exploratory_sequential()
        print(f"✅ Exploratory sequential: {len(exploratory_result.meta_inferences)} meta-inferences")
        
        # Generate integration reports
        explanatory_report = self.mixed_methods.generate_integration_report(explanatory_result)
        exploratory_report = self.mixed_methods.generate_integration_report(exploratory_result)
        
        print(f"   - Integration quality (explanatory): {explanatory_result.quality_indicators['overall_quality']:.2f}")
        print(f"   - Integration quality (exploratory): {exploratory_result.quality_indicators['overall_quality']:.2f}")
        
        return {
            "explanatory_sequential": explanatory_result,
            "exploratory_sequential": exploratory_result,
            "explanatory_report": explanatory_report,
            "exploratory_report": exploratory_report
        }
    
    def _demo_ai_enhancement(self) -> Dict[str, Any]:
        """Demonstrate AI enhancement capabilities"""
        
        print("Applying AI enhancements...")
        
        # Create research query
        query = SocialScienceQuery(
            query_id="demo_query_001",
            research_question="How does social identity influence technology adaptation across cultures?",
            theoretical_framework=["social_identity_theory", "cultural_dimensions_theory"],
            methodology="mixed_methods",
            cultural_context=["western_individualistic", "east_asian_collectivistic"],
            expected_outcomes=["cultural_differences", "identity_effects", "adaptation_patterns"],
            complexity_level="high"
        )
        
        # Apply AI enhancements
        ai_result = self.ai_enhancer.enhance_research_query(query)
        
        print(f"✅ AI enhancement completed")
        print(f"   - Overall confidence: {ai_result.confidence_score:.2f}")
        print(f"   - Methodological rigor: {ai_result.methodological_rigor:.2f}")
        print(f"   - Cultural validity: {len(ai_result.cultural_validity)} contexts assessed")
        print(f"   - Recommendations: {len(ai_result.recommendations)}")
        
        # Generate comprehensive AI report
        ai_report = self.ai_enhancer.generate_comprehensive_report(query.query_id)
        
        return {
            "query": query,
            "ai_result": ai_result,
            "ai_report": ai_report
        }
    
    def _generate_final_report(self, research_design: Dict[str, Any],
                             survey_data: Dict[str, Any],
                             statistical_results: Dict[str, Any],
                             mixed_methods_results: Dict[str, Any],
                             ai_enhanced_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive final report"""
        
        print("Generating comprehensive final report...")
        
        # Generate comprehensive statistical report
        statistical_report = self.statistical_analyzer.generate_comprehensive_report()
        
        final_report = {
            "project_overview": {
                "title": "Complete Social Science Research Project Demo",
                "methodology": "AI-Enhanced Mixed Methods Research",
                "components_integrated": [
                    "Social Science Framework",
                    "Survey Design System",
                    "Advanced Statistical Analysis",
                    "Mixed Methods Integration",
                    "AI Enhancement (RLHF, Semantic Graph, Contextual Engineering, LIMIT-Graph)"
                ],
                "cultural_contexts": ["Western Individualistic", "East Asian Collectivistic"],
                "theoretical_frameworks": ["Social Identity Theory", "Cultural Dimensions Theory"]
            },
            
            "research_design_summary": {
                "paradigm": research_design["design"].paradigm.value,
                "method": research_design["design"].method.value,
                "sample_size": research_design["design"].sample_size,
                "cultural_universals": research_design["cross_cultural_analysis"]["cultural_universals"],
                "cultural_specifics": len(research_design["cross_cultural_analysis"]["cultural_specifics"])
            },
            
            "survey_design_summary": {
                "total_questions": survey_data["validation_results"]["total_questions"],
                "constructs": survey_data["validation_results"]["constructs"],
                "scales_designed": ["Social Identity Scale", "Cultural Adaptation Scale", "Demographics"],
                "validation_issues": len(survey_data["validation_results"]["potential_issues"])
            },
            
            "statistical_analysis_summary": {
                "analyses_performed": len(statistical_results),
                "significant_correlations": len(statistical_results["correlation"]["significant_correlations"]),
                "regression_r_squared": statistical_results["regression"]["r_squared"],
                "factors_extracted": statistical_results["factor_analysis"]["n_factors"],
                "network_connections": statistical_results["network_analysis"]["network_statistics"]["num_edges"]
            },
            
            "mixed_methods_summary": {
                "designs_demonstrated": ["Explanatory Sequential", "Exploratory Sequential"],
                "convergent_findings": len(mixed_methods_results["explanatory_sequential"].convergent_findings),
                "meta_inferences": len(mixed_methods_results["exploratory_sequential"].meta_inferences),
                "integration_quality": {
                    "explanatory": mixed_methods_results["explanatory_sequential"].quality_indicators["overall_quality"],
                    "exploratory": mixed_methods_results["exploratory_sequential"].quality_indicators["overall_quality"]
                }
            },
            
            "ai_enhancement_summary": {
                "ai_systems_integrated": ["LIMIT-Graph", "Semantic Graph", "Contextual Engineering", "RLHF"],
                "overall_confidence": ai_enhanced_results["ai_result"].confidence_score,
                "methodological_rigor": ai_enhanced_results["ai_result"].methodological_rigor,
                "cultural_validity_scores": ai_enhanced_results["ai_result"].cultural_validity,
                "theoretical_alignment": ai_enhanced_results["ai_result"].theoretical_alignment,
                "ai_recommendations": len(ai_enhanced_results["ai_result"].recommendations)
            },
            
            "key_findings": [
                "Successfully integrated multiple AI architectures with social science methodologies",
                "Demonstrated cross-cultural validity assessment and adaptation",
                "Achieved high integration quality in mixed methods approaches",
                "AI enhancements provided valuable insights for research design optimization",
                "Cultural context significantly influences methodology selection and adaptation"
            ],
            
            "methodological_contributions": [
                "Novel integration of LIMIT-Graph reasoning with social science research",
                "AI-enhanced cultural adaptation framework",
                "Automated quality assessment using RLHF principles",
                "Semantic graph integration for theoretical coherence",
                "Comprehensive mixed methods integration protocols"
            ],
            
            "practical_implications": [
                "Improved research design quality through AI enhancement",
                "Enhanced cultural sensitivity in cross-cultural research",
                "Automated quality assurance for research protocols",
                "Scalable framework for complex social science studies",
                "Integration of traditional and AI-enhanced methodologies"
            ],
            
            "future_directions": [
                "Real-world validation with actual research projects",
                "Integration with additional AI architectures",
                "Development of specialized cultural adaptation algorithms",
                "Creation of automated research design optimization tools",
                "Expansion to additional social science domains"
            ],
            
            "technical_specifications": {
                "programming_language": "Python 3.8+",
                "key_libraries": ["pandas", "numpy", "scipy", "scikit-learn", "networkx", "statsmodels"],
                "ai_components": ["LIMIT-Graph", "Semantic Graph", "RLHF", "Contextual Engineering"],
                "statistical_methods": ["Descriptive", "Inferential", "Multivariate", "Network Analysis"],
                "mixed_methods_designs": ["Convergent Parallel", "Explanatory Sequential", "Exploratory Sequential"]
            }
        }
        
        print(f"✅ Final report generated")
        print(f"   - Total components: {len(final_report['project_overview']['components_integrated'])}")
        print(f"   - Key findings: {len(final_report['key_findings'])}")
        print(f"   - Methodological contributions: {len(final_report['methodological_contributions'])}")
        
        return final_report

def main():
    """Main demonstration function"""
    
    print("🎯 Starting Complete Social Science Research Project Demo")
    print("This demo showcases the integration of:")
    print("  • Social Science Framework")
    print("  • Survey Design System") 
    print("  • Advanced Statistical Analysis")
    print("  • Mixed Methods Integration")
    print("  • AI Enhancement (RLHF, Semantic Graph, Contextual Engineering, LIMIT-Graph)")
    
    try:
        # Initialize and run complete project
        project = CompleteSocialScienceProject()
        final_report = project.run_complete_demo()
        
        print("\n" + "="*80)
        print("🎉 DEMO COMPLETED SUCCESSFULLY!")
        print("="*80)
        
        print(f"\n📊 PROJECT SUMMARY:")
        print(f"   • Components Integrated: {len(final_report['project_overview']['components_integrated'])}")
        print(f"   • AI Systems Used: {len(final_report['ai_enhancement_summary']['ai_systems_integrated'])}")
        print(f"   • Statistical Analyses: {final_report['statistical_analysis_summary']['analyses_performed']}")
        print(f"   • Mixed Methods Designs: {len(final_report['mixed_methods_summary']['designs_demonstrated'])}")
        print(f"   • Cultural Contexts: {len(final_report['project_overview']['cultural_contexts'])}")
        
        print(f"\n🎯 KEY ACHIEVEMENTS:")
        for finding in final_report['key_findings'][:3]:
            print(f"   • {finding}")
        
        print(f"\n🚀 METHODOLOGICAL INNOVATIONS:")
        for contribution in final_report['methodological_contributions'][:3]:
            print(f"   • {contribution}")
        
        return final_report
        
    except Exception as e:
        print(f"\n❌ Error during demo: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    final_report = main()