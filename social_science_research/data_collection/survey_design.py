# -*- coding: utf-8 -*-
"""
Survey Design Module for Social Science Research
Comprehensive survey design following social science best practices
"""

import json
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np

class QuestionType(Enum):
    """Types of survey questions"""
    LIKERT_SCALE = "likert_scale"
    MULTIPLE_CHOICE = "multiple_choice"
    OPEN_ENDED = "open_ended"
    RANKING = "ranking"
    SEMANTIC_DIFFERENTIAL = "semantic_differential"
    MATRIX = "matrix"
    DEMOGRAPHIC = "demographic"

class ScaleType(Enum):
    """Types of measurement scales"""
    NOMINAL = "nominal"
    ORDINAL = "ordinal"
    INTERVAL = "interval"
    RATIO = "ratio"

@dataclass
class SurveyQuestion:
    """Individual survey question structure"""
    id: str
    text: str
    question_type: QuestionType
    scale_type: ScaleType
    options: List[str]
    required: bool
    construct: str  # What theoretical construct this measures
    reverse_coded: bool = False
    validation_rules: Dict[str, Any] = None

@dataclass
class SurveyConstruct:
    """Theoretical construct measured by survey"""
    name: str
    definition: str
    dimensions: List[str]
    questions: List[str]  # Question IDs
    reliability_target: float  # Target Cronbach's alpha
    validity_evidence: List[str]

class SurveyDesigner:
    """Comprehensive survey design system"""
    
    def __init__(self):
        """Initialize survey designer"""
        self.questions = {}
        self.constructs = {}
        self.survey_structure = {}
        
    def create_likert_question(self, question_id: str, text: str, construct: str,
                             scale_points: int = 5, reverse_coded: bool = False) -> SurveyQuestion:
        """Create Likert scale question"""
        
        scale_labels = {
            5: ["Strongly Disagree", "Disagree", "Neutral", "Agree", "Strongly Agree"],
            7: ["Strongly Disagree", "Disagree", "Somewhat Disagree", "Neutral", 
                "Somewhat Agree", "Agree", "Strongly Agree"]
        }
        
        question = SurveyQuestion(
            id=question_id,
            text=text,
            question_type=QuestionType.LIKERT_SCALE,
            scale_type=ScaleType.ORDINAL,
            options=scale_labels.get(scale_points, scale_labels[5]),
            required=True,
            construct=construct,
            reverse_coded=reverse_coded,
            validation_rules={"min": 1, "max": scale_points}
        )
        
        self.questions[question_id] = question
        return question
    
    def create_demographic_question(self, question_id: str, text: str, 
                                  options: List[str], required: bool = False) -> SurveyQuestion:
        """Create demographic question"""
        
        question = SurveyQuestion(
            id=question_id,
            text=text,
            question_type=QuestionType.DEMOGRAPHIC,
            scale_type=ScaleType.NOMINAL,
            options=options,
            required=required,
            construct="demographics"
        )
        
        self.questions[question_id] = question
        return question
    
    def create_construct(self, name: str, definition: str, dimensions: List[str],
                        reliability_target: float = 0.8) -> SurveyConstruct:
        """Create theoretical construct"""
        
        construct = SurveyConstruct(
            name=name,
            definition=definition,
            dimensions=dimensions,
            questions=[],
            reliability_target=reliability_target,
            validity_evidence=[]
        )
        
        self.constructs[name] = construct
        return construct
    
    def add_question_to_construct(self, question_id: str, construct_name: str):
        """Add question to construct"""
        if construct_name in self.constructs and question_id in self.questions:
            self.constructs[construct_name].questions.append(question_id)
    
    def design_social_identity_scale(self) -> List[SurveyQuestion]:
        """Design social identity measurement scale"""
        
        # Create social identity construct
        self.create_construct(
            name="social_identity",
            definition="Degree of identification with social groups",
            dimensions=["identification", "commitment", "belonging"],
            reliability_target=0.85
        )
        
        questions = []
        
        # Identification dimension
        q1 = self.create_likert_question(
            "si_01", "I identify strongly with my cultural group", "social_identity"
        )
        questions.append(q1)
        
        q2 = self.create_likert_question(
            "si_02", "Being a member of my cultural group is important to me", "social_identity"
        )
        questions.append(q2)
        
        # Commitment dimension
        q3 = self.create_likert_question(
            "si_03", "I am committed to my cultural group", "social_identity"
        )
        questions.append(q3)
        
        # Belonging dimension
        q4 = self.create_likert_question(
            "si_04", "I feel a sense of belonging to my cultural group", "social_identity"
        )
        questions.append(q4)
        
        # Add questions to construct
        for q in questions:
            self.add_question_to_construct(q.id, "social_identity")
        
        return questions
    
    def design_cultural_adaptation_scale(self) -> List[SurveyQuestion]:
        """Design cultural adaptation measurement scale"""
        
        # Create cultural adaptation construct
        self.create_construct(
            name="cultural_adaptation",
            definition="Degree of adaptation to new cultural environment",
            dimensions=["behavioral", "cognitive", "affective"],
            reliability_target=0.80
        )
        
        questions = []
        
        # Behavioral adaptation
        q1 = self.create_likert_question(
            "ca_01", "I have adapted my behavior to fit the local culture", "cultural_adaptation"
        )
        questions.append(q1)
        
        q2 = self.create_likert_question(
            "ca_02", "I follow local customs and practices", "cultural_adaptation"
        )
        questions.append(q2)
        
        # Cognitive adaptation
        q3 = self.create_likert_question(
            "ca_03", "I understand the local way of thinking", "cultural_adaptation"
        )
        questions.append(q3)
        
        # Affective adaptation
        q4 = self.create_likert_question(
            "ca_04", "I feel comfortable in this cultural environment", "cultural_adaptation"
        )
        questions.append(q4)
        
        # Add questions to construct
        for q in questions:
            self.add_question_to_construct(q.id, "cultural_adaptation")
        
        return questions
    
    def design_demographics_section(self) -> List[SurveyQuestion]:
        """Design demographics section"""
        
        questions = []
        
        # Age
        age_q = self.create_demographic_question(
            "demo_age", "What is your age?", 
            ["18-25", "26-35", "36-45", "46-55", "56-65", "65+"]
        )
        questions.append(age_q)
        
        # Gender
        gender_q = self.create_demographic_question(
            "demo_gender", "What is your gender?",
            ["Male", "Female", "Non-binary", "Prefer not to say"]
        )
        questions.append(gender_q)
        
        # Education
        education_q = self.create_demographic_question(
            "demo_education", "What is your highest level of education?",
            ["High School", "Bachelor's Degree", "Master's Degree", "Doctoral Degree", "Other"]
        )
        questions.append(education_q)
        
        # Cultural background
        culture_q = self.create_demographic_question(
            "demo_culture", "What is your cultural background?",
            ["Western", "East Asian", "Latin American", "African", "Middle Eastern", "Other"]
        )
        questions.append(culture_q)
        
        return questions
    
    def validate_survey_design(self) -> Dict[str, Any]:
        """Validate survey design"""
        
        validation_results = {
            "total_questions": len(self.questions),
            "constructs": len(self.constructs),
            "construct_coverage": {},
            "question_distribution": {},
            "potential_issues": []
        }
        
        # Check construct coverage
        for construct_name, construct in self.constructs.items():
            question_count = len(construct.questions)
            validation_results["construct_coverage"][construct_name] = question_count
            
            if question_count < 3:
                validation_results["potential_issues"].append(
                    f"Construct '{construct_name}' has only {question_count} questions (minimum 3 recommended)"
                )
        
        # Check question type distribution
        type_counts = {}
        for question in self.questions.values():
            q_type = question.question_type.value
            type_counts[q_type] = type_counts.get(q_type, 0) + 1
        
        validation_results["question_distribution"] = type_counts
        
        # Check for reverse-coded questions
        reverse_coded_count = sum(1 for q in self.questions.values() if q.reverse_coded)
        if reverse_coded_count == 0:
            validation_results["potential_issues"].append(
                "No reverse-coded questions found (recommended for response bias control)"
            )
        
        return validation_results
    
    def generate_survey_json(self) -> Dict[str, Any]:
        """Generate survey in JSON format"""
        
        survey_data = {
            "metadata": {
                "title": "Social Science Research Survey",
                "version": "1.0",
                "total_questions": len(self.questions),
                "estimated_time": len(self.questions) * 0.5  # 30 seconds per question
            },
            "constructs": {name: asdict(construct) for name, construct in self.constructs.items()},
            "questions": {qid: asdict(question) for qid, question in self.questions.items()},
            "sections": self._organize_sections()
        }
        
        return survey_data
    
    def _organize_sections(self) -> List[Dict[str, Any]]:
        """Organize questions into logical sections"""
        
        sections = []
        
        # Demographics section
        demo_questions = [qid for qid, q in self.questions.items() 
                         if q.question_type == QuestionType.DEMOGRAPHIC]
        if demo_questions:
            sections.append({
                "title": "Background Information",
                "description": "Please provide some basic information about yourself",
                "questions": demo_questions
            })
        
        # Construct-based sections
        for construct_name, construct in self.constructs.items():
            if construct.questions and construct_name != "demographics":
                sections.append({
                    "title": construct.name.replace("_", " ").title(),
                    "description": construct.definition,
                    "questions": construct.questions
                })
        
        return sections
    
    def calculate_reliability_estimates(self, response_data: pd.DataFrame) -> Dict[str, float]:
        """Calculate reliability estimates for constructs"""
        
        reliability_estimates = {}
        
        for construct_name, construct in self.constructs.items():
            if len(construct.questions) >= 2:
                # Get responses for construct questions
                construct_responses = response_data[construct.questions]
                
                # Calculate Cronbach's alpha
                alpha = self._calculate_cronbachs_alpha(construct_responses)
                reliability_estimates[construct_name] = alpha
        
        return reliability_estimates
    
    def _calculate_cronbachs_alpha(self, data: pd.DataFrame) -> float:
        """Calculate Cronbach's alpha reliability coefficient"""
        
        # Number of items
        k = data.shape[1]
        
        # Variance of individual items
        item_variances = data.var(axis=0, ddof=1)
        
        # Variance of sum of items
        sum_variance = data.sum(axis=1).var(ddof=1)
        
        # Cronbach's alpha formula
        alpha = (k / (k - 1)) * (1 - (item_variances.sum() / sum_variance))
        
        return alpha
    
    def generate_survey_report(self) -> Dict[str, Any]:
        """Generate comprehensive survey design report"""
        
        validation = self.validate_survey_design()
        
        report = {
            "survey_overview": {
                "total_questions": len(self.questions),
                "total_constructs": len(self.constructs),
                "estimated_completion_time": f"{len(self.questions) * 0.5:.1f} minutes"
            },
            "construct_analysis": self._analyze_constructs(),
            "question_analysis": self._analyze_questions(),
            "validation_results": validation,
            "recommendations": self._generate_design_recommendations(validation)
        }
        
        return report
    
    def _analyze_constructs(self) -> Dict[str, Any]:
        """Analyze construct design"""
        
        analysis = {}
        
        for name, construct in self.constructs.items():
            analysis[name] = {
                "definition": construct.definition,
                "dimensions": construct.dimensions,
                "question_count": len(construct.questions),
                "reliability_target": construct.reliability_target,
                "coverage_assessment": "Adequate" if len(construct.questions) >= 3 else "Insufficient"
            }
        
        return analysis
    
    def _analyze_questions(self) -> Dict[str, Any]:
        """Analyze question design"""
        
        type_distribution = {}
        scale_distribution = {}
        
        for question in self.questions.values():
            # Question type distribution
            q_type = question.question_type.value
            type_distribution[q_type] = type_distribution.get(q_type, 0) + 1
            
            # Scale type distribution
            s_type = question.scale_type.value
            scale_distribution[s_type] = scale_distribution.get(s_type, 0) + 1
        
        return {
            "type_distribution": type_distribution,
            "scale_distribution": scale_distribution,
            "reverse_coded_count": sum(1 for q in self.questions.values() if q.reverse_coded),
            "required_count": sum(1 for q in self.questions.values() if q.required)
        }
    
    def _generate_design_recommendations(self, validation: Dict[str, Any]) -> List[str]:
        """Generate design recommendations"""
        
        recommendations = []
        
        # Based on validation issues
        for issue in validation["potential_issues"]:
            if "reverse-coded" in issue:
                recommendations.append("Add reverse-coded questions to control for response bias")
            elif "questions" in issue and "minimum" in issue:
                recommendations.append("Increase number of questions for constructs with insufficient coverage")
        
        # General recommendations
        recommendations.extend([
            "Pilot test survey with small sample before full deployment",
            "Consider cultural adaptation of questions for cross-cultural research",
            "Include attention check questions for online surveys",
            "Randomize question order within constructs to reduce order effects"
        ])
        
        return recommendations
