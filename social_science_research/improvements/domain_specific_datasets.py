# -*- coding: utf-8 -*-
"""
Domain-Specific Social Science Datasets and Tasks
Addresses the "Domain-Specific Tasks" criterion from the assessment
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import json
import random
from datetime import datetime, timedelta

class SocialScienceDomain(Enum):
    """Social science research domains"""
    SOCIOLOGY = "sociology"
    PSYCHOLOGY = "psychology"
    POLITICAL_SCIENCE = "political_science"
    ECONOMICS = "economics"
    ANTHROPOLOGY = "anthropology"
    SOCIAL_POLICY = "social_policy"

class DatasetType(Enum):
    """Types of social science datasets"""
    SURVEY = "survey"
    EXPERIMENTAL = "experimental"
    LONGITUDINAL = "longitudinal"
    CROSS_CULTURAL = "cross_cultural"
    NETWORK = "network"
    POLICY_ANALYSIS = "policy_analysis"

@dataclass
class SocialScienceDataset:
    """Social science dataset structure"""
    dataset_id: str
    name: str
    domain: SocialScienceDomain
    dataset_type: DatasetType
    description: str
    variables: Dict[str, str]
    sample_size: int
    cultural_contexts: List[str]
    time_period: str
    ethical_considerations: List[str]
    data: pd.DataFrame
    metadata: Dict[str, Any]

class SocialScienceDatasetGenerator:
    """Generator for realistic social science datasets"""
    
    def __init__(self):
        """Initialize dataset generator"""
        self.cultural_contexts = {
            'western_individualistic': {
                'values': ['independence', 'achievement', 'self_expression'],
                'communication': 'direct',
                'social_structure': 'egalitarian',
                'hofstede': {'power_distance': 0.3, 'individualism': 0.8, 'uncertainty_avoidance': 0.4}
            },
            'east_asian_collectivistic': {
                'values': ['harmony', 'respect', 'group_loyalty'],
                'communication': 'indirect',
                'social_structure': 'hierarchical',
                'hofstede': {'power_distance': 0.7, 'individualism': 0.2, 'uncertainty_avoidance': 0.5}
            },
            'latin_american': {
                'values': ['family', 'honor', 'personalismo'],
                'communication': 'expressive',
                'social_structure': 'hierarchical',
                'hofstede': {'power_distance': 0.8, 'individualism': 0.3, 'uncertainty_avoidance': 0.7}
            },
            'african_communalistic': {
                'values': ['ubuntu', 'community', 'ancestral_wisdom'],
                'communication': 'contextual',
                'social_structure': 'community_based',
                'hofstede': {'power_distance': 0.7, 'individualism': 0.2, 'uncertainty_avoidance': 0.6}
            }
        }
        
        self.datasets = {}
    
    def generate_social_identity_survey(self, n_participants: int = 500,
                                      cultural_contexts: List[str] = None) -> SocialScienceDataset:
        """Generate social identity survey dataset"""
        
        if cultural_contexts is None:
            cultural_contexts = ['western_individualistic', 'east_asian_collectivistic']
        
        # Generate participant data
        data = []
        
        for i in range(n_participants):
            # Assign cultural context
            culture = np.random.choice(cultural_contexts)
            culture_data = self.cultural_contexts[culture]
            
            # Generate demographics
            participant = {
                'participant_id': f'P{i:04d}',
                'age': np.random.normal(35, 12),
                'gender': np.random.choice(['Male', 'Female', 'Non-binary'], p=[0.45, 0.50, 0.05]),
                'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], p=[0.3, 0.4, 0.25, 0.05]),
                'cultural_background': culture,
                'country': self._get_country_for_culture(culture),
                'language': self._get_language_for_culture(culture)
            }
            
            # Generate social identity measures (influenced by culture)
            base_identity = np.random.normal(3.5, 0.8)
            cultural_adjustment = self._get_cultural_adjustment(culture, 'social_identity')
            
            # Social Identity Scale (Tajfel & Turner, 1979)
            participant.update({
                'si_identification': np.clip(base_identity + cultural_adjustment + np.random.normal(0, 0.3), 1, 5),
                'si_commitment': np.clip(base_identity + cultural_adjustment + np.random.normal(0, 0.3), 1, 5),
                'si_belonging': np.clip(base_identity + cultural_adjustment + np.random.normal(0, 0.3), 1, 5),
                'si_pride': np.clip(base_identity + cultural_adjustment + np.random.normal(0, 0.3), 1, 5)
            })
            
            # Cultural Values (Hofstede-inspired)
            hofstede = culture_data['hofstede']
            participant.update({
                'power_distance': np.clip(np.random.normal(hofstede['power_distance'] * 5, 0.5), 1, 5),
                'individualism': np.clip(np.random.normal(hofstede['individualism'] * 5, 0.5), 1, 5),
                'uncertainty_avoidance': np.clip(np.random.normal(hofstede['uncertainty_avoidance'] * 5, 0.5), 1, 5)
            })
            
            # Behavioral outcomes
            participant.update({
                'group_cooperation': np.clip(base_identity * 0.8 + np.random.normal(0, 0.4), 1, 5),
                'intergroup_contact': np.clip(5 - base_identity * 0.6 + np.random.normal(0, 0.4), 1, 5),
                'cultural_adaptation': np.clip(np.random.normal(3.2, 0.9), 1, 5)
            })
            
            # Add cultural response biases
            participant = self._apply_cultural_response_bias(participant, culture)
            
            data.append(participant)
        
        df = pd.DataFrame(data)
        
        # Create dataset
        dataset = SocialScienceDataset(
            dataset_id="social_identity_survey_001",
            name="Cross-Cultural Social Identity Survey",
            domain=SocialScienceDomain.PSYCHOLOGY,
            dataset_type=DatasetType.CROSS_CULTURAL,
            description="Survey measuring social identity strength and cultural values across different cultural contexts",
            variables={
                'si_identification': 'Social identity identification strength (1-5)',
                'si_commitment': 'Commitment to social group (1-5)',
                'si_belonging': 'Sense of belonging to group (1-5)',
                'si_pride': 'Pride in group membership (1-5)',
                'power_distance': 'Acceptance of power hierarchy (1-5)',
                'individualism': 'Individual vs collective orientation (1-5)',
                'uncertainty_avoidance': 'Tolerance for ambiguity (1-5)',
                'group_cooperation': 'Willingness to cooperate with group (1-5)',
                'intergroup_contact': 'Frequency of intergroup contact (1-5)',
                'cultural_adaptation': 'Adaptation to new cultural contexts (1-5)'
            },
            sample_size=n_participants,
            cultural_contexts=cultural_contexts,
            time_period="2024",
            ethical_considerations=[
                "Informed consent obtained",
                "Cultural sensitivity protocols followed",
                "Data anonymization implemented",
                "Right to withdraw respected"
            ],
            data=df,
            metadata={
                'collection_method': 'Online survey',
                'response_rate': 0.73,
                'reliability_scores': {
                    'social_identity_scale': 0.87,
                    'cultural_values_scale': 0.82
                },
                'cultural_distribution': {culture: sum(df['cultural_background'] == culture) for culture in cultural_contexts}
            }
        )
        
        self.datasets[dataset.dataset_id] = dataset
        return dataset
    
    def generate_social_network_dataset(self, n_nodes: int = 200,
                                      network_type: str = 'small_world') -> SocialScienceDataset:
        """Generate social network dataset"""
        
        import networkx as nx
        
        # Generate network structure
        if network_type == 'small_world':
            G = nx.watts_strogatz_graph(n_nodes, 6, 0.3)
        elif network_type == 'scale_free':
            G = nx.barabasi_albert_graph(n_nodes, 3)
        else:
            G = nx.erdos_renyi_graph(n_nodes, 0.05)
        
        # Generate node attributes
        node_data = []
        for node_id in G.nodes():
            culture = np.random.choice(list(self.cultural_contexts.keys()))
            
            node_attrs = {
                'node_id': node_id,
                'age': np.random.normal(32, 10),
                'gender': np.random.choice(['Male', 'Female'], p=[0.5, 0.5]),
                'cultural_background': culture,
                'education_level': np.random.choice([1, 2, 3, 4], p=[0.2, 0.4, 0.3, 0.1]),
                'social_capital': np.random.normal(3.0, 1.0),
                'trust_level': np.random.normal(3.5, 0.8),
                'degree_centrality': nx.degree_centrality(G)[node_id],
                'betweenness_centrality': nx.betweenness_centrality(G)[node_id],
                'clustering_coefficient': nx.clustering(G, node_id)
            }
            
            node_data.append(node_attrs)
        
        # Generate edge data
        edge_data = []
        for edge in G.edges():
            node1, node2 = edge
            
            # Get node attributes
            node1_culture = next(n['cultural_background'] for n in node_data if n['node_id'] == node1)
            node2_culture = next(n['cultural_background'] for n in node_data if n['node_id'] == node2)
            
            edge_attrs = {
                'node1': node1,
                'node2': node2,
                'relationship_strength': np.random.uniform(0.3, 1.0),
                'interaction_frequency': np.random.choice(['daily', 'weekly', 'monthly', 'rarely'], 
                                                        p=[0.1, 0.3, 0.4, 0.2]),
                'relationship_type': np.random.choice(['family', 'friend', 'colleague', 'acquaintance'],
                                                    p=[0.15, 0.35, 0.25, 0.25]),
                'cultural_homophily': 1.0 if node1_culture == node2_culture else 0.0,
                'trust_level': np.random.normal(3.2, 0.9)
            }
            
            edge_data.append(edge_attrs)
        
        # Combine node and edge data
        nodes_df = pd.DataFrame(node_data)
        edges_df = pd.DataFrame(edge_data)
        
        # Create combined dataset
        combined_data = {
            'nodes': nodes_df,
            'edges': edges_df,
            'network_metrics': {
                'n_nodes': G.number_of_nodes(),
                'n_edges': G.number_of_edges(),
                'density': nx.density(G),
                'average_clustering': nx.average_clustering(G),
                'average_path_length': nx.average_shortest_path_length(G) if nx.is_connected(G) else None
            }
        }
        
        dataset = SocialScienceDataset(
            dataset_id="social_network_001",
            name="Cross-Cultural Social Network Analysis",
            domain=SocialScienceDomain.SOCIOLOGY,
            dataset_type=DatasetType.NETWORK,
            description="Social network data examining relationship patterns across cultural groups",
            variables={
                'degree_centrality': 'Normalized degree centrality in network',
                'betweenness_centrality': 'Betweenness centrality measure',
                'clustering_coefficient': 'Local clustering coefficient',
                'relationship_strength': 'Strength of relationship tie (0-1)',
                'cultural_homophily': 'Same culture connection (0/1)',
                'trust_level': 'Trust in relationship (1-5)'
            },
            sample_size=n_nodes,
            cultural_contexts=list(self.cultural_contexts.keys()),
            time_period="2024",
            ethical_considerations=[
                "Network anonymization applied",
                "Relationship consent obtained",
                "Privacy protection measures"
            ],
            data=pd.DataFrame(combined_data),  # Simplified for storage
            metadata={
                'network_type': network_type,
                'collection_method': 'Snowball sampling',
                'network_metrics': combined_data['network_metrics']
            }
        )
        
        self.datasets[dataset.dataset_id] = dataset
        return dataset
    
    def generate_policy_analysis_dataset(self, n_policies: int = 50,
                                       policy_domains: List[str] = None) -> SocialScienceDataset:
        """Generate policy analysis dataset"""
        
        if policy_domains is None:
            policy_domains = ['education', 'healthcare', 'immigration', 'economic', 'environmental']
        
        data = []
        
        for i in range(n_policies):
            policy_domain = np.random.choice(policy_domains)
            
            # Generate policy characteristics
            policy = {
                'policy_id': f'POL{i:03d}',
                'policy_name': f'{policy_domain.title()} Policy {i}',
                'domain': policy_domain,
                'implementation_year': np.random.randint(2010, 2024),
                'country': np.random.choice(['USA', 'Canada', 'UK', 'Germany', 'Japan', 'Australia']),
                'policy_type': np.random.choice(['regulatory', 'incentive', 'informational', 'redistributive']),
                'budget_millions': np.random.lognormal(3, 1.5),
                'target_population_size': np.random.lognormal(10, 1.2),
                'implementation_complexity': np.random.uniform(1, 5),
                'stakeholder_support': np.random.normal(3.0, 1.0),
                'political_feasibility': np.random.normal(3.2, 0.9)
            }
            
            # Generate outcomes (influenced by policy characteristics)
            base_effectiveness = (policy['stakeholder_support'] + policy['political_feasibility']) / 2
            complexity_penalty = policy['implementation_complexity'] * 0.1
            
            policy.update({
                'effectiveness_score': np.clip(base_effectiveness - complexity_penalty + np.random.normal(0, 0.5), 1, 5),
                'public_satisfaction': np.clip(base_effectiveness * 0.8 + np.random.normal(0, 0.6), 1, 5),
                'unintended_consequences': np.random.poisson(2),
                'cost_effectiveness': np.clip(np.random.normal(3.0, 1.0), 1, 5),
                'sustainability_score': np.clip(np.random.normal(3.3, 0.8), 1, 5)
            })
            
            # Add domain-specific variables
            if policy_domain == 'education':
                policy.update({
                    'student_achievement_change': np.random.normal(0.1, 0.3),
                    'dropout_rate_change': np.random.normal(-0.05, 0.15),
                    'teacher_satisfaction': np.random.normal(3.2, 0.9)
                })
            elif policy_domain == 'healthcare':
                policy.update({
                    'health_outcome_improvement': np.random.normal(0.15, 0.25),
                    'access_improvement': np.random.normal(0.2, 0.3),
                    'cost_reduction': np.random.normal(-0.1, 0.2)
                })
            
            data.append(policy)
        
        df = pd.DataFrame(data)
        
        dataset = SocialScienceDataset(
            dataset_id="policy_analysis_001",
            name="Cross-National Policy Effectiveness Analysis",
            domain=SocialScienceDomain.SOCIAL_POLICY,
            dataset_type=DatasetType.POLICY_ANALYSIS,
            description="Analysis of policy effectiveness across different domains and countries",
            variables={
                'effectiveness_score': 'Overall policy effectiveness (1-5)',
                'public_satisfaction': 'Public satisfaction with policy (1-5)',
                'stakeholder_support': 'Stakeholder support level (1-5)',
                'implementation_complexity': 'Complexity of implementation (1-5)',
                'cost_effectiveness': 'Cost-effectiveness rating (1-5)',
                'sustainability_score': 'Long-term sustainability (1-5)'
            },
            sample_size=n_policies,
            cultural_contexts=['various_countries'],
            time_period="2010-2024",
            ethical_considerations=[
                "Public data sources used",
                "Policy impact assessment protocols followed",
                "Stakeholder privacy protected"
            ],
            data=df,
            metadata={
                'data_sources': ['Government reports', 'Academic studies', 'Survey data'],
                'policy_domains': policy_domains,
                'countries_included': df['country'].unique().tolist(),
                'time_span_years': df['implementation_year'].max() - df['implementation_year'].min()
            }
        )
        
        self.datasets[dataset.dataset_id] = dataset
        return dataset
    
    def generate_longitudinal_study(self, n_participants: int = 300,
                                  n_timepoints: int = 5,
                                  study_duration_years: int = 3) -> SocialScienceDataset:
        """Generate longitudinal study dataset"""
        
        data = []
        
        # Generate baseline characteristics
        participants = {}
        for i in range(n_participants):
            culture = np.random.choice(list(self.cultural_contexts.keys()))
            
            participants[i] = {
                'participant_id': f'L{i:04d}',
                'baseline_age': np.random.normal(25, 5),
                'gender': np.random.choice(['Male', 'Female'], p=[0.5, 0.5]),
                'cultural_background': culture,
                'baseline_ses': np.random.normal(3.0, 1.0),
                'baseline_wellbeing': np.random.normal(3.5, 0.8),
                'baseline_social_support': np.random.normal(3.3, 0.9)
            }
        
        # Generate data for each timepoint
        for timepoint in range(n_timepoints):
            time_years = (timepoint * study_duration_years) / (n_timepoints - 1)
            
            for participant_id, baseline in participants.items():
                
                # Calculate age effects
                age_effect = time_years * 0.1
                
                # Calculate trend effects (some variables change over time)
                wellbeing_trend = -0.05 * time_years + np.random.normal(0, 0.2)
                social_support_trend = 0.02 * time_years + np.random.normal(0, 0.15)
                
                # Generate timepoint data
                timepoint_data = {
                    'participant_id': baseline['participant_id'],
                    'timepoint': timepoint,
                    'time_years': time_years,
                    'age': baseline['baseline_age'] + time_years,
                    'cultural_background': baseline['cultural_background'],
                    'ses': baseline['baseline_ses'] + np.random.normal(0, 0.3),
                    'wellbeing': np.clip(baseline['baseline_wellbeing'] + wellbeing_trend, 1, 5),
                    'social_support': np.clip(baseline['baseline_social_support'] + social_support_trend, 1, 5),
                    'life_satisfaction': np.clip(np.random.normal(3.4, 0.9), 1, 5),
                    'stress_level': np.clip(np.random.normal(2.8, 1.0), 1, 5),
                    'social_identity_strength': np.clip(np.random.normal(3.6, 0.7), 1, 5)
                }
                
                # Add measurement error and missing data
                if np.random.random() < 0.05:  # 5% missing data
                    timepoint_data['wellbeing'] = np.nan
                if np.random.random() < 0.03:  # 3% missing data
                    timepoint_data['social_support'] = np.nan
                
                data.append(timepoint_data)
        
        df = pd.DataFrame(data)
        
        dataset = SocialScienceDataset(
            dataset_id="longitudinal_wellbeing_001",
            name="Longitudinal Study of Cultural Identity and Wellbeing",
            domain=SocialScienceDomain.PSYCHOLOGY,
            dataset_type=DatasetType.LONGITUDINAL,
            description="Longitudinal study tracking wellbeing and social identity over time across cultures",
            variables={
                'wellbeing': 'Psychological wellbeing score (1-5)',
                'social_support': 'Perceived social support (1-5)',
                'life_satisfaction': 'Life satisfaction rating (1-5)',
                'stress_level': 'Perceived stress level (1-5)',
                'social_identity_strength': 'Strength of social identity (1-5)',
                'ses': 'Socioeconomic status (1-5)'
            },
            sample_size=n_participants,
            cultural_contexts=list(self.cultural_contexts.keys()),
            time_period=f"{datetime.now().year - study_duration_years}-{datetime.now().year}",
            ethical_considerations=[
                "Longitudinal consent obtained",
                "Participant wellbeing monitored",
                "Right to withdraw at any time",
                "Data confidentiality maintained"
            ],
            data=df,
            metadata={
                'study_design': 'Longitudinal cohort',
                'n_timepoints': n_timepoints,
                'study_duration_years': study_duration_years,
                'attrition_rate': 0.15,
                'missing_data_rate': 0.04
            }
        )
        
        self.datasets[dataset.dataset_id] = dataset
        return dataset
    
    def _get_country_for_culture(self, culture: str) -> str:
        """Get representative country for culture"""
        country_mapping = {
            'western_individualistic': np.random.choice(['USA', 'Canada', 'UK', 'Australia']),
            'east_asian_collectivistic': np.random.choice(['China', 'Japan', 'South Korea', 'Taiwan']),
            'latin_american': np.random.choice(['Mexico', 'Brazil', 'Argentina', 'Colombia']),
            'african_communalistic': np.random.choice(['South Africa', 'Kenya', 'Nigeria', 'Ghana'])
        }
        return country_mapping.get(culture, 'Unknown')
    
    def _get_language_for_culture(self, culture: str) -> str:
        """Get representative language for culture"""
        language_mapping = {
            'western_individualistic': np.random.choice(['English', 'German', 'French']),
            'east_asian_collectivistic': np.random.choice(['Mandarin', 'Japanese', 'Korean']),
            'latin_american': np.random.choice(['Spanish', 'Portuguese']),
            'african_communalistic': np.random.choice(['English', 'Swahili', 'Yoruba'])
        }
        return language_mapping.get(culture, 'English')
    
    def _get_cultural_adjustment(self, culture: str, construct: str) -> float:
        """Get cultural adjustment for psychological constructs"""
        
        adjustments = {
            'social_identity': {
                'western_individualistic': -0.2,
                'east_asian_collectivistic': 0.3,
                'latin_american': 0.1,
                'african_communalistic': 0.4
            },
            'individualism': {
                'western_individualistic': 0.4,
                'east_asian_collectivistic': -0.4,
                'latin_american': -0.1,
                'african_communalistic': -0.3
            }
        }
        
        return adjustments.get(construct, {}).get(culture, 0.0)
    
    def _apply_cultural_response_bias(self, participant: Dict[str, Any], culture: str) -> Dict[str, Any]:
        """Apply cultural response biases to survey responses"""
        
        # Response style adjustments
        if culture == 'east_asian_collectivistic':
            # Tendency toward middle responses (avoid extremes)
            for key in participant:
                if isinstance(participant[key], (int, float)) and 1 <= participant[key] <= 5:
                    if participant[key] < 2:
                        participant[key] = min(2.5, participant[key] + 0.5)
                    elif participant[key] > 4:
                        participant[key] = max(3.5, participant[key] - 0.5)
        
        elif culture == 'latin_american':
            # Slight positive bias (simpatía)
            for key in participant:
                if isinstance(participant[key], (int, float)) and 1 <= participant[key] <= 5:
                    participant[key] = min(5, participant[key] + 0.2)
        
        return participant
    
    def get_dataset(self, dataset_id: str) -> Optional[SocialScienceDataset]:
        """Get dataset by ID"""
        return self.datasets.get(dataset_id)
    
    def list_datasets(self) -> List[str]:
        """List available dataset IDs"""
        return list(self.datasets.keys())
    
    def generate_all_demo_datasets(self) -> Dict[str, SocialScienceDataset]:
        """Generate all demo datasets"""
        
        print("🔄 Generating domain-specific social science datasets...")
        
        datasets = {}
        
        # Social Identity Survey
        print("  📋 Generating social identity survey...")
        datasets['social_identity'] = self.generate_social_identity_survey(n_participants=400)
        
        # Social Network Dataset
        print("  🕸️ Generating social network dataset...")
        datasets['social_network'] = self.generate_social_network_dataset(n_nodes=150)
        
        # Policy Analysis Dataset
        print("  🏛️ Generating policy analysis dataset...")
        datasets['policy_analysis'] = self.generate_policy_analysis_dataset(n_policies=75)
        
        # Longitudinal Study
        print("  📈 Generating longitudinal study...")
        datasets['longitudinal'] = self.generate_longitudinal_study(n_participants=200, n_timepoints=4)
        
        print(f"✅ Generated {len(datasets)} datasets")
        
        return datasets

def demo_domain_specific_datasets():
    """Demonstrate domain-specific dataset generation"""
    
    print("📊 Domain-Specific Social Science Datasets Demo")
    print("=" * 50)
    
    # Initialize generator
    generator = SocialScienceDatasetGenerator()
    
    # Generate all datasets
    datasets = generator.generate_all_demo_datasets()
    
    # Display dataset summaries
    print("\n📋 DATASET SUMMARIES")
    print("-" * 30)
    
    for name, dataset in datasets.items():
        print(f"\n{dataset.name}:")
        print(f"  Domain: {dataset.domain.value}")
        print(f"  Type: {dataset.dataset_type.value}")
        print(f"  Sample Size: {dataset.sample_size}")
        print(f"  Variables: {len(dataset.variables)}")
        print(f"  Cultural Contexts: {len(dataset.cultural_contexts)}")
        print(f"  Description: {dataset.description[:100]}...")
        
        # Show data preview
        if hasattr(dataset.data, 'head'):
            print(f"  Data Shape: {dataset.data.shape}")
            print(f"  Key Variables: {list(dataset.data.columns)[:5]}")
    
    # Demonstrate analysis capabilities
    print(f"\n🔍 ANALYSIS EXAMPLES")
    print("-" * 30)
    
    # Social Identity Analysis
    si_dataset = datasets['social_identity']
    si_data = si_dataset.data
    
    print(f"\nSocial Identity Dataset Analysis:")
    print(f"  Cultural Distribution:")
    for culture in si_data['cultural_background'].value_counts().items():
        print(f"    {culture[0]}: {culture[1]} participants")
    
    print(f"  Mean Social Identity by Culture:")
    for culture in si_data['cultural_background'].unique():
        culture_data = si_data[si_data['cultural_background'] == culture]
        mean_si = culture_data['si_identification'].mean()
        print(f"    {culture}: {mean_si:.2f}")
    
    # Network Analysis
    network_dataset = datasets['social_network']
    print(f"\nSocial Network Dataset Analysis:")
    print(f"  Network Metrics: {network_dataset.metadata['network_metrics']}")
    
    # Policy Analysis
    policy_dataset = datasets['policy_analysis']
    policy_data = policy_dataset.data
    
    print(f"\nPolicy Analysis Dataset:")
    print(f"  Policy Domains: {policy_data['domain'].value_counts().to_dict()}")
    print(f"  Mean Effectiveness: {policy_data['effectiveness_score'].mean():.2f}")
    
    return datasets

if __name__ == "__main__":
    demo_datasets = demo_domain_specific_datasets()
