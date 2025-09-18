# -*- coding: utf-8 -*-
"""
Multi-Agent Simulation Framework for Social Science Research
Addresses the "Agentic Simulation" criterion from the assessment
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import random
import networkx as nx
from abc import ABC, abstractmethod

class AgentRole(Enum):
    """Types of agents in social science simulation"""
    RESEARCHER = "researcher"
    PARTICIPANT = "participant"
    OBSERVER = "observer"
    MODERATOR = "moderator"
    CULTURAL_INFORMANT = "cultural_informant"

class InteractionType(Enum):
    """Types of social interactions"""
    SURVEY_RESPONSE = "survey_response"
    INTERVIEW = "interview"
    GROUP_DISCUSSION = "group_discussion"
    OBSERVATION = "observation"
    PEER_INTERACTION = "peer_interaction"
    CULTURAL_EXCHANGE = "cultural_exchange"

class SocialContext(Enum):
    """Social contexts for interactions"""
    FORMAL_RESEARCH = "formal_research"
    INFORMAL_SOCIAL = "informal_social"
    WORKPLACE = "workplace"
    EDUCATIONAL = "educational"
    COMMUNITY = "community"
    FAMILY = "family"

@dataclass
class AgentProfile:
    """Profile of an agent in the simulation"""
    agent_id: str
    role: AgentRole
    cultural_background: str
    personality_traits: Dict[str, float]  # Big 5 + cultural dimensions
    social_identity_strength: float
    cultural_adaptation_level: float
    communication_style: str
    trust_level: float
    knowledge_domains: List[str]

@dataclass
class Interaction:
    """Record of an interaction between agents"""
    interaction_id: str
    timestamp: float
    initiator_id: str
    target_id: str
    interaction_type: InteractionType
    context: SocialContext
    content: Dict[str, Any]
    outcome: Dict[str, Any]
    cultural_factors: List[str]

class SocialAgent(ABC):
    """Abstract base class for social agents"""
    
    def __init__(self, profile: AgentProfile):
        """Initialize agent with profile"""
        self.profile = profile
        self.interaction_history = []
        self.relationships = {}  # agent_id -> relationship_strength
        self.current_state = {
            "mood": 0.5,
            "engagement": 0.5,
            "trust": self.profile.trust_level,
            "cultural_comfort": 0.5
        }
        
    @abstractmethod
    def decide_interaction(self, available_agents: List['SocialAgent'], 
                          context: SocialContext) -> Optional[Tuple[str, InteractionType]]:
        """Decide whether and how to interact with other agents"""
        pass
    
    @abstractmethod
    def process_interaction(self, interaction: Interaction) -> Dict[str, Any]:
        """Process an incoming interaction and generate response"""
        pass
    
    def update_relationships(self, other_agent_id: str, interaction_outcome: Dict[str, Any]):
        """Update relationship strength based on interaction outcome"""
        current_strength = self.relationships.get(other_agent_id, 0.0)
        
        # Simple relationship update based on interaction success
        success_factor = interaction_outcome.get('success', 0.5)
        cultural_compatibility = interaction_outcome.get('cultural_compatibility', 0.5)
        
        # Update relationship strength
        change = 0.1 * (success_factor + cultural_compatibility - 1.0)
        new_strength = max(0.0, min(1.0, current_strength + change))
        self.relationships[other_agent_id] = new_strength
    
    def get_cultural_compatibility(self, other_agent: 'SocialAgent') -> float:
        """Calculate cultural compatibility with another agent"""
        
        # Simple compatibility based on cultural background similarity
        if self.profile.cultural_background == other_agent.profile.cultural_background:
            base_compatibility = 0.8
        else:
            base_compatibility = 0.4
        
        # Adjust based on cultural adaptation levels
        adaptation_factor = (self.profile.cultural_adaptation_level + 
                           other_agent.profile.cultural_adaptation_level) / 2
        
        return base_compatibility * (0.5 + 0.5 * adaptation_factor)

class ResearcherAgent(SocialAgent):
    """Agent representing a researcher"""
    
    def __init__(self, profile: AgentProfile):
        super().__init__(profile)
        self.research_goals = []
        self.data_collected = []
        self.cultural_sensitivity_level = 0.7
        
    def decide_interaction(self, available_agents: List[SocialAgent], 
                          context: SocialContext) -> Optional[Tuple[str, InteractionType]]:
        """Researcher decides on data collection interactions"""
        
        # Prioritize participants for data collection
        participants = [agent for agent in available_agents 
                       if agent.profile.role == AgentRole.PARTICIPANT]
        
        if not participants:
            return None
        
        # Choose participant based on research needs and relationship
        best_participant = None
        best_score = -1
        
        for participant in participants:
            # Score based on relationship strength and data needs
            relationship_strength = self.relationships.get(participant.profile.agent_id, 0.0)
            cultural_compatibility = self.get_cultural_compatibility(participant)
            
            score = relationship_strength * 0.4 + cultural_compatibility * 0.6
            
            if score > best_score:
                best_score = score
                best_participant = participant
        
        if best_participant:
            # Choose interaction type based on context
            if context == SocialContext.FORMAL_RESEARCH:
                interaction_type = InteractionType.SURVEY_RESPONSE
            else:
                interaction_type = InteractionType.INTERVIEW
            
            return (best_participant.profile.agent_id, interaction_type)
        
        return None
    
    def process_interaction(self, interaction: Interaction) -> Dict[str, Any]:
        """Process research interaction"""
        
        # Simulate data collection quality based on cultural factors
        cultural_sensitivity_bonus = self.cultural_sensitivity_level * 0.2
        base_quality = 0.6 + cultural_sensitivity_bonus
        
        # Adjust based on participant's cultural comfort
        participant_comfort = random.uniform(0.3, 1.0)  # Simplified
        final_quality = base_quality * participant_comfort
        
        # Record data
        data_point = {
            'participant_id': interaction.target_id,
            'interaction_type': interaction.interaction_type.value,
            'data_quality': final_quality,
            'cultural_factors': interaction.cultural_factors,
            'timestamp': interaction.timestamp
        }
        self.data_collected.append(data_point)
        
        return {
            'success': final_quality,
            'cultural_compatibility': participant_comfort,
            'data_collected': True
        }

class ParticipantAgent(SocialAgent):
    """Agent representing a research participant"""
    
    def __init__(self, profile: AgentProfile):
        super().__init__(profile)
        self.participation_willingness = random.uniform(0.3, 0.9)
        self.response_patterns = self._generate_response_patterns()
        
    def _generate_response_patterns(self) -> Dict[str, Any]:
        """Generate culturally-influenced response patterns"""
        
        patterns = {
            'social_desirability_bias': 0.3,
            'acquiescence_bias': 0.2,
            'extreme_response_style': 0.1,
            'cultural_response_style': 'moderate'
        }
        
        # Adjust based on cultural background
        if 'collectivistic' in self.profile.cultural_background.lower():
            patterns['social_desirability_bias'] += 0.2
            patterns['cultural_response_style'] = 'harmonious'
        elif 'individualistic' in self.profile.cultural_background.lower():
            patterns['extreme_response_style'] += 0.1
            patterns['cultural_response_style'] = 'direct'
        
        return patterns
    
    def decide_interaction(self, available_agents: List[SocialAgent], 
                          context: SocialContext) -> Optional[Tuple[str, InteractionType]]:
        """Participant decides on social interactions"""
        
        # Participants mainly respond to researchers or interact socially
        if context == SocialContext.FORMAL_RESEARCH:
            return None  # Wait for researcher to initiate
        
        # Social interactions with other participants
        other_participants = [agent for agent in available_agents 
                            if agent.profile.role == AgentRole.PARTICIPANT 
                            and agent.profile.agent_id != self.profile.agent_id]
        
        if other_participants and random.random() < 0.3:  # 30% chance of social interaction
            target = random.choice(other_participants)
            return (target.profile.agent_id, InteractionType.PEER_INTERACTION)
        
        return None
    
    def process_interaction(self, interaction: Interaction) -> Dict[str, Any]:
        """Process interaction as participant"""
        
        if interaction.interaction_type == InteractionType.SURVEY_RESPONSE:
            return self._respond_to_survey(interaction)
        elif interaction.interaction_type == InteractionType.INTERVIEW:
            return self._respond_to_interview(interaction)
        elif interaction.interaction_type == InteractionType.PEER_INTERACTION:
            return self._engage_in_peer_interaction(interaction)
        else:
            return {'success': 0.5, 'cultural_compatibility': 0.5}
    
    def _respond_to_survey(self, interaction: Interaction) -> Dict[str, Any]:
        """Respond to survey with cultural biases"""
        
        # Base willingness to participate
        participation_prob = self.participation_willingness
        
        # Adjust based on cultural comfort with researcher
        cultural_comfort = self.current_state['cultural_comfort']
        final_participation = participation_prob * cultural_comfort
        
        if final_participation < 0.4:
            return {'success': 0.2, 'cultural_compatibility': cultural_comfort, 'participated': False}
        
        # Generate responses with cultural biases
        responses = {}
        for question_type in ['social_identity', 'cultural_adaptation', 'technology_adoption']:
            base_response = random.uniform(1, 5)  # Likert scale
            
            # Apply cultural response biases
            if self.response_patterns['cultural_response_style'] == 'harmonious':
                base_response = min(4.5, base_response + 0.5)  # Avoid extremes
            elif self.response_patterns['cultural_response_style'] == 'direct':
                if random.random() < 0.3:  # 30% chance of extreme response
                    base_response = 1 if base_response < 3 else 5
            
            # Apply social desirability bias
            if random.random() < self.response_patterns['social_desirability_bias']:
                base_response = min(5, base_response + 0.8)
            
            responses[question_type] = round(base_response, 1)
        
        return {
            'success': 0.8,
            'cultural_compatibility': cultural_comfort,
            'participated': True,
            'responses': responses,
            'response_quality': final_participation
        }
    
    def _respond_to_interview(self, interaction: Interaction) -> Dict[str, Any]:
        """Respond to interview with cultural considerations"""
        
        cultural_comfort = self.current_state['cultural_comfort']
        
        # Generate interview responses
        response_depth = cultural_comfort * self.participation_willingness
        
        interview_data = {
            'depth_of_response': response_depth,
            'cultural_themes_mentioned': [],
            'personal_disclosure_level': response_depth * 0.8
        }
        
        # Add cultural themes based on background
        if 'collectivistic' in self.profile.cultural_background.lower():
            interview_data['cultural_themes_mentioned'].extend([
                'family_importance', 'group_harmony', 'collective_decision_making'
            ])
        elif 'individualistic' in self.profile.cultural_background.lower():
            interview_data['cultural_themes_mentioned'].extend([
                'personal_achievement', 'individual_rights', 'self_expression'
            ])
        
        return {
            'success': response_depth,
            'cultural_compatibility': cultural_comfort,
            'interview_data': interview_data
        }
    
    def _engage_in_peer_interaction(self, interaction: Interaction) -> Dict[str, Any]:
        """Engage in peer-to-peer interaction"""
        
        # Simple peer interaction simulation
        compatibility = random.uniform(0.3, 0.9)
        
        return {
            'success': compatibility,
            'cultural_compatibility': compatibility,
            'social_bond_formed': compatibility > 0.6
        }

class SocialSimulationEnvironment:
    """Environment for running social science simulations"""
    
    def __init__(self):
        """Initialize simulation environment"""
        self.agents = {}
        self.interaction_log = []
        self.time_step = 0
        self.social_network = nx.Graph()
        self.cultural_contexts = {}
        
    def add_agent(self, agent: SocialAgent):
        """Add agent to simulation"""
        self.agents[agent.profile.agent_id] = agent
        self.social_network.add_node(agent.profile.agent_id, 
                                   role=agent.profile.role.value,
                                   culture=agent.profile.cultural_background)
    
    def create_research_scenario(self, n_researchers: int = 2, n_participants: int = 20,
                               cultural_distribution: Dict[str, float] = None) -> None:
        """Create a research scenario with specified agents"""
        
        if cultural_distribution is None:
            cultural_distribution = {
                'western_individualistic': 0.4,
                'east_asian_collectivistic': 0.4,
                'latin_american': 0.2
            }
        
        # Create researchers
        for i in range(n_researchers):
            profile = AgentProfile(
                agent_id=f"researcher_{i}",
                role=AgentRole.RESEARCHER,
                cultural_background="western_individualistic",  # Assume Western researchers
                personality_traits={
                    'openness': random.uniform(0.6, 0.9),
                    'conscientiousness': random.uniform(0.7, 0.9),
                    'extraversion': random.uniform(0.4, 0.8),
                    'agreeableness': random.uniform(0.6, 0.9),
                    'neuroticism': random.uniform(0.1, 0.4)
                },
                social_identity_strength=random.uniform(0.5, 0.8),
                cultural_adaptation_level=random.uniform(0.6, 0.9),
                communication_style="direct",
                trust_level=random.uniform(0.6, 0.8),
                knowledge_domains=["social_science", "research_methods", "statistics"]
            )
            
            researcher = ResearcherAgent(profile)
            self.add_agent(researcher)
        
        # Create participants with cultural distribution
        cultures = list(cultural_distribution.keys())
        culture_weights = list(cultural_distribution.values())
        
        for i in range(n_participants):
            culture = np.random.choice(cultures, p=culture_weights)
            
            # Generate culturally-influenced personality traits
            personality = self._generate_cultural_personality(culture)
            
            profile = AgentProfile(
                agent_id=f"participant_{i}",
                role=AgentRole.PARTICIPANT,
                cultural_background=culture,
                personality_traits=personality,
                social_identity_strength=random.uniform(0.3, 0.9),
                cultural_adaptation_level=random.uniform(0.2, 0.8),
                communication_style=self._get_cultural_communication_style(culture),
                trust_level=random.uniform(0.3, 0.8),
                knowledge_domains=["personal_experience", "cultural_knowledge"]
            )
            
            participant = ParticipantAgent(profile)
            self.add_agent(participant)
    
    def _generate_cultural_personality(self, culture: str) -> Dict[str, float]:
        """Generate personality traits influenced by culture"""
        
        base_traits = {
            'openness': random.uniform(0.3, 0.7),
            'conscientiousness': random.uniform(0.3, 0.7),
            'extraversion': random.uniform(0.3, 0.7),
            'agreeableness': random.uniform(0.3, 0.7),
            'neuroticism': random.uniform(0.3, 0.7)
        }
        
        # Cultural adjustments
        if 'collectivistic' in culture:
            base_traits['agreeableness'] += 0.2
            base_traits['extraversion'] -= 0.1
        elif 'individualistic' in culture:
            base_traits['openness'] += 0.1
            base_traits['extraversion'] += 0.1
        
        # Ensure values stay in valid range
        for trait in base_traits:
            base_traits[trait] = max(0.0, min(1.0, base_traits[trait]))
        
        return base_traits
    
    def _get_cultural_communication_style(self, culture: str) -> str:
        """Get communication style based on culture"""
        
        if 'individualistic' in culture:
            return "direct"
        elif 'collectivistic' in culture:
            return "indirect"
        else:
            return "moderate"
    
    def run_simulation(self, n_steps: int = 100, context: SocialContext = SocialContext.FORMAL_RESEARCH) -> Dict[str, Any]:
        """Run the social simulation"""
        
        print(f"🔄 Running social simulation for {n_steps} steps...")
        
        for step in range(n_steps):
            self.time_step = step
            
            # Get all agents
            agent_list = list(self.agents.values())
            
            # Each agent decides on interactions
            for agent in agent_list:
                available_agents = [a for a in agent_list if a.profile.agent_id != agent.profile.agent_id]
                
                decision = agent.decide_interaction(available_agents, context)
                
                if decision:
                    target_id, interaction_type = decision
                    target_agent = self.agents[target_id]
                    
                    # Create interaction
                    interaction = Interaction(
                        interaction_id=f"interaction_{step}_{agent.profile.agent_id}_{target_id}",
                        timestamp=step,
                        initiator_id=agent.profile.agent_id,
                        target_id=target_id,
                        interaction_type=interaction_type,
                        context=context,
                        content={},
                        outcome={},
                        cultural_factors=[agent.profile.cultural_background, target_agent.profile.cultural_background]
                    )
                    
                    # Process interaction
                    outcome = target_agent.process_interaction(interaction)
                    interaction.outcome = outcome
                    
                    # Update relationships
                    agent.update_relationships(target_id, outcome)
                    target_agent.update_relationships(agent.profile.agent_id, outcome)
                    
                    # Update social network
                    if outcome.get('success', 0) > 0.5:
                        self.social_network.add_edge(agent.profile.agent_id, target_id, 
                                                   weight=outcome.get('success', 0.5))
                    
                    # Log interaction
                    self.interaction_log.append(interaction)
        
        # Generate simulation results
        results = self._analyze_simulation_results()
        
        print(f"✅ Simulation completed: {len(self.interaction_log)} interactions")
        
        return results
    
    def _analyze_simulation_results(self) -> Dict[str, Any]:
        """Analyze simulation results"""
        
        # Basic statistics
        total_interactions = len(self.interaction_log)
        successful_interactions = len([i for i in self.interaction_log if i.outcome.get('success', 0) > 0.5])
        
        # Interaction types
        interaction_types = {}
        for interaction in self.interaction_log:
            itype = interaction.interaction_type.value
            interaction_types[itype] = interaction_types.get(itype, 0) + 1
        
        # Cultural analysis
        cultural_interactions = {}
        cross_cultural_interactions = 0
        
        for interaction in self.interaction_log:
            cultures = interaction.cultural_factors
            if len(set(cultures)) > 1:
                cross_cultural_interactions += 1
            
            culture_pair = tuple(sorted(cultures))
            cultural_interactions[culture_pair] = cultural_interactions.get(culture_pair, 0) + 1
        
        # Network analysis
        network_stats = {
            'nodes': self.social_network.number_of_nodes(),
            'edges': self.social_network.number_of_edges(),
            'density': nx.density(self.social_network),
            'connected_components': nx.number_connected_components(self.social_network)
        }
        
        # Data collection analysis (for researchers)
        researchers = [agent for agent in self.agents.values() if agent.profile.role == AgentRole.RESEARCHER]
        total_data_collected = sum(len(r.data_collected) for r in researchers)
        
        return {
            'simulation_stats': {
                'total_interactions': total_interactions,
                'successful_interactions': successful_interactions,
                'success_rate': successful_interactions / total_interactions if total_interactions > 0 else 0,
                'cross_cultural_interactions': cross_cultural_interactions,
                'cross_cultural_rate': cross_cultural_interactions / total_interactions if total_interactions > 0 else 0
            },
            'interaction_types': interaction_types,
            'cultural_interactions': cultural_interactions,
            'network_stats': network_stats,
            'data_collection': {
                'total_data_points': total_data_collected,
                'researchers': len(researchers),
                'participants': len([a for a in self.agents.values() if a.profile.role == AgentRole.PARTICIPANT])
            },
            'agents': {agent_id: {
                'role': agent.profile.role.value,
                'culture': agent.profile.cultural_background,
                'relationships': len(agent.relationships),
                'interactions': len(agent.interaction_history)
            } for agent_id, agent in self.agents.items()}
        }

def demo_multi_agent_simulation():
    """Demonstrate multi-agent simulation capabilities"""
    
    print("🤖 Multi-Agent Social Science Simulation Demo")
    print("=" * 50)
    
    # Create simulation environment
    env = SocialSimulationEnvironment()
    
    # Create research scenario
    print("\n📋 Creating research scenario...")
    env.create_research_scenario(
        n_researchers=2,
        n_participants=15,
        cultural_distribution={
            'western_individualistic': 0.4,
            'east_asian_collectivistic': 0.4,
            'latin_american': 0.2
        }
    )
    
    print(f"✅ Created scenario with {len(env.agents)} agents")
    
    # Run simulation
    print("\n🔄 Running simulation...")
    results = env.run_simulation(n_steps=50, context=SocialContext.FORMAL_RESEARCH)
    
    # Display results
    print("\n📊 SIMULATION RESULTS")
    print("-" * 30)
    
    stats = results['simulation_stats']
    print(f"Total Interactions: {stats['total_interactions']}")
    print(f"Success Rate: {stats['success_rate']:.1%}")
    print(f"Cross-Cultural Rate: {stats['cross_cultural_rate']:.1%}")
    
    print(f"\nData Collection:")
    data_stats = results['data_collection']
    print(f"  Data Points: {data_stats['total_data_points']}")
    print(f"  Researchers: {data_stats['researchers']}")
    print(f"  Participants: {data_stats['participants']}")
    
    print(f"\nNetwork Statistics:")
    network = results['network_stats']
    print(f"  Nodes: {network['nodes']}")
    print(f"  Edges: {network['edges']}")
    print(f"  Density: {network['density']:.3f}")
    
    print(f"\nInteraction Types:")
    for itype, count in results['interaction_types'].items():
        print(f"  {itype.replace('_', ' ').title()}: {count}")
    
    return results

if __name__ == "__main__":
    demo_results = demo_multi_agent_simulation()