"""Insurance and risk assessment module for the evacuation simulation."""

from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from datetime import datetime

__all__ = [
    'InsurancePolicy',
    'RiskAssessor',
    'calculate_risk_score',
    'generate_insurance_claim'
]

class InsurancePolicy:
    """Class representing an insurance policy for an agent or property."""
    
    def __init__(
        self,
        policy_id: str,
        policy_type: str = 'standard',
        coverage_limits: Optional[Dict[str, float]] = None,
        deductibles: Optional[Dict[str, float]] = None,
        premium: float = 1000.0,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ):
        """Initialize an insurance policy.
        
        Args:
            policy_id: Unique identifier for the policy
            policy_type: Type of policy ('standard', 'premium', 'basic')
            coverage_limits: Dictionary of coverage limits by category
            deductibles: Dictionary of deductibles by category
            premium: Annual premium amount
            start_date: Policy start date
            end_date: Policy end date
        """
        self.policy_id = policy_id
        self.policy_type = policy_type
        self.premium = premium
        self.start_date = start_date or datetime.now()
        self.end_date = end_date or (self.start_date.replace(year=self.start_date.year + 1))
        
        # Set default coverage limits if not provided
        if coverage_limits is None:
            self.coverage_limits = {
                'medical': 1000000.0,
                'property': 500000.0,
                'liability': 2000000.0,
                'evacuation': 10000.0,
                'business_interruption': 250000.0
            }
        else:
            self.coverage_limits = coverage_limits
            
        # Set default deductibles if not provided
        if deductibles is None:
            self.deductibles = {
                'medical': 1000.0,
                'property': 2500.0,
                'liability': 5000.0,
                'evacuation': 500.0,
                'business_interruption': 5000.0
            }
        else:
            self.deductibles = deductibles
        
        # Initialize claims
        self.claims = []
        self.total_claimed = 0.0
        self.total_paid = 0.0
    
    def add_claim(
        self,
        claim_type: str,
        amount: float,
        description: str = "",
        timestamp: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """Add a new insurance claim.
        
        Args:
            claim_type: Type of claim ('medical', 'property', etc.)
            amount: Claim amount
            description: Description of the claim
            timestamp: When the claim occurred
            
        Returns:
            Dictionary with claim details
        """
        if timestamp is None:
            timestamp = datetime.now()
            
        # Check if claim type is valid
        if claim_type not in self.coverage_limits:
            raise ValueError(f"Invalid claim type: {claim_type}")
        
        # Calculate covered amount (after deductible)
        deductible = self.deductibles.get(claim_type, 0.0)
        covered_amount = max(0.0, amount - deductible)
        
        # Apply coverage limit
        coverage_limit = self.coverage_limits.get(claim_type, 0.0)
        if covered_amount > coverage_limit:
            covered_amount = coverage_limit
        
        # Create claim record
        claim = {
            'claim_id': f"CLM-{len(self.claims) + 1:06d}",
            'claim_type': claim_type,
            'amount': amount,
            'covered_amount': covered_amount,
            'deductible': min(deductible, amount),
            'description': description,
            'timestamp': timestamp,
            'status': 'submitted',
            'processed_date': None,
            'payout_date': None
        }
        
        self.claims.append(claim)
        self.total_claimed += amount
        self.total_paid += covered_amount
        
        return claim
    
    def process_claim(self, claim_id: str, status: str = 'approved', notes: str = "") -> bool:
        """Process an insurance claim.
        
        Args:
            claim_id: ID of the claim to process
            status: New status ('approved', 'denied', 'pending')
            notes: Processing notes
            
        Returns:
            True if claim was found and processed, False otherwise
        """
        for claim in self.claims:
            if claim['claim_id'] == claim_id:
                claim['status'] = status
                claim['processed_date'] = datetime.now()
                claim['notes'] = notes
                
                if status == 'approved':
                    claim['payout_date'] = datetime.now()
                
                return True
        
        return False
    
    def get_claim(self, claim_id: str) -> Optional[Dict[str, Any]]:
        """Get claim details by ID.
        
        Args:
            claim_id: ID of the claim to retrieve
            
        Returns:
            Claim details or None if not found
        """
        for claim in self.claims:
            if claim['claim_id'] == claim_id:
                return claim
        return None
    
    def get_claims_by_type(self, claim_type: str) -> List[Dict[str, Any]]:
        """Get all claims of a specific type.
        
        Args:
            claim_type: Type of claims to retrieve
            
        Returns:
            List of matching claims
        """
        return [claim for claim in self.claims if claim['claim_type'] == claim_type]
    
    def get_coverage_utilization(self) -> Dict[str, float]:
        """Get the utilization percentage for each coverage type.
        
        Returns:
            Dictionary mapping coverage types to utilization percentage (0-1)
        """
        utilization = {}
        
        for claim_type, limit in self.coverage_limits.items():
            total_claimed = sum(
                claim['amount'] 
                for claim in self.claims 
                if claim['claim_type'] == claim_type
            )
            utilization[claim_type] = min(1.0, total_claimed / limit) if limit > 0 else 0.0
        
        return utilization
    
    def is_active(self) -> bool:
        """Check if the policy is currently active."""
        now = datetime.now()
        return self.start_date <= now <= self.end_date


class RiskAssessor:
    """Class for assessing and managing risk in the simulation."""
    
    def __init__(self):
        """Initialize the risk assessor."""
        self.risk_factors = {
            'hazard_exposure': 0.5,    # Weight for hazard exposure
            'congestion': 0.3,         # Weight for congestion
            'panic_level': 0.2,        # Weight for panic level
            'agent_vulnerability': 0.3, # Weight for agent vulnerability
            'environmental_risk': 0.2  # Weight for environmental risk factors
        }
        
        self.risk_history = []
        self.risk_thresholds = {
            'low': 0.3,
            'medium': 0.6,
            'high': 0.9
        }
    
    def assess_agent_risk(
        self,
        agent: 'EvacuationAgent',
        environment: 'EvacuationEnvironment'
    ) -> float:
        """Assess the risk level for a specific agent.
        
        Args:
            agent: The agent to assess
            environment: The simulation environment
            
        Returns:
            Risk score between 0 and 1
        """
        # Calculate hazard exposure
        hazard_exposure = self._calculate_hazard_exposure(agent, environment)
        
        # Calculate congestion factor
        congestion = self._calculate_congestion(agent, environment)
        
        # Get agent panic level
        panic_level = agent.panic_level
        
        # Calculate agent vulnerability
        vulnerability = self._calculate_agent_vulnerability(agent)
        
        # Calculate environmental risk
        env_risk = self._calculate_environmental_risk(environment)
        
        # Calculate weighted risk score
        risk_score = (
            self.risk_factors['hazard_exposure'] * hazard_exposure +
            self.risk_factors['congestion'] * congestion +
            self.risk_factors['panic_level'] * panic_level +
            self.risk_factors['agent_vulnerability'] * vulnerability +
            self.risk_factors['environmental_risk'] * env_risk
        )
        
        # Ensure risk score is between 0 and 1
        risk_score = max(0.0, min(1.0, risk_score))
        
        # Store risk assessment
        self.risk_history.append({
            'agent_id': agent.agent_id,
            'timestamp': environment.current_time,
            'risk_score': risk_score,
            'hazard_exposure': hazard_exposure,
            'congestion': congestion,
            'panic_level': panic_level,
            'vulnerability': vulnerability,
            'environmental_risk': env_risk
        })
        
        return risk_score
    
    def assess_environment_risk(
        self,
        environment: 'EvacuationEnvironment'
    ) -> Dict[str, Any]:
        """Assess the overall risk level of the environment.
        
        Args:
            environment: The simulation environment
            
        Returns:
            Dictionary with risk assessment results
        """
        # Calculate average agent risk
        agent_risks = []
        for agent in environment.agents:
            agent_risk = self.assess_agent_risk(agent, environment)
            agent_risks.append(agent_risk)
        
        avg_agent_risk = sum(agent_risks) / len(agent_risks) if agent_risks else 0.0
        
        # Calculate hazard risk
        hazard_risk = self._calculate_environmental_risk(environment)
        
        # Calculate evacuation progress
        if environment.initial_agent_count > 0:
            evacuation_progress = (
                environment.initial_agent_count - len(environment.agents)
            ) / environment.initial_agent_count
        else:
            evacuation_progress = 0.0
        
        # Calculate overall risk score
        risk_score = 0.6 * avg_agent_risk + 0.4 * hazard_risk
        
        # Determine risk level
        if risk_score < self.risk_thresholds['low']:
            risk_level = 'low'
        elif risk_score < self.risk_thresholds['medium']:
            risk_level = 'medium'
        elif risk_score < self.risk_thresholds['high']:
            risk_level = 'high'
        else:
            risk_level = 'extreme'
        
        return {
            'risk_score': risk_score,
            'risk_level': risk_level,
            'avg_agent_risk': avg_agent_risk,
            'hazard_risk': hazard_risk,
            'evacuation_progress': evacuation_progress,
            'timestamp': environment.current_time
        }
    
    def _calculate_hazard_exposure(
        self,
        agent: 'EvacuationAgent',
        environment: 'EvacuationEnvironment'
    ) -> float:
        """Calculate the hazard exposure for an agent."""
        exposure = 0.0
        
        for hazard_type, hazard_data in environment.hazards.items():
            for pos, intensity in zip(hazard_data['positions'], hazard_data['intensities']):
                dist = np.linalg.norm(agent.position - np.array(pos))
                if dist < hazard_data['radius']:
                    influence = (1 - dist / hazard_data['radius']) * intensity
                    exposure = max(exposure, influence)
        
        return exposure
    
    def _calculate_congestion(
        self,
        agent: 'EvacuationAgent',
        environment: 'EvacuationEnvironment',
        radius: float = 5.0
    ) -> float:
        """Calculate the local congestion around an agent."""
        neighbor_count = 0
        
        for other in environment.agents:
            if other.agent_id == agent.agent_id:
                continue
                
            dist = np.linalg.norm(agent.position - other.position)
            if dist < radius:
                neighbor_count += 1
        
        # Normalize by maximum expected neighbors in radius
        max_neighbors = 10  # Approximate maximum in a 5m radius
        return min(1.0, neighbor_count / max_neighbors)
    
    def _calculate_agent_vulnerability(self, agent: 'EvacuationAgent') -> float:
        """Calculate the vulnerability of an agent based on their attributes."""
        # Base vulnerability on agent type and health
        type_vulnerability = {
            'adult': 0.3,
            'child': 0.7,
            'elderly': 0.6,
            'mobility_impaired': 0.9
        }.get(agent.agent_type, 0.5)
        
        # Adjust based on health (lower health = more vulnerable)
        health_factor = 1.0 - agent.health
        
        # Adjust based on panic level (higher panic = more vulnerable)
        panic_factor = agent.panic_level
        
        # Calculate combined vulnerability
        vulnerability = (
            0.5 * type_vulnerability +
            0.3 * health_factor +
            0.2 * panic_factor
        )
        
        return max(0.0, min(1.0, vulnerability))
    
    def _calculate_environmental_risk(self, environment: 'EvacuationEnvironment') -> float:
        """Calculate environmental risk factors."""
        # Calculate hazard intensity
        hazard_intensity = 0.0
        for hazard_data in environment.hazards.values():
            if hazard_data['positions']:
                hazard_intensity = max(hazard_intensity, max(hazard_data['intensities'], default=0.0))
        
        # Calculate congestion (global)
        agent_density = len(environment.agents) / (environment.width * environment.height)
        max_density = 0.1  # Maximum agents per square meter
        congestion = min(1.0, agent_density / max_density)
        
        # Calculate evacuation progress
        if environment.initial_agent_count > 0:
            remaining = len(environment.agents) / environment.initial_agent_count
        else:
            remaining = 1.0
        
        # Environmental risk is higher when hazards are intense, congestion is high, 
        # and many agents remain
        env_risk = 0.5 * hazard_intensity + 0.3 * congestion + 0.2 * remaining
        
        return max(0.0, min(1.0, env_risk))
    
    def get_risk_level(self, risk_score: float) -> str:
        """Get a human-readable risk level from a risk score."""
        if risk_score < self.risk_thresholds['low']:
            return 'low'
        elif risk_score < self.risk_thresholds['medium']:
            return 'medium'
        elif risk_score < self.risk_thresholds['high']:
            return 'high'
        else:
            return 'extreme'
    
    def get_risk_mitigation_recommendations(
        self,
        risk_assessment: Dict[str, Any]
    ) -> List[str]:
        """Get recommendations for mitigating identified risks.
        
        Args:
            risk_assessment: Result from assess_environment_risk()
            
        Returns:
            List of recommended actions
        """
        recommendations = []
        
        if risk_assessment['risk_level'] == 'low':
            recommendations.append("Continue monitoring the situation.")
        
        if risk_assessment['risk_level'] in ['medium', 'high', 'extreme']:
            if risk_assessment['hazard_risk'] > 0.5:
                recommendations.append(
                    "Activate emergency response for hazard mitigation."
                )
            
            if risk_assessment['avg_agent_risk'] > 0.5:
                recommendations.append(
                    "Deploy additional staff to assist with evacuation."
                )
            
            if risk_assessment['evacuation_progress'] < 0.5:
                recommendations.append(
                    "Open additional exits to improve evacuation flow."
                )
        
        if risk_assessment['risk_level'] in ['high', 'extreme']:
            recommendations.append("Activate emergency alert system.")
            
            if risk_assessment['hazard_risk'] > 0.7:
                recommendations.append(
                    "Consider partial or full building evacuation."
                )
        
        if risk_assessment['risk_level'] == 'extreme':
            recommendations.append("Activate disaster recovery protocols.")
            recommendations.append("Notify emergency services immediately.")
        
        return recommendations


def calculate_risk_score(
    hazard_exposure: float,
    congestion: float,
    panic_level: float,
    weights: Optional[Dict[str, float]] = None
) -> float:
    """Calculate a risk score based on various factors.
    
    Args:
        hazard_exposure: Exposure to hazards (0-1)
        congestion: Local congestion level (0-1)
        panic_level: Average panic level (0-1)
        weights: Optional dictionary of weights for each factor
        
    Returns:
        Risk score between 0 and 1
    """
    # Default weights
    if weights is None:
        weights = {
            'hazard_exposure': 0.5,
            'congestion': 0.3,
            'panic_level': 0.2
        }
    
    # Ensure weights sum to 1
    total_weight = sum(weights.values())
    if total_weight > 0:
        weights = {k: v / total_weight for k, v in weights.items()}
    
    # Calculate weighted risk score
    risk_score = (
        weights.get('hazard_exposure', 0.0) * hazard_exposure +
        weights.get('congestion', 0.0) * congestion +
        weights.get('panic_level', 0.0) * panic_level
    )
    
    return max(0.0, min(1.0, risk_score))


def generate_insurance_claim(
    claim_type: str,
    amount: float,
    description: str = "",
    timestamp: Optional[datetime] = None,
    policy_id: Optional[str] = None,
    agent_id: Optional[str] = None
) -> Dict[str, Any]:
    """Generate an insurance claim.
    
    Args:
        claim_type: Type of claim ('medical', 'property', etc.)
        amount: Claim amount
        description: Description of the claim
        timestamp: When the claim occurred
        policy_id: ID of the insurance policy
        agent_id: ID of the agent making the claim
        
    Returns:
        Dictionary with claim details
    """
    if timestamp is None:
        timestamp = datetime.now()
    
    if policy_id is None:
        policy_id = f"POL-{np.random.randint(100000, 999999)}"
    
    if agent_id is None:
        agent_id = f"AGT-{np.random.randint(1000, 9999)}"
    
    return {
        'claim_id': f"CLM-{np.random.randint(100000, 999999)}",
        'policy_id': policy_id,
        'agent_id': agent_id,
        'claim_type': claim_type,
        'amount': amount,
        'description': description,
        'timestamp': timestamp,
        'status': 'submitted',
        'processed_date': None,
        'payout_date': None,
        'notes': ""
    }
