import random

# Constants
INITIAL_MONEY = 100
INITIAL_CREDIT_SCORE = 100
PROJECT_TYPES = ['Residential', 'Commercial', 'Parks', 'Metro']
BASE_PROJECT_COSTS = {'Residential': 20, 'Commercial': 40, 'Parks': 15, 'Metro': 50}
SCORES_TEMPLATE = {'Prosperity': 0, 'Happiness': 0, 'Environmental': 0, 'ROI': 0}
NUM_ROUNDS = 10
CREDIT_SCORE_IMPACT = 10  # Impact for breaking a deal
DYNAMIC_COST_FLUCTUATION = 0.1  # Fluctuation rate for dynamic project costs
NEGOTIATION_PENALTY = 10  # Money penalty for failed negotiations

class Agent:
    def __init__(self, name, strategy):
        self.name = name
        self.strategy = strategy
        self.money = INITIAL_MONEY
        self.credit_score = INITIAL_CREDIT_SCORE
        self.projects = []
        self.scores = SCORES_TEMPLATE.copy()
        self.negotiation_targets = []  # Targets for negotiation

    def bid(self, project_type, project_costs):
        if project_type in self.negotiation_targets:
            return 0  # Skip bidding due to negotiation
        credit_score_factor = max(0.8, 1 - (100 - self.credit_score) / 200.0)
        bid_multiplier = {
            "Economist": 1.2 if project_type == "Commercial" else 1.0,
            "Environmentalist": 1.5 if project_type in ["Parks", "Metro"] else 1.0,
            "Urban Planner": 1.1,  # Applies to all project types
            "Social Engineer": 1.2 if project_type in ["Residential", "Parks"] else 1.0,
            "Opportunist": random.uniform(0.8, 1.2),  # Varies each time
            "Conservative": 0.9,  # Prefers not to overbid
            "Visionary": 2.0 if project_type == "Metro" else 1.0
        }
        adjusted_bid = project_costs[project_type] * bid_multiplier[self.strategy] * credit_score_factor
        bid_amount = min(self.money, adjusted_bid)
        return bid_amount

    def update_scores(self, project_type, project_costs):
        score_updates = {
            "Residential": {'Prosperity': 2, 'Happiness': 2, 'Environmental': 1, 'ROI': 10},
            "Commercial": {'Prosperity': 5, 'Happiness': 1, 'Environmental': -1, 'ROI': 30},
            "Parks": {'Prosperity': 1, 'Happiness': 4, 'Environmental': 3, 'ROI': 5},
            "Metro": {'Prosperity': 3, 'Happiness': 3, 'Environmental': 5, 'ROI': 20}
        }
        for key, value in score_updates[project_type].items():
            self.scores[key] += value
        self.money -= project_costs[project_type]  # Deduct the cost of the project
        self.projects.append(project_type)
        self.negotiation_targets.clear()  # Clear negotiation targets after updating scores

    def negotiate(self, other_agents):
        # Placeholder for more complex negotiation logic
        # This could involve dynamically deciding which projects to negotiate based on strategy and game state
        pass

def adjust_project_costs(project_costs):
    for project_type in project_costs:
        fluctuation = project_costs[project_type] * DYNAMIC_COST_FLUCTUATION
        project_costs[project_type] += round(random.uniform(-fluctuation, fluctuation))
    return project_costs

def simulate_game(agents, rounds=NUM_ROUNDS):
    project_costs = BASE_PROJECT_COSTS.copy()

    for round_num in range(1, rounds + 1):
        print(f"--- Round {round_num} ---")
        project_costs = adjust_project_costs(project_costs)  # Adjust project costs dynamically

        # Negotiation phase
        for agent in agents:
            other_agents = [a for a in agents if a != agent]
            agent.negotiate(other_agents)  # Placeholder for negotiation logic

        # Bidding phase
        for project_type in PROJECT_TYPES:
            highest_bid = 0
            winning_agent = None
            for agent in agents:
                bid = agent.bid(project_type, project_costs)
                if bid > highest_bid:
                    highest_bid = bid
                    winning_agent = agent
            if winning_agent:
                winning_agent.update_scores(project_type, project_costs)
                print(f"{winning_agent.name} won {project_type} project with a bid of {highest_bid}")
        print("---------------------\n")

# Agents setup
agents = [
    Agent("Economist", "Economist"),
    Agent("Environmentalist", "Environmentalist"),
    Agent("Urban Planner", "Urban Planner"),
    Agent("Social Engineer", "Social Engineer"),
    Agent("Opportunist", "Opportunist"),
    Agent("Conservative", "Conservative"),
    Agent("Visionary", "Visionary")
]

simulate_game(agents)