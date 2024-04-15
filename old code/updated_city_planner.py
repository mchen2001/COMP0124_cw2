import random

# Constants and Initial Setup
INITIAL_MONEY = 100
INITIAL_CREDIT_SCORE = 100
PROJECT_TYPES = ['Residential', 'Commercial', 'Parks', 'Metro']
BASE_PROJECT_COSTS = {'Residential': 20, 'Commercial': 40, 'Parks': 15, 'Metro': 50}
SCORES_TEMPLATE = {'Prosperity': 0, 'Happiness': 0, 'Environmental': 0, 'ROI': 0}
NUM_ROUNDS = 10
CREDIT_SCORE_IMPACT = 10
DYNAMIC_COST_FLUCTUATION = 0.1
NEGOTIATION_PENALTY = 10
DISCOUNT_RATE = 0.85

class Agent:
    def __init__(self, name, strategy):
        self.name = name
        self.strategy = strategy
        self.money = INITIAL_MONEY
        self.credit_score = INITIAL_CREDIT_SCORE
        self.projects = []  # Projects are now (type, ownership status) tuples
        self.scores = SCORES_TEMPLATE.copy()
        self.reputation = 100  # New attribute for reputation
        self.received_proposals = []

    def bid(self, project_type, project_costs, minimum_bid):
        # Example strategy: bid the minimum plus a strategy-based increment
        increment = random.uniform(0.1, 0.5) * self.reputation / 100
        bid = minimum_bid + (increment * minimum_bid)
        return min(self.money, bid) if self.money >= minimum_bid else 0

    def update_scores(self, project_type, base_project_cost):
        # Simplified score update logic
        self.scores['Prosperity'] += 2  # Example increment
        self.money -= base_project_cost
        self.projects.append((project_type, True))  # Assume full ownership

    def propose_cooperation(self, target_agent, incentive, condition):
        target_agent.received_proposals.append((self, incentive, condition))

    def evaluate_proposals(self):
        accepted = []
        for proposer, incentive, condition in self.received_proposals:
            if condition in self.strategy:  # Simplified evaluation based on strategy
                self.money += incentive
                accepted.append((proposer.name, incentive, condition))
        self.received_proposals = []  # Clear proposals after evaluation
        return accepted

    def auction_projects(self):
        # Placeholder for auction logic, returns money earned from auction
        earnings = 0
        for project in self.projects:
            if project[1]:  # Check ownership status
                sale_price = BASE_PROJECT_COSTS[project[0]] * DISCOUNT_RATE
                earnings += sale_price
                self.projects.remove(project)
        self.money += earnings
        return earnings

def adjust_project_costs(project_costs):
    for project_type in project_costs:
        fluctuation = project_costs[project_type] * DYNAMIC_COST_FLUCTUATION
        project_costs[project_type] += round(random.uniform(-fluctuation, fluctuation))
    return project_costs

def simulate_game(agents, num_rounds=NUM_ROUNDS):
    for round_num in range(1, num_rounds + 1):
        print(f"\n--- Round {round_num} ---")

        # Auction projects
        for agent in agents:
            earnings = agent.auction_projects()
            print(f"{agent.name} earned {earnings} from auctions.")

        # Adjust project costs for the round
        project_costs = adjust_project_costs(BASE_PROJECT_COSTS.copy())
        minimum_bid = min(project_costs.values())  # Assuming a minimum bid requirement

        # Agents propose cooperation
        for agent in agents:
            # Example: target the first agent for simplicity
            if agents.index(agent) < len(agents) - 1:
                agent.propose_cooperation(agents[agents.index(agent) + 1], 10, "Cooperate on Metro")

        # Agents evaluate received proposals
        for agent in agents:
            accepted = agent.evaluate_proposals()
            for deal in accepted:
                print(f"{agent.name} accepted a deal from {deal[0]} for {deal[1]} on condition '{deal[2]}'")

        # Participate in the auction
        for project_type in PROJECT_TYPES:
            highest_bid = 0
            winning_agent = None
            for agent in agents:
                bid = agent.bid(project_type, project_costs, minimum_bid)
                if bid > highest_bid:
                    highest_bid = bid
                    winning_agent = agent
            if winning_agent:
                winning_agent.update_scores(project_type, project_costs[project_type])
                print(f"{winning_agent.name} won the {project_type} project with a bid of {highest_bid}")

# Initialize agents with different strategies
agents = [
    Agent("Agent 1", "Economist"),
    Agent("Agent 2", "Environmentalist"),
    Agent("Agent 3", "Urban Planner"),
    Agent("Agent 4", "Social Engineer"),
    Agent("Agent 5", "Opportunist"),
    Agent("Agent 6", "Conservative"),
    Agent("Agent 7", "Visionary")
]

simulate_game(agents)
