import random

# Constants
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
        self.reputation = 100  # Initialize reputation
        self.projects = []  # Projects are now stored as dictionaries
        self.scores = SCORES_TEMPLATE.copy()
        self.received_proposals = []
        self.pending_payments = []  # To handle cashing out

    def bid(self, project_type, project_costs):
        if any(agreement['project_to_avoid'] == project_type for agreement in self.pending_payments):
            return 0  # Agent has agreed not to bid on this project
        strategy_modifiers = {
            "Economist": 1.5 if project_type == "Commercial" else 0.8,
            "Environmentalist": 1.5 if project_type in ["Parks", "Metro"] else 0.5,
            "Urban Planner": 1.2,
            "Social Engineer": 1.5 if project_type in ["Residential", "Parks"] else 0.8,
            "Opportunist": random.uniform(0.8, 1.5),
            "Conservative": 0.9,
            "Visionary": 2.0 if project_type == "Metro" else 1.0
        }
        bid = project_costs[project_type] * strategy_modifiers[self.strategy]
        return min(self.money, bid)

    def update_scores(self, project_type, project_cost):
        score_increments = {
            'Residential': {'Prosperity': 2, 'Happiness': 2, 'Environmental': 1, 'ROI': 10},
            'Commercial': {'Prosperity': 5, 'Happiness': 1, 'Environmental': -1, 'ROI': 30},
            'Parks': {'Prosperity': 1, 'Happiness': 4, 'Environmental': 3, 'ROI': 5},
            'Metro': {'Prosperity': 3, 'Happiness': 3, 'Environmental': 5, 'ROI': 20}
        }
        increments = score_increments.get(project_type, {})
        for key, value in increments.items():
            self.scores[key] += value
        self.money -= project_cost
        self.projects.append({'type': project_type, 'ownership': 'single', 'consent': True})

    def propose_cooperation(self, target_agent, incentive, project_to_avoid):
        proposal = {'proposer': self, 'incentive': incentive, 'project_to_avoid': project_to_avoid}
        target_agent.received_proposals.append(proposal)

    def evaluate_proposals(self):
        for proposal in self.received_proposals:
            if proposal['project_to_avoid'] in [p['type'] for p in self.projects]:
                continue  # Skip if agent already owns the project
            if random.choice([True, False]):  # Randomly accepting proposals for simplicity
                self.pending_payments.append(proposal)
                print(f"{self.name} agreed not to bid on {proposal['project_to_avoid']} for ${proposal['incentive']}")

    def cash_out(self):
        # Decision to finalize or renege on agreements
        while self.pending_payments:
            payment = self.pending_payments.pop()
            if random.choice([True, False]):  # Random decision to honor the payment
                payment['proposer'].money += payment['incentive']
                self.reputation += 10
                print(f"{self.name} has paid {payment['incentive']} to {payment['proposer'].name}")
            else:
                self.reputation -= 20
                print(f"{self.name} decided not to pay {payment['incentive']} to {payment['proposer'].name}")

def adjust_project_costs(project_costs):
    for project_type in project_costs:
        fluctuation = project_costs[project_type] * DYNAMIC_COST_FLUCTUATION
        project_costs[project_type] += round(random.uniform(-fluctuation, fluctuation))
    return project_costs

def simulate_game(agents, num_rounds=NUM_ROUNDS):
    for round_num in range(1, num_rounds + 1):
        print(f"\n--- Round {round_num} ---")
        available_projects = random.sample(PROJECT_TYPES * ((len(agents) // len(PROJECT_TYPES)) + 1), len(agents))
        print(f"Projects available this round: {available_projects}")
        project_costs = adjust_project_costs(BASE_PROJECT_COSTS.copy())

        for agent in agents:
            agent.evaluate_proposals()
            agent.cash_out()
            earnings = agent.auction_projects()
            print(f"{agent.name} earned {earnings} from auctions.")

        for project_type in available_projects:
            highest_bid = 0
            winning_agent = None
            for agent in agents:
                bid = agent.bid(project_type, project_costs)
                if bid > highest_bid:
                    highest_bid = bid
                    winning_agent = agent
            if winning_agent:
                winning_agent.update_scores(project_type, project_costs[project_type])
                print(f"{winning_agent.name} won the {project_type} project with a bid of {highest_bid}")

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
