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
        self.pending_agreements = []  # To track agreements for cashing out

    def cash_out(self):
        # Decision to honor or renege on agreements
        for agreement in list(self.pending_agreements):
            if random.choice([True, False]):  # Decision to honor the payment
                self.money -= agreement['amount']
                agreement['payee'].money += agreement['amount']  # Transfer the incentive
                self.reputation += 10  # Increase reputation for honoring the agreement
                print(f"{self.name} honors payment of {agreement['amount']} to {agreement['payee'].name}")
            else:
                self.reputation -= 20  # Decrease reputation for reneging
                print(f"{self.name} reneges on payment of {agreement['amount']} to {agreement['payee'].name}")
            self.pending_agreements.remove(agreement)  # Remove the processed agreement
            
    def bid(self, project_type, project_costs):
        strategy_modifiers = {
            "Economist": 1.5 if project_type == "Commercial" else 0.8,
            "Environmentalist": 1.5 if project_type in ["Parks", "Metro"] else 0.5,
            "Urban Planner": 1.2,  # Balanced bidding on all types
            "Social Engineer": 1.5 if project_type in ["Residential", "Parks"] else 0.8,
            "Opportunist": random.uniform(0.8, 1.5),  # Dynamic bidding based on market conditions
            "Conservative": 0.9,  # Safe, low bids
            "Visionary": 2.0 if project_type == "Metro" else 1.0  # Long-term focus
        }
        bid = project_costs[project_type] * strategy_modifiers[self.strategy]
        return min(self.money, bid)  # Ensure the bid does not exceed the agent's available money

    def update_scores(self, project_type, project_cost):
        # Score increments by project type. These values should be adjusted based on game design needs.
        score_increments = {
            'Residential': {'Prosperity': 2, 'Happiness': 2, 'Environmental': 1, 'ROI': 10},
            'Commercial': {'Prosperity': 5, 'Happiness': 1, 'Environmental': -1, 'ROI': 30},
            'Parks': {'Prosperity': 1, 'Happiness': 4, 'Environmental': 3, 'ROI': 5},
            'Metro': {'Prosperity': 3, 'Happiness': 3, 'Environmental': 5, 'ROI': 20}
        }

        # Update scores based on the specific project type attributes
        increments = score_increments.get(project_type, {})
        for key, value in increments.items():
            self.scores[key] += value

        # Deduct the project cost from the agent's money
        self.money -= project_cost
        self.projects.append({'type': project_type, 'ownership': 'single', 'consent': True})  # Default to full ownership

    def propose_cooperation(self, target_agent, incentive, condition):
        # Include reputation in the proposal to influence the decision
        target_agent.received_proposals.append((self, incentive, condition, self.reputation))

    def evaluate_proposals(self):
        accepted = []
        for proposer, incentive, condition, rep in self.received_proposals:
            # Accept based on proposer's reputation and a strategy compatibility check
            if condition in self.strategy and proposer.reputation > 75:  # example threshold
                self.money += incentive
                proposer.reputation += 5  # Improve reputation for successful cooperation
                accepted.append((proposer.name, incentive, condition))
                # Update project ownership if proposal involves a project
                for project in self.projects:
                    if project['type'] == condition:  # Example condition matches project type
                        project['ownership'] = 'joint'
                        project['partner'] = proposer.name
                        project['consent'] = False  # Consent needed from both sides
            else:
                proposer.reputation -= 5  # Penalty for failed cooperation attempt
        self.received_proposals = []  # Clear proposals after evaluation
        return accepted

    def auction_projects(self):
        earnings = 0
        for project in list(self.projects):
            if project['ownership'] == 'single' or (project['ownership'] == 'joint' and project['consent']):
                sale_price = BASE_PROJECT_COSTS[project['type']] * DISCOUNT_RATE
                earnings += sale_price
                self.projects.remove(project)
        self.money += earnings
        return earnings

    def give_consent(self):
        for project in self.projects:
            if project['ownership'] == 'joint':
                project['consent'] = True  # Give consent for all joint projects
    
    def cash_out(self):
        # Decision to finalize or renege on agreements
        for proposal in self.accepted_proposals:
            if random.choice([True, False]):  # Random decision for demonstration
                self.money += proposal.incentive  # Cash out and finalize the agreement
                self.reputation += 10  # Boost reputation for honoring the agreement
            else:
                self.reputation -= 20  # Major reputation penalty for reneging

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
        
        # Auction projects
        for agent in agents:
            earnings = agent.auction_projects()
            print(f"{agent.name} earned {earnings} from auctions.")

        # Agents propose cooperation
        for agent in agents:
            if agents.index(agent) < len(agents) - 1:
                agent.propose_cooperation(agents[agents.index(agent) + 1], 10, "Cooperate on Metro")

        # Agents evaluate received proposals and give consent
        for agent in agents:
            accepted = agent.evaluate_proposals()
            agent.give_consent()  # Agents give consent to joint projects
            for deal in accepted:
                print(f"{agent.name} accepted a deal from {deal[0]} for {deal[1]} on condition '{deal[2]}'")

        # Participate in the auction
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

# Initialize and run the simulation
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
