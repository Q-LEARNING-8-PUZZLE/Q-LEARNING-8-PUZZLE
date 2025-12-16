import pytest
import numpy as np
from app.environment import EightPuzzle
from app.agent import QLearningAgent, QTable
from app.config import REWARD_GOAL, REWARD_STEP, REWARD_INVALID

# Fixture for a default 8-Puzzle environment
@pytest.fixture
def default_env():
    return EightPuzzle()

# Fixture for a QTable
@pytest.fixture
def q_table():
    return QTable(initial_value=0.0)

# Fixture for a QLearningAgent
@pytest.fixture
def agent(q_table):
    return QLearningAgent(q_table=q_table, epsilon=0.1, alpha=0.1, gamma=0.9)

# --- EightPuzzle Tests ---

def test_eight_puzzle_init(default_env):
    assert default_env.GOAL_STATE == (1, 2, 3, 4, 5, 6, 7, 8, 0)
    assert default_env.reward_goal == REWARD_GOAL
    assert default_env.reward_step == REWARD_STEP
    assert default_env.reward_invalid == REWARD_INVALID

def test_eight_puzzle_reset(default_env):
    initial_state = default_env.reset(random_start=False)
    assert initial_state == default_env.GOAL_STATE
    
    # Test random start
    random_state = default_env.reset(random_start=True, shuffles=5)
    assert random_state != default_env.GOAL_STATE or list(random_state).count(0) == 1

def test_eight_puzzle_is_goal(default_env):
    assert default_env.is_goal(default_env.GOAL_STATE) == True
    assert default_env.is_goal((1, 2, 3, 4, 5, 6, 7, 0, 8)) == False

def test_eight_puzzle_get_valid_actions_middle(default_env):
    default_env.state = (1, 2, 3, 4, 0, 5, 6, 7, 8) # 0 in middle
    actions = default_env.get_valid_actions()
    assert set(actions) == {0, 1, 2, 3} # UP, DOWN, LEFT, RIGHT

def test_eight_puzzle_get_valid_actions_corner(default_env):
    default_env.state = (0, 1, 2, 3, 4, 5, 6, 7, 8) # 0 in top-left corner
    actions = default_env.get_valid_actions()
    assert set(actions) == {1, 3} # DOWN, RIGHT

def test_eight_puzzle_step_valid(default_env):
    initial_state = (1, 2, 3, 4, 0, 5, 6, 7, 8)
    default_env.state = initial_state
    
    # Move UP (action 0)
    new_state, valid = default_env.step(0, verbose=False)
    assert valid == True
    assert new_state == (1, 0, 3, 4, 2, 5, 6, 7, 8)

def test_eight_puzzle_step_invalid(default_env):
    initial_state = (0, 1, 2, 3, 4, 5, 6, 7, 8) # 0 in top-left
    default_env.state = initial_state
    
    # Move UP (action 0) - invalid
    new_state, valid = default_env.step(0, verbose=False)
    assert valid == False
    assert new_state == initial_state # State should not change for invalid move

def test_eight_puzzle_get_reward(default_env):
    # Goal state
    assert default_env.get_reward(default_env.GOAL_STATE, True) == REWARD_GOAL
    # Normal step
    assert default_env.get_reward((1,2,3,4,5,6,7,0,8), True) == REWARD_STEP
    # Invalid move
    assert default_env.get_reward((1,2,3,4,5,6,7,8,0), False) == REWARD_INVALID

def test_eight_puzzle_render(default_env, capsys):
    default_env.state = (1, 2, 3, 4, 5, 6, 7, 8, 0)
    default_env.render(return_string=False)
    captured = capsys.readouterr()
    assert "1 2 3" in captured.out
    assert "4 5 6" in captured.out
    assert "7 8  " in captured.out

    rendered_str = default_env.render(return_string=True)
    assert "1 2 3" in rendered_str
    assert "4 5 6" in rendered_str
    assert "7 8  " in rendered_str


# --- QTable Tests ---

def test_q_table_init(q_table):
    assert len(q_table) == 0
    assert q_table.initial_value == 0.0

def test_q_table_get_set(q_table):
    state = (1, 2, 3, 4, 5, 6, 7, 8, 0)
    action = 0 # UP
    
    # Access a new (state, action) pair
    assert q_table[(state, action)] == 0.0 # Should return initial_value
    assert len(q_table) == 1 # Should have added the pair

    # Set a value
    q_table[(state, action)] = 10.5
    assert q_table[(state, action)] == 10.5

def test_q_table_len(q_table):
    state1 = (1, 2, 3, 4, 5, 6, 7, 8, 0)
    action1 = 0
    state2 = (1, 2, 3, 4, 5, 6, 7, 0, 8)
    action2 = 1
    
    q_table[(state1, action1)] # Adds one entry
    q_table[(state2, action2)] # Adds another entry
    q_table[(state1, action1)] # Accesses existing, doesn't add
    
    assert len(q_table) == 2


# --- QLearningAgent Tests ---

def test_q_learning_agent_init(agent, q_table):
    assert agent.q_table == q_table
    assert agent.epsilon == 0.1
    assert agent.alpha == 0.1
    assert agent.gamma == 0.9

def test_q_learning_agent_set_epsilon(agent):
    agent.set_epsilon(0.5)
    assert agent.epsilon == 0.5

def test_q_learning_agent_choose_action_exploration(agent):
    agent.set_epsilon(1.0) # Always explore
    state = (1, 2, 3, 4, 0, 5, 6, 7, 8)
    valid_actions = [0, 1, 2, 3]
    action = agent.choose_action(state, valid_actions)
    assert action in valid_actions

def test_q_learning_agent_choose_action_exploitation(agent):
    agent.set_epsilon(0.0) # Always exploit
    state = (1, 2, 3, 4, 0, 5, 6, 7, 8)
    valid_actions = [0, 1, 2, 3]
    
    # Manually set Q-values to make '2' the best action
    agent.q_table[(state, 0)] = 0.1
    agent.q_table[(state, 1)] = 0.2
    agent.q_table[(state, 2)] = 10.0 # Best action
    agent.q_table[(state, 3)] = 0.5
    
    action = agent.choose_action(state, valid_actions)
    assert action == 2

def test_q_learning_agent_update(agent):
    state = (1, 2, 3, 4, 0, 5, 6, 7, 8)
    action = 0 # UP
    reward = -10.0
    next_state = (1, 0, 3, 4, 2, 5, 6, 7, 8)
    next_valid_actions = [0, 1, 2, 3] # Example valid actions for next_state
    
    # Ensure initial Q-value is 0
    assert agent.q_table[(state, action)] == 0.0
    
    # Manually set a Q-value for next_state to test gamma
    agent.q_table[(next_state, 1)] = 50.0 # Max Q for next_state
    
    old_q_value = agent.q_table[(state, action)]
    agent.update(state, action, reward, next_state, next_valid_actions)
    new_q_value = agent.q_table[(state, action)]
    
    # Expected calculation: Q(s,a) = Q(s,a) + alpha * [r + gamma * max(Q(s',a')) - Q(s,a)]
    # 0.0 + 0.1 * [-10.0 + 0.9 * 50.0 - 0.0]
    # 0.1 * [-10.0 + 45.0]
    # 0.1 * 35.0 = 3.5
    expected_q_value = 3.5
    
    assert new_q_value == pytest.approx(expected_q_value)
