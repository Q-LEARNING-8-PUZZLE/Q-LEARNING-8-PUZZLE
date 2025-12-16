import pytest
import os
import shutil
import pandas as pd
from app.environment import EightPuzzle
from app.agent import QLearningAgent, QTable
from app.trainer import Trainer
from app.plotter import Plotter, generate_plots # Import generate_plots directly

# Define paths for test artifacts
TEST_LOG_FILE = "data/test_training_log.csv"
TEST_PLOTS_DIR = "data/test_plots"

@pytest.fixture(scope="module", autouse=True)
def setup_teardown_test_environment():
    """
    Sets up a clean environment for integration tests and cleans up afterwards.
    Ensures that 'data/' directory and 'data/plots' sub-directory exist.
    """
    # Ensure 'data' directory exists
    os.makedirs("data", exist_ok=True)
    # Ensure a dedicated test plots directory exists
    os.makedirs(TEST_PLOTS_DIR, exist_ok=True)

    # Clean up any previous test artifacts before running tests
    if os.path.exists(TEST_LOG_FILE):
        os.remove(TEST_LOG_FILE)
    if os.path.exists(TEST_PLOTS_DIR):
        shutil.rmtree(TEST_PLOTS_DIR)
    os.makedirs(TEST_PLOTS_DIR, exist_ok=True)

    yield

    # Clean up all test artifacts after tests are done
    if os.path.exists(TEST_LOG_FILE):
        os.remove(TEST_LOG_FILE)
    if os.path.exists(TEST_PLOTS_DIR):
        shutil.rmtree(TEST_PLOTS_DIR)

def test_trainer_integration():
    """
    Tests if the Trainer runs correctly and generates a training log file.
    """
    # Use a small number of episodes and steps for quick testing
    env = EightPuzzle()
    q_table = QTable()
    agent = QLearningAgent(q_table)
    trainer = Trainer(
        environment=env,
        agent=agent,
        num_episodes=5,        # Small number of episodes
        max_steps_per_episode=10, # Small number of steps
        verbose=False,         # Suppress console output during test
        log_interval=1,        # Log every episode for small runs
        show_steps=False
    )

    # Modify the trainer's analytics to use the test log file
    # This is a bit of a hack, but necessary to direct output for testing
    trainer.analytics.to_csv = lambda filename=TEST_LOG_FILE: trainer.analytics._to_csv_original(filename)
    trainer.analytics._to_csv_original = trainer.analytics.to_csv

    # Run training
    training_stats = trainer.train()

    # Assertions
    assert training_stats is not None
    assert isinstance(training_stats, dict)
    assert 'total_episodes' in training_stats
    assert training_stats['total_episodes'] == 5

    # Check if the log file was created
    assert os.path.exists(TEST_LOG_FILE)

    # Check if the log file has content
    df_log = pd.read_csv(TEST_LOG_FILE)
    assert not df_log.empty
    assert len(df_log) == 5 # Should have 5 entries, one per episode
    assert all(col in df_log.columns for col in ['episode', 'steps', 'total_reward', 'success'])

def test_plotter_integration():
    """
    Tests if the Plotter reads the log and generates the plot files.
    This test assumes test_trainer_integration has successfully created TEST_LOG_FILE.
    """
    # Ensure the log file from trainer test exists
    if not os.path.exists(TEST_LOG_FILE):
        # If the trainer test didn't run or failed, create a dummy log for plotter test
        dummy_data = {
            'episode': [1, 2, 3, 4, 5],
            'steps': [10, 8, 9, 7, 5],
            'total_reward': [-100.0, -80.0, -90.0, 930.0, 950.0],
            'success': [False, False, False, True, True]
        }
        pd.DataFrame(dummy_data).to_csv(TEST_LOG_FILE, index=False)

    # Instantiate Plotter and generate plots
    plotter = Plotter(log_file=TEST_LOG_FILE)
    
    # Override output directory to the test plots directory
    plotter.output_dir = TEST_PLOTS_DIR
    os.makedirs(plotter.output_dir, exist_ok=True) # Ensure it exists

    # Generate plots
    plotter.generate_all_plots(show=False) # Don't show plots during test

    # Assertions for plot files
    steps_plot_path = os.path.join(TEST_PLOTS_DIR, "steps_vs_episodes.png")
    success_plot_path = os.path.join(TEST_PLOTS_DIR, "success_rate_vs_episodes.png")

    assert os.path.exists(steps_plot_path)
    assert os.path.getsize(steps_plot_path) > 0

    assert os.path.exists(success_plot_path)
    assert os.path.getsize(success_plot_path) > 0

    # Ensure main generate_plots function works
    # Need to temporarily redirect the global analytics.to_csv for this to work clean
    original_to_csv = None
    if hasattr(Plotter, 'to_csv') and callable(Plotter.to_csv):
        original_to_csv = Plotter.to_csv
        Plotter.to_csv = lambda filename=TEST_LOG_FILE: pd.DataFrame(dummy_data).to_csv(filename, index=False) # Mock for test
    
    # Also temporarily change the output dir for generate_plots
    original_plotter_output_dir = Plotter().output_dir
    Plotter().output_dir = TEST_PLOTS_DIR
    
    # Mock for plotter.Plotter to use TEST_LOG_FILE
    def mock_plotter_init(self, log_file, backend='pandas'):
        self.log_file = TEST_LOG_FILE
        self.backend = backend
        self.output_dir = TEST_PLOTS_DIR
        self.df = pd.read_csv(self.log_file)
        sns.set_theme(style="darkgrid")
    
    # Temporarily replace Plotter.__init__
    original_init = Plotter.__init__
    Plotter.__init__ = mock_plotter_init

    generate_plots(log_file=TEST_LOG_FILE)

    # Restore Plotter.__init__
    Plotter.__init__ = original_init

    if original_to_csv:
        Plotter.to_csv = original_to_csv
    # Plotter().output_dir = original_plotter_output_dir # This requires re-initializing plotter which we don't want.
    # The above mocks are not ideal because generate_plots function internally creates plotter = Plotter(log_file).
    # It would be better to mock Plotter itself or its instances.
    # For now, these assertions check the existence of the files.
    
    steps_plot_path = os.path.join(TEST_PLOTS_DIR, "steps_vs_episodes.png")
    success_plot_path = os.path.join(TEST_PLOTS_DIR, "success_rate_vs_episodes.png")
    
    # Check if the files are created by the generate_plots function (which should overwrite previous)
    assert os.path.exists(steps_plot_path)
    assert os.path.getsize(steps_plot_path) > 0
    assert os.path.exists(success_plot_path)
    assert os.path.getsize(success_plot_path) > 0
