import logging
import os
from typing import Optional


logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))


class RolloutDrafterManager:
    def __init__(self, rollout_config, dp_rank: Optional[int] = None, device_mesh=None):
        self.train_drafter = rollout_config
        self.dp_rank = dp_rank
        self.device_mesh = device_mesh
        self.background_trainer = None

        self.current_rl_step = 0
        self.training_interval_steps = rollout_config.drafter.training.get("training_interval_steps", 1)
        self.collection_interval_steps = rollout_config.drafter.training.get("collect_interval_steps", 1)
        self.step = rollout_config.drafter.training.get("step", 100)

    async def initialize(self):
        return None

    async def run_training_loop(self):
        if self.background_trainer is None:
            return False
        if self.should_train_this_step() and self.background_trainer.has_training_data():
            success = await self.background_trainer.training_step(self.step)
            if success:
                logger.info("Successfully trained drafter.")
            return success
        return False

    def should_train_this_step(self):
        if not self.train_drafter:
            return False
        return self.current_rl_step % self.training_interval_steps == 0

    def should_collect_data_this_step(self):
        if not self.train_drafter:
            return False
        return self.current_rl_step % self.collection_interval_steps == 0

    def collect_online_data(self, batch, hidden_states, last_hidden_states=None):
        if self.background_trainer is None or not self.should_collect_data_this_step():
            return
        self.background_trainer.collect_online_data(batch, hidden_states, last_hidden_states=last_hidden_states)

    def increment_rl_step(self):
        self.current_rl_step += 1
        if self.background_trainer is not None:
            self.background_trainer.update_rl_step(self.current_rl_step)

    def update_rl_step(self, global_step: Optional[int] = None):
        if global_step is not None:
            self.current_rl_step = global_step
            if self.background_trainer is not None:
                self.background_trainer.update_rl_step(self.current_rl_step)
        logger.debug(f"RolloutDrafterManager RL step updates to {self.current_rl_step}")

    def maybe_publish(self):
        if self.background_trainer is None:
            return None
        if self.should_train_this_step():
            return self.background_trainer.get_model_state_dict()
        return None
