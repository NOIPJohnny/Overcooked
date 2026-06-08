from stable_baselines3.common.callbacks import BaseCallback
from tqdm.auto import tqdm


class TqdmProgressCallback(BaseCallback):
    def __init__(self, total_timesteps: int):
        super().__init__()
        self.total_timesteps = total_timesteps
        self.progress_bar = None
        self.last_timesteps = 0

    def _on_training_start(self) -> None:
        self.last_timesteps = self.num_timesteps
        self.progress_bar = tqdm(total=self.total_timesteps, unit="step")

    def _on_step(self) -> bool:
        if self.progress_bar is None:
            return True
        delta = self.num_timesteps - self.last_timesteps
        if delta > 0:
            self.progress_bar.update(delta)
            self.last_timesteps = self.num_timesteps
        return True

    def _on_training_end(self) -> None:
        if self.progress_bar is not None:
            remaining = self.total_timesteps - self.progress_bar.n
            if remaining > 0:
                self.progress_bar.update(remaining)
            self.progress_bar.close()
            self.progress_bar = None
