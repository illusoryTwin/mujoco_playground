from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Union, Type

class TrainingEnvLoader(ABC):
  """
  Abstract base class for loading model constants, configuration, and assets
  for further usage while training.
  """
  def __init__(self):
    self.config = self.set_default_config()
    self.model_constants = self.set_model_constants()

  @abstractmethod
  def set_model_constants(self) -> Type:
    pass


  @abstractmethod
  def get_assets(self) -> Dict[str, bytes]:
    pass


  @abstractmethod
  def set_default_config(self):
    pass


  @abstractmethod
  def get_brax_ppo_config(self):
    pass

  @abstractmethod
  def get_randomizer(self):
    pass

  @abstractmethod
  def get_env(self):
    pass
