
from typing import Dict
import functools
import matplotlib.pyplot as plt
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
from IPython import display
from IPython.display import HTML, clear_output
from datetime import datetime


from mujoco_playground import wrapper
from mujoco_playground import registry
from mujoco_playground._src import mjx_env
from mujoco_playground._src.env_wrapper.training_env_loader import TrainingEnvLoader
from mujoco_playground._src.env_wrapper.go1_example.default_config import default_config
from mujoco_playground._src.env_wrapper.go1_example.constants import Go1Constants
from mujoco_playground._src.env_wrapper.go1_example.brax_ppo_params import brax_ppo_config
from mujoco_playground._src.env_wrapper.go1_example.randomizer import domain_randomize
from mujoco_playground._src.env_wrapper.go1_example.env import Go1JoystickEnv


x_data, y_data, y_dataerr = [], [], []
times = [datetime.now()]

def progress(num_steps, metrics):
  clear_output(wait=True)

  times.append(datetime.now())
  x_data.append(num_steps)
  y_data.append(metrics["eval/episode_reward"])
  y_dataerr.append(metrics["eval/episode_reward_std"])

  plt.xlim([0, ppo_params["num_timesteps"] * 1.25])
  plt.xlabel("# environment steps")
  plt.ylabel("reward per episode")
  plt.title(f"y={y_data[-1]:.3f}")
  plt.errorbar(x_data, y_data, yerr=y_dataerr, color="blue")

  display(plt.gcf())



class Go1JoystickFlatTerrainLoader(TrainingEnvLoader):

  def set_model_constants(self):
    return Go1Constants

  def get_assets(self) -> Dict[str, bytes]:
    assets = {}
    xml_path = self.model_constants.XML_ROOT_PATH
    assets_path = xml_path / "assets"

    # menagerie_xml_path = mjx_env.MENAGERIE_PATH / "unitree_go1"
    menagerie_xml_path = self.model_constants.MENAGERIE_XML_PATH
    menagerie_assets_path = menagerie_xml_path / "assets"

    mjx_env.update_assets(assets, xml_path, "*.xml")
    mjx_env.update_assets(assets, assets_path)

    mjx_env.update_assets(assets, menagerie_xml_path, "*.xml")
    mjx_env.update_assets(assets, menagerie_assets_path)
    return assets


  def set_default_config(self):
    return default_config()

  def get_brax_ppo_config(self):
    return brax_ppo_config(self.config)

  def get_randomizer(self):
    return domain_randomize

  def get_env(self):
    return Go1JoystickEnv(
        self.model_constants,
        get_assets=self.get_assets,
        task="flat_terrain",
        config=self.config,
    )


go1_loader = Go1JoystickFlatTerrainLoader()
go1_js_env_test = go1_loader.get_env()
ppo_params = go1_loader.get_brax_ppo_config()

env_name = 'Go1JoystickFlatTerrain'
env_cfg_ref = registry.get_default_config(env_name)
# env_ref = registry.load(env_name)


ppo_training_params = dict(ppo_params)
network_factory = ppo_networks.make_ppo_networks
if "network_factory" in ppo_params:
  del ppo_training_params["network_factory"]
  network_factory = functools.partial(
      ppo_networks.make_ppo_networks,
      **ppo_params.network_factory
  )

train_fn = functools.partial(
    ppo.train, **dict(ppo_training_params),
    network_factory=network_factory,
    randomization_fn=domain_randomize, #randomizer,
    progress_fn=progress
)



make_inference_fn, params, metrics = train_fn(
    environment=go1_js_env_test, #env_ref,
    eval_env=registry.load(env_name, config=env_cfg_ref),
    wrap_env_fn=wrapper.wrap_for_brax_training,
)
print(f"time to jit: {times[1] - times[0]}")
print(f"time to train: {times[-1] - times[1]}")
