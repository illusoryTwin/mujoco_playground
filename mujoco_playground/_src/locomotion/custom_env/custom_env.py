"""Base class for any custom model"""
from typing import Any, Dict, Optional, Union

from etils import epath
import jax
from ml_collections import config_dict
import mujoco
from mujoco import mjx

from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion.g1 import g1_constants as consts

def get_assets() -> Dict[str, bytes]:
  assets = {}
  mjx_env.update_assets(assets, consts.ROOT_PATH / "xmls", "*.xml")
  mjx_env.update_assets(assets, consts.ROOT_PATH / "xmls" / "assets")
  path = mjx_env.MENAGERIE_PATH / "unitree_g1"
  mjx_env.update_assets(assets, path, "*.xml")
  mjx_env.update_assets(assets, path / "assets")
  return assets


class CustomEnv(mjx_env.MjxEnv):
    
        def __init__(
                self, 
                xml_path: str,
                config: config_dict.ConfigDict,
                config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
        ):
                super().__init__(config, config_overrides)

                # self.mj_model = mujcoo.Mj_model.from_xml_path()
                self._mj_model = mujoco.MjModel.from_xml_string(epath.Path(xml_path).read_text(), assets=get_assets())
                self.mjx_model = mjx.put_model()
                self._xml_path = xml_path 

        
        @property 
        def xml_path(self) -> str:
                return self._xml_path 
        
        @property 
        def mj_model(self) -> mujoco.MjModel:
                return self._mj_model

        @property 
        def mjx_model(self) -> mjx.Model:
                return self._mjx_model