from mujoco_playground._src import mjx_env
from etils import epath

class Go1Constants:

  FEET_SITES = [
      "FR",
      "FL",
      "RR",
      "RL",
  ]

  FEET_GEOMS = [
      "FR",
      "FL",
      "RR",
      "RL",
  ]

  FEET_POS_SENSOR = [f"{site}_pos" for site in FEET_SITES]

  ROOT_BODY = "trunk"

  UPVECTOR_SENSOR = "upvector"
  GLOBAL_LINVEL_SENSOR = "global_linvel"
  GLOBAL_ANGVEL_SENSOR = "global_angvel"
  LOCAL_LINVEL_SENSOR = "local_linvel"
  ACCELEROMETER_SENSOR = "accelerometer"
  GYRO_SENSOR = "gyro"

  ROOT_PATH = mjx_env.ROOT_PATH / "locomotion" / "go1"
  XML_ROOT_PATH = mjx_env.ROOT_PATH / "locomotion" / "go1" / "xmls"
  MENAGERIE_XML_PATH = mjx_env.MENAGERIE_PATH / "unitree_go1"
  FEET_ONLY_FLAT_TERRAIN_XML = (
      ROOT_PATH / "xmls" / "scene_mjx_feetonly_flat_terrain.xml"
  )

  FEET_ONLY_ROUGH_TERRAIN_XML = (
      ROOT_PATH / "xmls" / "scene_mjx_feetonly_rough_terrain.xml"
  )
  FULL_FLAT_TERRAIN_XML = ROOT_PATH / "xmls" / "scene_mjx_flat_terrain.xml"
  FULL_COLLISIONS_FLAT_TERRAIN_XML = (
      ROOT_PATH / "xmls" / "scene_mjx_fullcollisions_flat_terrain.xml"
  )

  @classmethod
  def task_to_xml(cls, task_name: str) -> epath.Path:
      task_map = {
          "flat_terrain": cls.FEET_ONLY_FLAT_TERRAIN_XML,
          "rough_terrain": cls.FEET_ONLY_ROUGH_TERRAIN_XML,
      }
      if task_name not in task_map:
          raise ValueError(f"Invalid task name: {task_name}. Must be one of {list(task_map.keys())}")
      return task_map[task_name]