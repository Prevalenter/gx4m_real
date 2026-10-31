"""GX16 (GX4M hand) hardware files from GEX16: the ROBOTIS Dynamixel SDK and the hand's motor config.

gx4m_real.robot.hand is the driver; see SOURCE.json for what was copied.
"""
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parent/'gx16.yaml'
