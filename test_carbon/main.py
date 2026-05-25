import sys
import os

from acc import ACCSetup

if __name__ == "__main__":
    simulation = ACCSetup()
    simulation.setup()
    simulation.run()
