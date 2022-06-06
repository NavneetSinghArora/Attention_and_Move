from os.path import dirname, join
from pathlib import Path


def make_scene_name(type, num):
    if type == "":
        return "FloorPlan" + str(num) + "_physics"
    elif num < 10:
        return "FloorPlan" + type + "0" + str(num) + "_physics"
    else:
        return "FloorPlan" + type + str(num) + "_physics"

def rotate_tuple_90_clockwise(t):
    return (t[1], -t[0])

CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'])
PROJECT_ROOT_DIR = dirname(Path(__file__).resolve().parents[2])

ABS_PATH_TO_ANALYSIS_RESULTS_DIR = join(PROJECT_ROOT_DIR, "output/analysis")
ABS_PATH_TO_DATA_DIR = join(PROJECT_ROOT_DIR, "data/internal")

AGENT_SELF_0 = 6
AGENT_SELF_90 = 7
AGENT_SELF_180 = 8
AGENT_SELF_270 = 9

AGENT_OTHER_0 = 10
AGENT_OTHER_90 = 11
AGENT_OTHER_180 = 12
AGENT_OTHER_270 = 13

AGENT_SYM = 2

CARD_DIR_STRS = ["North", "East", "South", "West"]
EGO_DIR_STRS = ["Ahead", "Right", "Back", "Left"]
EXPLORATION_BONUS = 0.5
FAILED_ACTION_PENALTY = -0.02
FOV = 90.0
GOAL_OBJ_SYM = 4

JOINT_MOVE_WITH_OBJECT_KEYWORD = "WithObject"
JOINT_MOVE_OBJECT_KEYWORD = "MoveLifted"
JOINT_PASS_PENALTY = -0.1
JOINT_ROTATE_KEYWORD = "RotateLifted"

MOVE_MAP = {
    0: dict(row=-1, col=0),
    90: dict(row=0, col=1),
    180: dict(row=1, col=0),
    270: dict(row=0, col=-1),
}

NO_INFO_SYM = 3
PICK_UP_HEIGHT = 1.0
REACHABLE_SYM = 1
ROTATE_OBJECT_ACTIONS = ["RotateLiftedObjectLeft", "RotateLiftedObjectRight"]

SCENES_NAMES_SPLIT_BY_TYPE = [
    [make_scene_name(j, i) for i in range(1, 31)] for j in ["", "2", "3", "4"]
]

SMALL_TELEVISION_TEMPLATE_STRING = """
0 0 2 0 0
0 2 1 2 0
0 2 1 1 2 
0 2 1 2 0
0 0 2 0 0 
"""

STEP_PENALTY = -0.01

TELEVISION_ROT_0_POINTS = ([(0.25 * i, 0.25) for i in [-1, 0, 1]] + [(0.25 * i, 0) for i in [-2, -1, 0, 1, 2]] + [(0.25 * i, -0.25) for i in [-2, -1, 0, 1, 2]])
TELEVISION_ROT_45_POINTS = [rotate_tuple_90_clockwise(t) for t in TELEVISION_ROT_0_POINTS]
TELEVISION_ROT_90_POINTS = [rotate_tuple_90_clockwise(t) for t in TELEVISION_ROT_0_POINTS]
TELEVISION_ROT_135_POINTS = [rotate_tuple_90_clockwise(t) for t in TELEVISION_ROT_90_POINTS]
TELEVISION_ROT_180_POINTS = [rotate_tuple_90_clockwise(t) for t in TELEVISION_ROT_90_POINTS]
TELEVISION_ROT_225_POINTS = [rotate_tuple_90_clockwise(t) for t in TELEVISION_ROT_180_POINTS]
TELEVISION_ROT_270_POINTS = [rotate_tuple_90_clockwise(t) for t in TELEVISION_ROT_180_POINTS]
TELEVISION_ROT_315_POINTS = [rotate_tuple_90_clockwise(t) for t in TELEVISION_ROT_270_POINTS]
TELEVISION_ROT_360_POINTS = [rotate_tuple_90_clockwise(t) for t in TELEVISION_ROT_270_POINTS]

TELEVISION_ROTATION_TO_OCCUPATIONS = {
    0: TELEVISION_ROT_0_POINTS,
    45: TELEVISION_ROT_45_POINTS,
    90: TELEVISION_ROT_90_POINTS,
    135: TELEVISION_ROT_135_POINTS,
    180: TELEVISION_ROT_180_POINTS,
    225: TELEVISION_ROT_225_POINTS,
    270: TELEVISION_ROT_270_POINTS,
    315: TELEVISION_ROT_315_POINTS,
    360: TELEVISION_ROT_360_POINTS,
}

TRAIN_SCENE_NAMES = [
    SCENES_NAMES_SPLIT_BY_TYPE[j][i] for j in range(3) for i in range(20)
]

TV_STAND_ROT_0_POINTS = [(0.25 * i, 0.25 * j) for i in range(-2, 3) for j in range(-1, 2)]
TV_STAND_ROT_90_POINTS = [(0.25 * i, 0.25 * j) for i in range(-1, 2) for j in range(-2, 3)]
TV_STAND_ROT_180_POINTS = TV_STAND_ROT_0_POINTS
TV_STAND_ROT_270_POINTS = TV_STAND_ROT_90_POINTS

TV_STAND_ROTATION_TO_OCCUPATIONS = {
    0: TV_STAND_ROT_0_POINTS,
    90: TV_STAND_ROT_90_POINTS,
    180: TV_STAND_ROT_180_POINTS,
    270: TV_STAND_ROT_270_POINTS,
}

UNREACHABLE_SYM = 0

VALID_SCENE_NAMES = [
    SCENES_NAMES_SPLIT_BY_TYPE[j][i] for j in range(3) for i in range(20, 25)
]

VISIBILITY_DISTANCE = 1.25
VISITED_SYM = 5
