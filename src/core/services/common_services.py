
import matplotlib.pyplot as plt

from os import makedirs
from os.path import join

from src.core.utils.constants import PROJECT_ROOT_DIR


def visualize_frames(rgb_frames, figsize, move_number):
    """Plots the rgb_frames for each agent."""
    fig, axs = plt.subplots(1, len(rgb_frames), figsize=figsize, facecolor='white', dpi=300)
    for i, frame in enumerate(rgb_frames):
        ax = axs[i]
        ax.imshow(frame)
        ax.set_title(f'AgentId: {i}', fontname='Andale Mono')
        ax.axis('off')
        if i == 1:
            print('Capturing the RGB Frame for the agents')
            try:
                makedirs("output/agent_movements/", exist_ok=True)  # makes sure folder exists
                plt.savefig(join(PROJECT_ROOT_DIR, 'output/agent_movements/' + 'Movement_' + str(move_number) + '_.png'))
            except FileExistsError:
                pass


# def locate_object_in_frame(rgb_frames, frame_objects, target_object) -> bool:
#
#     return True

# def rotate_agent(_agentId):
#
#     event_0 = self.__controller.step('RotateLeft', agentId=_agentId)
