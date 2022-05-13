from os.path import join
import matplotlib.pyplot as plt


def visualize_frames(rgb_frames, figsize, move_number, root_directory):
    """Plots the rgb_frames for each agent."""
    fig, axs = plt.subplots(1, len(rgb_frames), figsize=figsize, facecolor='white', dpi=300)
    for i, frame in enumerate(rgb_frames):
        ax = axs[i]
        ax.imshow(frame)
        ax.set_title(f'AgentId: {i}', fontname='Andale Mono')
        ax.axis('off')
        if i == 1:
            plt.savefig(join(root_directory, 'output/agent_movements/' + 'Move_' + str(move_number) + '_.png'))
