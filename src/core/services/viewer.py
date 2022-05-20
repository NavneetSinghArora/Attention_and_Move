import matplotlib.pyplot as plt
from os import makedirs
from os.path import join
from PIL import Image


class Viewer(object):

    _instance = None
    _plt = None
    _agent_count = None

    def __new__(cls, agentCount: int):
        if cls._instance is None:
            cls._instance = super(Viewer, cls).__new__(cls)
            cls._agent_count = agentCount
            cls._plt = plt.ion()
            cls._fig, cls._axs = plt.subplots(ncols=agentCount, nrows=2, figsize=(8, 8), facecolor='white')
            cls._img = []
        return cls._instance

    def __init__(self, agentCount: int) -> None:
        plt.subplots_adjust(wspace=0.15, hspace=0.15)
        for row in range(2):
            for col in range(agentCount):
                self._axs[row, col].axis('off')
                self._img.append(self._axs[row, col].imshow(Image.new(mode='RGB', size=[300, 300], color='white')))

    @classmethod
    def update(self, multiAgentEvent, moveNumber: int, saveFigure: bool, rootDirectory):
        
        for i,e in enumerate(multiAgentEvent.events):
            self._axs[0, i].set_title(f'AgentId: {i}', fontname='Andale Mono')
            self._img[i].set_data(e.frame)

        self._axs[1, 0].set_title(f'birds eye view', fontname='Andale Mono')
        self._img[self._agent_count].set_data(Image.fromarray(multiAgentEvent.events[0].third_party_camera_frames[0]))
        
        self._fig.canvas.draw()
        self._fig.canvas.flush_events()

        if saveFigure:
            try:
                makedirs("output/agent_movements/", exist_ok=True)
                plt.savefig(join(rootDirectory, 'output/agent_movements/' + 'Move_' + str(moveNumber) + '_.png'))
            except FileExistsError:
                pass
