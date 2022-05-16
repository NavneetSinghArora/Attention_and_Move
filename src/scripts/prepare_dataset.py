"""
This file is used to collect and prepare the simulator agent_view dataset for model tuning and evaluation.
The dataset collected is divided into training, validation and testing set.
"""

# Importing python libraries for processing
import numpy as np
from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering
import matplotlib.pyplot as plt
import csv


class PrepareDataset:

    def __init__(self):
        # Initializing the controller
        self.controller = Controller(platform=CloudRendering,
                                     agentCount=2, agentMode="default", visibilityDistance=1.5, scene="FloorPlan212",
                                     gridSize=0.25, snapToGrid=True, rotateStepDegrees=30,
                                     renderDepthImage=False, renderInstanceSegmentation=False,
                                     width=300, height=300, fieldOfView=90)

        # Creating the list of all the scenes available to retrieve the data
        self.scenes = self.controller.ithor_scenes()

        # Collection of Moves available
        self.move_actions = ["Done", "MoveAhead", "MoveBack", "MoveLeft", "MoveRight"]
        self.rotate_action = "RotateRight"
        self.pose_action = ["Crouch", "Stand"]
        self.camera_actions = ["LookUp", "LookDown"]

        # Collection of Frames
        self.frames = None

    def reset_controller(self, scene):
        self.controller.reset(scene=scene)

    def get_reachable_positions(self):
        positions = self.controller.step(action="GetReachablePositions").metadata["actionReturn"]
        return positions

    def teleport_agent(self, position, agent):
        self.controller.step(action="Teleport", position=position, agentId=agent)

    def take_action(self, action):
        self.controller.step(action=action)

    def collect_frames(self):
        rgb_frames = [event.frame for event in self.controller.last_event.events]
        np.append(self.frames, rgb_frames)

    def create_files(self, rgb_frame, image_count, frame_object):
        plt.figure(figsize=(8, 8), dpi=100)
        plt.imshow(rgb_frame)
        plt.axis('off')
        image_name = 'image_' + str(image_count) + '.png'
        plt.savefig('/export2/scratch/cv_proj_team1/Attention_and_Move/output/dataset/images/' + image_name,
                    bbox_inches='tight', pad_inches=0)

        images_objects = frame_object
        for item in images_objects:
            if item['visible']:
                item_name = (item['objectId'].split('|'))[0]
                item_size = item['axisAlignedBoundingBox']['size']
                item_center = item['axisAlignedBoundingBox']['center']
                item_corners = item['axisAlignedBoundingBox']['cornerPoints']

                data = [item_name, item_size, item_center, item_corners]
                annotation_file_name = 'image_' + str(image_count) + '.csv'
                with open('/export2/scratch/cv_proj_team1/Attention_and_Move/output/dataset/annotations/' + annotation_file_name, 'a', encoding='UTF8') as file:
                    writer = csv.writer(file)
                    writer.writerow(data)

        plt.close()

    def collect_dataset(self):
        image_count = 0
        for scene in self.scenes:
            print(f"Processing Image: {image_count}")
            self.reset_controller(scene)
            all_possible_positions = self.get_reachable_positions()
            fixed_agent_processed = False
            for position in all_possible_positions:
                self.teleport_agent(position, 0)
                for action in self.move_actions:
                    step_result = self.controller.step(action, agentId=0)
                    if step_result.metadata['errorMessage'] is None or step_result.metadata['errorMessage'] == '':
                        for i in range(0, 12):
                            if not fixed_agent_processed:
                                agent_event = self.controller.step(self.rotate_action, agentId=1)
                                rgb_frames = [event.frame for event in agent_event.events]
                                frame_objects = [event.metadata['objects'] for event in agent_event.events]
                                self.create_files(rgb_frames[1], image_count, frame_objects[1])
                                image_count = image_count + 1

                        fixed_agent_processed = True

                        for i in range(0, 12):
                            agent_event = self.controller.step(self.rotate_action, agentId=0)
                            rgb_frames = [event.frame for event in agent_event.events]
                            frame_objects = [event.metadata['objects'] for event in agent_event.events]

                            self.create_files(rgb_frames[0], image_count, frame_objects[0])
                            image_count = image_count + 1


if __name__ == '__main__':
    prepare_dataset = PrepareDataset()
    prepare_dataset.collect_dataset()
