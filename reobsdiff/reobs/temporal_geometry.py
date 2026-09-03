"""Extension point for rigid now and scene-flow geometry in V2."""

from abc import ABC, abstractmethod


class TemporalGeometryProvider(ABC):
    @abstractmethod
    def pose(self, frame_id):
        pass


class RigidTemporalGeometryProvider(TemporalGeometryProvider):
    def __init__(self, camera_pose):
        self.camera_pose = camera_pose

    def pose(self, frame_id):
        return self.camera_pose


class SceneFlowTemporalGeometryProvider(TemporalGeometryProvider):
    def pose(self, frame_id):
        raise NotImplementedError("scene-flow temporal geometry is reserved for ReObsDiff V2")
