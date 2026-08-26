"""Stable on-disk frame contract and frame sources."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import struct
import time

import cv2
import numpy as np


@dataclass(frozen=True)
class FramePacket:
    """One RGB-D observation with a fixed camera-to-world pose."""

    sequence_id: int
    timestamp_ns: int
    frame_name: str
    rgb_path: str
    inverse_depth_path: str
    inverse_depth_scale: float
    inverse_depth_offset: float
    width: int
    height: int
    intrinsics: dict
    T_world_camera: list
    world_frame: str = ''
    camera_frame: str = 'camera_color_optical_frame'
    schema_version: int = 1

    def validate(self):
        if self.schema_version != 1:
            raise ValueError('Unsupported FramePacket schema version')
        if self.sequence_id < 0 or self.timestamp_ns < 0:
            raise ValueError('Frame identifiers must be non-negative')
        if self.width <= 0 or self.height <= 0:
            raise ValueError('Frame dimensions must be positive')
        for name in ('fx', 'fy', 'cx', 'cy'):
            if name not in self.intrinsics:
                raise ValueError('Missing intrinsic %s' % name)
        if self.intrinsics['fx'] <= 0 or self.intrinsics['fy'] <= 0:
            raise ValueError('Focal lengths must be positive')
        transform = np.asarray(self.T_world_camera, dtype=np.float64)
        if transform.shape != (4, 4):
            raise ValueError('T_world_camera must be 4x4')
        if not np.isfinite(transform).all():
            raise ValueError('T_world_camera contains non-finite values')
        if not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-7):
            raise ValueError('T_world_camera has an invalid homogeneous row')
        return self

    @classmethod
    def from_json(cls, path):
        with Path(path).open(encoding='utf-8') as stream:
            value = cls(**json.load(stream))
        return value.validate()

    def write_atomic(self, path):
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + '.tmp')
        with temporary.open('w', encoding='utf-8') as stream:
            json.dump(asdict(self), stream, indent=2, sort_keys=True)
            stream.write('\n')
        temporary.replace(destination)

    def resolve(self, session_directory):
        root = Path(session_directory).expanduser().resolve()
        return root / self.rgb_path, root / self.inverse_depth_path

    def load_images(self, session_directory):
        rgb_path, depth_path = self.resolve(session_directory)
        bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        encoded = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
        if bgr is None or encoded is None:
            raise OSError('Could not load frame %d images' % self.sequence_id)
        if bgr.shape[:2] != (self.height, self.width):
            raise ValueError('RGB dimensions do not match frame manifest')
        if encoded.shape[:2] != (self.height, self.width):
            raise ValueError('Depth dimensions do not match frame manifest')
        if encoded.ndim != 2 or encoded.dtype != np.uint16:
            raise ValueError('Inverse depth must be a single-channel uint16 PNG')
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        valid = encoded > 0
        inverse_depth = encoded.astype(np.float32) / float(2**16) * float(self.inverse_depth_scale) + float(self.inverse_depth_offset)
        valid &= np.isfinite(inverse_depth) & (inverse_depth > 0)
        inverse_depth[~valid] = 0
        metric_depth = np.zeros_like(inverse_depth)
        metric_depth[valid] = 1.0 / inverse_depth[valid]
        return rgb, inverse_depth, metric_depth, valid


def _rotation_from_qvec(qvec):
    w, x, y, z = np.asarray(qvec, dtype=np.float64)
    return np.array(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
            [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
            [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
        ]
    )


def _read_colmap_camera(path):
    with Path(path).open('rb') as stream:
        if struct.unpack('<Q', stream.read(8))[0] != 1:
            raise ValueError('Replay supports one shared PINHOLE camera')
        camera_id, model_id, width, height = struct.unpack('<iiQQ', stream.read(24))
        if model_id == 1:
            fx, fy, cx, cy = struct.unpack('<dddd', stream.read(32))
        elif model_id == 0:
            focal, cx, cy = struct.unpack('<ddd', stream.read(24))
            fx = fy = focal
        else:
            raise ValueError('Only COLMAP PINHOLE cameras are supported')
    return camera_id, int(width), int(height), dict(fx=fx, fy=fy, cx=cx, cy=cy)


def _read_colmap_images(path):
    images = []
    with Path(path).open('rb') as stream:
        count = struct.unpack('<Q', stream.read(8))[0]
        for _ in range(count):
            values = struct.unpack('<idddddddi', stream.read(64))
            image_id = values[0]
            qvec = values[1:5]
            translation = np.asarray(values[5:8])
            camera_id = values[8]
            name_bytes = bytearray()
            while True:
                value = stream.read(1)
                if not value or value == b'\0':
                    break
                name_bytes.extend(value)
            point_count = struct.unpack('<Q', stream.read(8))[0]
            stream.seek(point_count * 24, 1)
            rotation = _rotation_from_qvec(qvec)
            transform = np.eye(4, dtype=np.float64)
            transform[:3, :3] = rotation.T
            transform[:3, 3] = -rotation.T @ translation
            images.append((image_id, camera_id, name_bytes.decode(), transform))
    return images


def read_colmap_point_cloud(path):
    """Read XYZ and RGB values from a COLMAP ``points3D.bin`` file."""
    path = Path(path)
    with path.open('rb') as stream:
        count_bytes = stream.read(8)
        if len(count_bytes) != 8:
            raise ValueError('COLMAP points3D.bin is missing its header')
        count = struct.unpack('<Q', count_bytes)[0]
        points = np.empty((count, 3), dtype=np.float32)
        colors = np.empty((count, 3), dtype=np.float32)
        record_format = '<QdddBBBd'
        record_size = struct.calcsize(record_format)
        for index in range(count):
            record = stream.read(record_size)
            if len(record) != record_size:
                raise ValueError('COLMAP points3D.bin ended inside a point')
            values = struct.unpack(record_format, record)
            points[index] = values[1:4]
            colors[index] = values[4:7]
            track_bytes = stream.read(8)
            if len(track_bytes) != 8:
                raise ValueError('COLMAP points3D.bin is missing a track')
            track_length = struct.unpack('<Q', track_bytes)[0]
            stream.seek(8 * track_length, 1)
    return points, colors / 255.0


class ColmapReplaySource:
    """Adapt an existing Graphdeco/COLMAP dataset to FramePackets."""

    def __init__(self, source):
        self.root = Path(source).expanduser().resolve()

    def __iter__(self):
        camera_id, width, height, intrinsics = _read_colmap_camera(self.root / 'sparse/0/cameras.bin')
        with (self.root / 'sparse/0/depth_params.json').open() as stream:
            depth_parameters = json.load(stream)
        images = _read_colmap_images(self.root / 'sparse/0/images.bin')

        def sort_key(record):
            stem = Path(record[2]).stem
            return (0, int(stem)) if stem.isdigit() else (1, stem)

        for sequence_id, (_, image_camera_id, name, transform) in enumerate(sorted(images, key=sort_key)):
            if image_camera_id != camera_id:
                raise ValueError('Image references an unexpected camera')
            stem = Path(name).stem
            parameters = depth_parameters[stem]
            yield FramePacket(
                sequence_id=sequence_id,
                timestamp_ns=sequence_id * 1_000_000_000,
                frame_name=name,
                rgb_path=str(Path('images') / name),
                inverse_depth_path=str(Path('depth') / (stem + '.png')),
                inverse_depth_scale=parameters['scale'],
                inverse_depth_offset=parameters['offset'],
                width=width,
                height=height,
                intrinsics=intrinsics,
                T_world_camera=transform.tolist(),
                world_frame='colmap_world',
            ).validate()


class LiveManifestSource:
    """Yield atomically published packets in sequence order."""

    def __init__(self, session, start_after=-1, poll_seconds=0.1):
        self.root = Path(session).expanduser().resolve()
        self.manifests = self.root / 'manifests'
        self.start_after = start_after
        self.poll_seconds = poll_seconds

    def pending(self):
        packets = []
        for path in self.manifests.glob('*.json'):
            packet = FramePacket.from_json(path)
            if packet.sequence_id > self.start_after:
                packets.append(packet)
        return sorted(packets, key=lambda packet: packet.sequence_id)

    def __iter__(self):
        while True:
            packets = self.pending()
            if not packets:
                time.sleep(self.poll_seconds)
                continue
            for packet in packets:
                if packet.sequence_id <= self.start_after:
                    continue
                self.start_after = packet.sequence_id
                yield packet
