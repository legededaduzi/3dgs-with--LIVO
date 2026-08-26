"""Application orchestration for live, resumed, and replay mapping runs."""

import argparse
import json
from pathlib import Path
import time

from .config import MapperConfig
from .frame_packet import ColmapReplaySource, LiveManifestSource, read_colmap_point_cloud
from .geometry import point_cloud_scales


def _frame_timing(result):
    """Return the compact, stable console representation of one frame."""
    return {
        'frame': result['last_processed'],
        'keyframe': result['keyframe_added'],
        'queue': result['queue_length'],
        'timing_ms': {name: round(value, 2) for name, value in result['timing_ms'].items()},
    }


def _print_frame_timing(result):
    print(json.dumps(_frame_timing(result), sort_keys=True), flush=True)


def _boolean(value):
    if isinstance(value, bool):
        return value
    normalized = value.lower()
    if normalized in {'true', '1', 'yes', 'on'}:
        return True
    if normalized in {'false', '0', 'no', 'off'}:
        return False
    raise argparse.ArgumentTypeError('expected true or false')


def _create_mapper(args, config, checkpoint=None, save_checkpoints=True, unique_final_ply=False, initial_point_cloud=None, refinement_packets=None):
    """Create the numerical mapper without modifying configuration values."""
    from .online_mapper import OnlineMapper

    output = args.output or str(Path(args.session) / 'online_output')
    if checkpoint is not None:
        checkpoint = Path(checkpoint).expanduser()
        if not checkpoint.is_file():
            checkpoint = None
    return OnlineMapper(
        args.session,
        output,
        config,
        checkpoint,
        save_checkpoints=save_checkpoints,
        unique_final_ply=unique_final_ply,
        refine_on_close=config.final_refinement_iterations > 0,
        initial_point_cloud=initial_point_cloud,
        refinement_packets=refinement_packets,
        preview=getattr(args, 'preview', False),
        preview_depth_min=getattr(args, 'preview_depth_min', 0.2),
        preview_depth_max=getattr(args, 'preview_depth_max', 5.0),
    )


def replay(args):
    """Replay a manifest archive or a Graphdeco/COLMAP dataset."""
    args.session = args.source
    config = MapperConfig.load(args.config)
    source_root = Path(args.source).expanduser()
    sparse_root = source_root / 'sparse/0'
    colmap_metadata = (sparse_root / 'cameras.bin', sparse_root / 'images.bin', sparse_root / 'depth_params.json')
    if all(path.is_file() for path in colmap_metadata):
        packets = list(ColmapReplaySource(args.source))
        frame_source = 'colmap'
    else:
        packets = LiveManifestSource(args.source).pending()
        frame_source = 'manifests'
    if not packets:
        raise RuntimeError('Replay source contains neither complete COLMAP metadata nor ' 'frame manifests')
    if args.max_frames is not None:
        packets = packets[: args.max_frames]
    print(json.dumps({'phase': 'frame_source', 'source': frame_source, 'frames': len(packets)}, sort_keys=True), flush=True)

    initial_point_cloud = None
    checkpoint = Path(args.checkpoint).expanduser() if args.checkpoint else None
    has_checkpoint = checkpoint is not None and checkpoint.is_file()
    if not args.no_point_cloud_bootstrap and not has_checkpoint:
        point_path = sparse_root / 'points3D.bin'
        if point_path.is_file():
            points, colors = read_colmap_point_cloud(point_path)
            if len(points):
                scales = point_cloud_scales(points, fallback=config.voxel_size).clip(config.minimum_gaussian_scale, config.maximum_gaussian_scale)
                initial_point_cloud = points, colors, scales
                print(json.dumps({'phase': 'point_cloud_bootstrap', 'points': len(points), 'source': str(point_path)}, sort_keys=True), flush=True)

    mapper = _create_mapper(args, config, args.checkpoint, initial_point_cloud=initial_point_cloud, refinement_packets=packets)
    try:
        for packet in packets:
            if mapper.stopped:
                break
            result = mapper.process(packet)
            if result is not None:
                _print_frame_timing(result)
    finally:
        mapper.close()


def live(args):
    """Consume atomically published manifests until the process is stopped."""
    config = MapperConfig.load(args.config)
    resume_mode = args.command == 'resume'
    refinement_packets = []
    mapper = _create_mapper(args, config, args.checkpoint, save_checkpoints=resume_mode, unique_final_ply=True, refinement_packets=refinement_packets)
    source = LiveManifestSource(args.session, mapper.last_processed)
    if mapper.last_processed >= 0:
        refinement_packets.extend(packet for packet in LiveManifestSource(args.session).pending() if packet.sequence_id <= mapper.last_processed)
    try:
        while not mapper.stopped:
            packets = source.pending()
            if not packets:
                time.sleep(source.poll_seconds)
                continue
            for index, packet in enumerate(packets):
                if mapper.stopped:
                    break
                optimize = len(mapper.model.xyz) == 0 or len(packets) <= 2 or index == len(packets) - 1
                result = mapper.process(packet, optimize=optimize, queue_length=len(packets) - index - 1)
                refinement_packets.append(packet)
                source.start_after = packet.sequence_id
                if result is not None:
                    _print_frame_timing(result)
    finally:
        mapper.close()


def build_parser():
    parser = argparse.ArgumentParser(
        prog='gs-slam-backend', description=('Algorithm switches and tuning parameters are read exclusively ' 'from --config.')
    )
    commands = parser.add_subparsers(dest='command', required=True)

    def common(command, checkpoint_required=False):
        command.add_argument('--config', required=True)
        command.add_argument('--output')
        command.add_argument('--checkpoint', required=checkpoint_required)

    def live_preview(command):
        command.add_argument(
            '--preview', nargs='?', const=True, default=False, type=_boolean, help='show RGB, metric depth, and the reused training render'
        )
        command.add_argument('--preview-depth-min', type=float, default=0.2)
        command.add_argument('--preview-depth-max', type=float, default=5.0)

    replay_parser = commands.add_parser('replay')
    replay_parser.add_argument('--source', required=True)
    replay_parser.add_argument('--max-frames', type=int)
    replay_parser.add_argument('--no-point-cloud-bootstrap', action='store_true')
    common(replay_parser)
    replay_parser.set_defaults(handler=replay)

    live_parser = commands.add_parser('live')
    live_parser.add_argument('--session', required=True)
    common(live_parser)
    live_preview(live_parser)
    live_parser.set_defaults(handler=live)

    resume_parser = commands.add_parser('resume')
    resume_parser.add_argument('--session', required=True)
    common(resume_parser, checkpoint_required=True)
    live_preview(resume_parser)
    resume_parser.set_defaults(handler=live)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.handler(args)


if __name__ == '__main__':
    main()
