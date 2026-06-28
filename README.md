# Defense Lab

A terminal-only football video tracking project. Select the defensive jersey color, pass a local clip, and generate an annotated video, defender-candidate trajectories, and a conservative alignment report. No browser, server, graphical interface, cloud inference, API key, or network request during analysis.

## Status

This is a working classical computer-vision prototype, not a trained NFL player detector or a validated coverage classifier. It uses jersey-color segmentation and motion-gated association. A same-colored official, field marking, or sideline person can become a candidate. IDs can split or switch at crossings and occlusions. Review outputs against the original film.

The program draws observed image-space motion. With approximate calibration it can describe near-line counts and single-high/two-high shell candidates. It does **not** reliably infer Cover 1/2/3/4, man-versus-zone, defensive assignments, or the actual play call. Those capabilities need football-specific detection, calibrated field geometry, and labeled validation data.

## Setup

Python 3.9 or newer is required. Installation downloads OpenCV and its dependencies; processing afterward is local.

```sh
python -m venv .venv
```

Activate the environment:

- Windows Command Prompt: `.venv\Scripts\activate.bat`
- Windows PowerShell: `.venv\Scripts\Activate.ps1`
- macOS/Linux: `source .venv/bin/activate`

```sh
python -m pip install .
```

Use `python3` instead of `python` if that is your system's Python command.

## Track a clip

```sh
python defense_tracker.py play.mp4 --jersey white --start 2 --end 10
```

Or use the installed `defense-lab` command with the same arguments. Supported colors: white, black, blue, red, orange, yellow, purple. Choose the defense's jersey color. Both teams must be visually distinguishable.

The default window is 10 seconds; a selected range may be up to 120 seconds. Processing does not open windows or play video.

### Draw paths and estimate the alignment

For a **stationary, approximately overhead image with the line of scrimmage horizontal**, supply the line's image y-coordinate and approximate vertical pixels per yard:

```sh
python defense_tracker.py play.mp4 --jersey blue --start 0 --end 8 --fixed-camera --los 450 --pixels-per-yard 12 --defense-direction up --output reports/play-01
```

Coordinates refer to the original video resolution. `up` means the defense is above the specified line. The first stable tracking snapshot is used for the shell estimate, so start before the snap. Depth thresholds are rough heuristics: near-line candidates lie from -1 to +2 approximate yards from the line; deep candidates are at least 10 approximate yards behind it. These are not trained football rules and do not determine coverage.

For camera pans, zooms, perspective-heavy broadcast views, or cuts, omit `--fixed-camera`. Boxes and image coordinates are still produced, but field movement and alignment inference are withheld. A simple scene-change check resets IDs on large visual changes; it cannot detect every cut or camera movement.

### Reduce false detections

```sh
python defense_tracker.py play.mp4 --jersey white --roi 100 80 1000 550 --min-area 45 --max-area 1500 --max-distance 40
```

`--roi X Y W H` restricts detections to a field region. Area limits apply to jersey-color blobs, not whole bodies. Distance is a per-frame pixel gate; adjust it for resolution, frame rate and player speed. Uniform colors and thresholds are approximate and may fail under shadows or compression. Merged players are not separated reliably. A track must be detected for three observations before it is exported. Missing observations are left as gaps, not invented positions.

## Output

Each run creates a new folder under `reports/` unless `--output` is specified. Existing folders are never overwritten.

| File | Content |
|---|---|
| `tracked.mp4` | Silent video with candidate boxes, IDs, and optional trails |
| `tracks.csv` | Observed frame, timestamp, ID and image-center coordinates |
| `movement.png` | Image-space movement diagram for a declared fixed camera, or a withholding notice |
| `report.txt` | Readable alignment evidence and limitations |
| `report.json` | Structured report, method and processing metadata |

Input clips are never included in source control. Original audio is not copied. Tracking coordinates represent jersey-blob centers, not feet or calibrated field positions. Lost players can receive new IDs, so unique track count is not a player count.

## Tests

```sh
python -m unittest -v
```

Tests cover synthetic detections, ID continuity, missed detections, scene reset behavior, conservative alignment rules and a complete synthetic-video run. Synthetic tests establish pipeline behavior, not accuracy on real NFL footage. Real-footage tracking and scheme accuracy have not yet been benchmarked.

## Implementation

OpenCV HSV segmentation, connected components, and motion-gated nearest-neighbor association. No trained model weights are downloaded or used. All computation runs on the local CPU. See the [OpenCV documentation](https://docs.opencv.org/4.x/) for image processing and video I/O primitives.

Independent project; not affiliated with the NFL.
