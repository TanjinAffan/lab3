"""Lab 3 live monitor. Copy your notebook's define_zones(), then run with your actual port.

Example: python lidar_zones.py --port COM5. Use the Python environment from Lab 2.
Live captures and timestamps.csv are saved under results/live-YYYYMMDD-HHMMSS/.
Report your results and explanations in your engineering notebook in the course GitHub repo.

Coordinates are in metres in the LiDAR frame. Teaching display only; no robot control.
"""
import argparse
import csv
import math
from pathlib import Path
import threading
import time

HERE = Path(__file__).resolve().parent


def define_zones():
    inner = [(0.5, -0.4), (1.5, -0.4), (1.5, 0.4), (0.5, 0.4)]
    outer = [(0.2, -0.8), (2.2, -0.8), (2.2, 0.8), (0.2, 0.8)]
    return inner, outer


def circle(cx, cy, radius, vertices=120):
    """Optional design tool: polygon approximation of a circle."""
    return [(cx + radius * math.cos(2 * math.pi * i / vertices),
             cy + radius * math.sin(2 * math.pi * i / vertices)) for i in range(vertices)]


def inside(point, boundary):
    """Ray crossing for a simple polygon; edges count as inside."""
    x, y = point
    result = False
    for a, b in zip(boundary, boundary[1:] + boundary[:1]):
        ax, ay = a
        bx, by = b
        cross = (x-ax)*(by-ay) - (y-ay)*(bx-ax)
        if abs(cross) <= 1e-9 * max(1.0, math.hypot(bx-ax, by-ay)):
            if min(ax, bx)-1e-9 <= x <= max(ax, bx)+1e-9 and min(ay, by)-1e-9 <= y <= max(ay, by)+1e-9:
                return True
        if (ay > y) != (by > y) and x < ax + (y-ay)*(bx-ax)/(by-ay):
            result = not result
    return result


def validate_zones(inner, outer):
    for name, boundary in [('inner', inner), ('outer', outer)]:
        if len(boundary) < 3 or any(len(p) != 2 or not all(math.isfinite(v) for v in p) for p in boundary):
            raise ValueError(f'{name}: use at least three finite (x, y) vertices.')
        if any(a == b for a, b in zip(boundary, boundary[1:]+boundary[:1])):
            raise ValueError(f'{name}: do not repeat adjacent vertices or the first vertex.')
        area = sum(a[0]*b[1]-b[0]*a[1] for a, b in zip(boundary, boundary[1:]+boundary[:1]))
        if abs(area) < 1e-10:
            raise ValueError(f'{name}: boundary has zero area.')
    def proper_cross(a, b, c, d):
        def orient(p, q, r):
            return (q[0]-p[0])*(r[1]-p[1])-(q[1]-p[1])*(r[0]-p[0])
        return orient(a,b,c)*orient(a,b,d) < -1e-16 and orient(c,d,a)*orient(c,d,b) < -1e-16
    for boundary in (inner, outer):
        edges = list(zip(boundary, boundary[1:]+boundary[:1]))
        if any(proper_cross(a,b,c,d) for i,(a,b) in enumerate(edges) for c,d in edges[i+1:]):
            raise ValueError('Use simple boundaries without crossing edges.')
    for a, b in zip(inner, inner[1:]+inner[:1]):
        if not inside(a, outer) or not inside(((a[0]+b[0])/2, (a[1]+b[1])/2), outer):
            raise ValueError('Place the entire inner zone inside the outer zone.')
        if any(proper_cross(a,b,c,d) for c,d in zip(outer, outer[1:]+outer[:1])):
            raise ValueError('Inner boundary crosses outer boundary.')


def classify(points, inner, outer):
    if not points or any(not all(math.isfinite(v) for v in p) for p in points):
        return 'UNKNOWN'
    if any(inside(p, inner) for p in points):
        return 'RED'
    if any(inside(p, outer) for p in points):
        return 'YELLOW'
    return 'GREEN'


COLORS = {'RED': '#f5b5b5', 'YELLOW': '#fff0a3', 'GREEN': '#c7e8c5', 'UNKNOWN': '#d6d6d6'}


def draw(ax, inner, outer, points=(), title='Zone design', state=None, limits=None):
    ax.clear()
    ax.set_facecolor(COLORS.get(state, 'white'))
    for boundary, color, label in [(outer, '#a87900', 'Outer boundary'), (inner, '#b22222', 'Inner boundary')]:
        closed = boundary + boundary[:1]
        ax.plot(*zip(*closed), color=color, linewidth=2, label=label)
    if points:
        ax.scatter(*zip(*points), s=12, c='#005b8e', label='Points')
    ax.plot(0, 0, 'k+', markersize=12, label='LiDAR')
    ax.set(xlabel='x (m): angle 0 degrees', ylabel='y (m): angle 90 degrees', title=title, aspect='equal')
    ax.title.set_fontsize(10)
    ax.margins(.15)
    if limits is not None:
        ax.set(xlim=limits, ylim=limits, autoscale_on=False)
    ax.grid(alpha=.3)
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=8)


def run(port, inner, outer, output=HERE/'results', max_age=.5):
    """Read independently of plotting; log receipt timestamps for complete scans."""
    import matplotlib.pyplot as plt
    from rplidar import RPLidar
    folder = Path(output) / time.strftime('live-%Y%m%d-%H%M%S')
    folder.mkdir(parents=True, exist_ok=False)
    stop = threading.Event()
    lock = threading.Lock()
    shared = {'frame': None, 'error': None}

    def acquire():
        lidar = None
        try:
            with (folder/'timestamps.csv').open('w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(['scan_id', 'received_s', 'scan_interval_ms'])
                lidar = RPLidar(port, baudrate=115200, timeout=.5)
                if lidar.get_health()[0] != 'Good':
                    raise RuntimeError('LiDAR health is not Good.')
                raw, started, previous, scan_id = [], False, None, 0
                for marker, quality, angle, distance in lidar.iter_measures(max_buf_meas=False):
                    if stop.is_set():
                        break
                    if lidar._serial.in_waiting > 3000:
                        raise RuntimeError('Serial backlog: stop and repeat with less host load.')
                    if marker:
                        if started:
                            received = time.perf_counter()
                            interval = '' if previous is None else (received-previous)*1000
                            writer.writerow([scan_id, f'{received:.9f}', interval])
                            file.flush()
                            with lock:
                                shared['frame'] = (scan_id, received, tuple(raw), interval)
                            previous, scan_id = received, scan_id+1
                        raw, started = [], True
                    if started:
                        raw.append((quality, angle, distance))
                if not stop.is_set():
                    raise RuntimeError('LiDAR stream ended.')
        except Exception as error:
            with lock:
                shared['error'] = f'{type(error).__name__}: {error}'
        finally:
            if lidar is not None:
                for action in (lidar.stop, lidar.stop_motor, lidar.disconnect):
                    try:
                        action()
                    except Exception as error:
                        with lock:
                            shared['error'] = shared['error'] or f'Cleanup failed: {error}'

    fig, ax = plt.subplots(figsize=(8, 7))
    fig.subplots_adjust(left=.12, right=.75, bottom=.12, top=.84)
    worker = threading.Thread(target=acquire, daemon=True)
    captured = set()
    worker.start()
    print(f'LIVE on {port}. Close the plot or press Ctrl+C to stop. Output: {folder}')
    try:
        while plt.fignum_exists(fig.number):
            with lock:
                frame, error = shared['frame'], shared['error']
            points, state, caption = [], 'UNKNOWN', 'Waiting for a complete scan'
            if frame:
                scan_id, received, raw, interval = frame
                points = [(d/1000*math.cos(math.radians(a)), d/1000*math.sin(math.radians(a)))
                          for q,a,d in raw if all(math.isfinite(v) for v in (q,a,d)) and q > 0 and d > 0 and 0 <= a < 360]
                age = time.perf_counter()-received
                state = classify(points, inner, outer) if age <= max_age and not error else 'UNKNOWN'
                caption = f'scan {scan_id} | receipt {received:.6f} s\nage {age*1000:.0f} ms'
                if interval != '':
                    caption += f' | interval {interval:.1f} ms'
            draw(ax, inner, outer, points, f'LIVE | {state}\n{caption}', state, limits=(-4, 4))
            if frame and state not in captured:
                fig.savefig(folder/f'T3-2-{state.lower()}.png', dpi=140)
                captured.add(state)
            plt.pause(.1)
            if error:
                raise RuntimeError(error)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        worker.join(4)
        plt.close(fig)
        if worker.is_alive():
            raise RuntimeError('Reader did not stop. Close this Python process before reconnecting.')
        if shared['error']:
            print(shared['error'])


TEST_POINTS = [
    ('P1', 1.0, 0.0, 'RED'),
    ('P2', 0.5, 0.4, 'RED'),
    ('P3', 1.8, 0.6, 'YELLOW'),
    ('P4', 0.3, -0.7, 'YELLOW'),
    ('P5', 3.0, 0.0, 'GREEN'),
]


def check(inner, outer):
    import matplotlib.pyplot as plt
    rows = []
    for name, x, y, expected in TEST_POINTS:
        got = classify([(x, y)], inner, outer)
        rows.append([name, x, y, expected, got, 'PASS' if got == expected else 'FAIL'])
        print(f'{name} ({x:.2f}, {y:.2f}) expected {expected:6} got {got:6} {rows[-1][-1]}')
    with open(HERE/'data'/'T3-1-point-check.csv', 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['point', 'x_m', 'y_m', 'expected', 'result', 'check'])
        writer.writerows(rows)
    fig, ax = plt.subplots(figsize=(8, 7))
    fig.subplots_adjust(right=.75)
    draw(ax, inner, outer, [(x, y) for _, x, y, _ in TEST_POINTS], 'T3-1 Zone design and test points')
    for name, x, y, _ in TEST_POINTS:
        ax.annotate(name, (x, y), textcoords='offset points', xytext=(5, 5), fontsize=8)
    fig.savefig(HERE/'screenshots'/'T3-1-1.png', dpi=140)
    print('Saved screenshots/T3-1-1.png and data/T3-1-point-check.csv')
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog='Close other LiDAR programs before running. Use the saved evidence in your engineering notebook.')
    parser.add_argument('--port', help='Actual serial port, e.g. COM5')
    parser.add_argument('--check', action='store_true', help='Plot zones and test five points without LiDAR')
    args = parser.parse_args()
    inner, outer = define_zones()
    validate_zones(inner, outer)
    print('Zones valid.')
    if args.check:
        check(inner, outer)
    elif args.port:
        run(args.port, inner, outer)
    else:
        parser.error('use --check or --port PORT')


if __name__ == '__main__':
    main()
