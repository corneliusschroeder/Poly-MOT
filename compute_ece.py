#!/usr/bin/env python3
import re
import sys
from collections import OrderedDict

def parse_ci_file(path):
    """
    Parse a text file with blocks of the form:

        Confidence intervalls for class CLASSNAME
        CI 0.1:   0.8004
        CI yaw_0.1:   0.6437
        CI 0.2:   0.8694
        CI yaw_0.2:   0.7123
        ...

    Returns:
        OrderedDict: {
            class_name: {
                'position': [(ci, value), ...],
                'yaw': [(ci, value), ...]
            },
            ...
        }
    """
    classes = OrderedDict()
    current_class = None

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Match "Confidence intervalls for class <name>"
            m_class = re.match(r"Confidence intervalls for class (.+)", line)
            if m_class:
                current_class = m_class.group(1).strip()
                classes[current_class] = {'position': [], 'yaw': []}
                continue

            # Match "CI 0.100000:   0.8306" or "CI yaw_0.100000:   0.6437"
            m_ci = re.match(r"CI\s+(yaw_)?([0-9.]+):\s+([0-9.]+)", line)
            if m_ci and current_class is not None:
                is_yaw = m_ci.group(1) is not None
                ci = float(m_ci.group(2))
                val = float(m_ci.group(3))
                
                if is_yaw:
                    classes[current_class]['yaw'].append((ci, val))
                else:
                    classes[current_class]['position'].append((ci, val))

  # Ensure each class' curves are sorted by confidence and replace last point with (1.0, 1.0)
    for k in classes:
        classes[k]['position'] = sorted(classes[k]['position'], key=lambda x: x[0])
        classes[k]['yaw'] = sorted(classes[k]['yaw'], key=lambda x: x[0])
        
        # Replace last point with (1.0, 1.0) for position
        if classes[k]['position']:
            classes[k]['position'][-1] = (1.0, 1.0)
        
        # Replace last point with (1.0, 1.0) for yaw
        if classes[k]['yaw']:
            classes[k]['yaw'][-1] = (1.0, 1.0)
            
    return classes


def ece_integral(curve):
    """
    Approximate Expected Calibration Error for a single class.

    curve: list of (confidence, accuracy) pairs, sorted by confidence.

    We approximate:
        ECE ≈ ∫_0^1 |acc(c) - c| dc

    by a piecewise-constant Riemann sum using the given points.
    """
    if not curve:
        return 0.0

    curve = sorted(curve, key=lambda x: x[0])
    ece = 0.0
    prev_ci = 0.0
    prev_acc = curve[0][1]

    for ci, acc in curve:
        width = max(ci - prev_ci, 0.0)
        # use accuracy at the right endpoint for this interval
        ece += width * abs(acc - ci)
        prev_ci = ci
        prev_acc = acc

    # If last CI < 1.0, extend with constant accuracy up to 1.0
    if prev_ci < 1.0:
        width = 1.0 - prev_ci
        ece += width * abs(prev_acc - 1.0)

    return ece


def main():
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        # fallback default file name
        path = "ci.txt"

    classes = parse_ci_file(path)

    print(f"Reading calibration curves from: {path}\n")
    
    macro_ece_position = 0.0
    macro_ece_yaw = 0.0
    
    for name, curves in classes.items():
        ece_pos = ece_integral(curves['position'])
        ece_yaw = ece_integral(curves['yaw'])
        macro_ece_position += ece_pos
        macro_ece_yaw += ece_yaw
        avg_ece = (ece_pos + ece_yaw)/2 
        print(f"Class {name:>20s} | Position ECE ≈ {ece_pos:.6f} | Yaw ECE ≈ {ece_yaw:.6f} | Avg ECE ≈ {avg_ece:.6f}")

    if classes:
        macro_ece_position /= len(classes)
        macro_ece_yaw /= len(classes)
        print(f"\nMacro-average Position ECE over {len(classes)} classes: {macro_ece_position:.6f}")
        print(f"Macro-average Yaw ECE over {len(classes)} classes: {macro_ece_yaw:.6f}")
        print(f"Overall macro-average ECE: {(macro_ece_position + macro_ece_yaw) / 2:.6f}")


if __name__ == "__main__":
    main()