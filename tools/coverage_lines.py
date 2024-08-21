#!/usr/bin/env python

"""
usage: coverage_lines.py info_file expected
"""
import os
import sys


def get_lines(info_file):
    hits = 0.0
    total = 0.0

    with open(info_file) as info_file:
        for line in info_file:
            line = line.strip()

            if not line.startswith("DA:"):
                continue

            line = line[3:]

            total += 1

            if int(line.split(",")[1]) > 0:
                hits += 1

    if total == 0:
        print("no data found")
        sys.exit()

    return hits / total


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit()

    info_file = sys.argv[1]
    expected = float(sys.argv[2])

    if not os.path.isfile(info_file):
        print(f"info file {info_file} is not exists, ignored")
        sys.exit()

    actual = get_lines(info_file)
    actual = round(actual, 3)

    if actual < expected:
        print(
            f"expected >= {round(expected * 100, 1)} %, actual {round(actual * 100, 1)} %, failed"
        )

        sys.exit(1)

    print(
        f"expected >= {round(expected * 100, 1)} %, actual {round(actual * 100, 1)} %, passed"
    )
