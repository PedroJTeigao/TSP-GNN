#!/usr/bin/env bash
# Download and build the stand-alone LKH-3 binary in tsp/third_party/LKH-3 (phase 3 benchmark).
#
# The official site (webhotel4.ruc.dk/~keld/research/LKH-3) cannot be reached from the server, so the
# source is the GitHub mirror cerebis/LKH3, tag v3.0.6 (LKH-3.0.6, May 2019; unmodified distribution).
# The mirror lacks the empty SRC/OBJ directory of the tarball, which make needs.
# LKH is "distributed for research use, the author reserves all rights": tsp/third_party/ stays out of git.
#
# Run from tsp/:   bash scripts/build_lkh3.sh
set -euo pipefail
cd "$(dirname "$0")/.."
COMMIT=ba346db0548e9517e9d37923c19674e53bd8d2f4
SHA256=a6e3c025d30e200d132a1fd4fc4362068245c009733f9162834defc6f03e002f
mkdir -p third_party/download
cd third_party/download
[ -f cerebis-LKH3-v3.0.6.tar.gz ] || curl -sSfL -o cerebis-LKH3-v3.0.6.tar.gz \
    "https://github.com/cerebis/LKH3/archive/$COMMIT.tar.gz"
echo "$SHA256  cerebis-LKH3-v3.0.6.tar.gz" | sha256sum -c -
mkdir -p cerebis && tar -xzf cerebis-LKH3-v3.0.6.tar.gz -C cerebis
cd ..
[ -e LKH-3 ] && { echo "third_party/LKH-3 already exists: not overwritten"; exit 1; }
cp -r "download/cerebis/LKH3-$COMMIT" LKH-3
mkdir -p LKH-3/SRC/OBJ
nice -n 10 make -C LKH-3 -j4 > build_LKH-3.log 2>&1
# Check: the bundled pr2392 instance must reach its known optimum, 378032.
(cd LKH-3 && ./LKH pr2392.par < /dev/null | grep "Cost.min = 378032") && echo "LKH-3 built: third_party/LKH-3/LKH"
