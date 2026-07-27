#!/bin/sh

cd `dirname $0`
set -ex

: ${TRACT_RUN:=cargo run -p tract-cli $CARGO_OPTS --}

# --allow-random-input is seeded from a fixed constant, so `data` is the same
# every run. The graph gathers the same elements twice, once on the contiguous
# last-axis fast path and once forced onto the generic path, and outputs the
# summed absolute difference: any divergence between the two makes it non-zero.
#
# --assert-op-count keeps both gathers in the graph: if an optimizing pass ever
# rewrote one leg into the other the comparison would go vacuous instead of
# failing. Run unoptimized and optimized, since that could come from either.
$TRACT_RUN . run --allow-random-input \
    --assert-op-count GatherElements 2 \
    --assert-output 'mismatch:1,1,1,f32=0'

$TRACT_RUN . -O run --allow-random-input \
    --assert-op-count GatherElements 2 \
    --assert-output 'mismatch:1,1,1,f32=0'
