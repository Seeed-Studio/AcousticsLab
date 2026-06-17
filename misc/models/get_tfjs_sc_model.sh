#!/bin/bash

set -e

pushd "$(dirname "$0")"

# sc_preproc_model
curl -o sc_preproc_model.tar.gz \
    -fSsL "https://storage.googleapis.com/tfjs-models/tfjs/speech-commands/conversion/sc_preproc_model.tar.gz"
tar xzvf sc_preproc_model.tar.gz --directory sc_preproc_model --strip-components=1

# tfjs_sc_model
curl -o metadata.json \
    -fSsL "https://storage.googleapis.com/tfjs-models/tfjs/speech-commands/v0.2/browser_fft/18w/metadata.json"
curl -o model.json \
    -fSsL "https://storage.googleapis.com/tfjs-models/tfjs/speech-commands/v0.2/browser_fft/18w/model.json"
curl -o group1-shard1of2 \
    -fSsL "https://storage.googleapis.com/tfjs-models/tfjs/speech-commands/v0.2/browser_fft/18w/group1-shard1of2"
curl -o group1-shard2of2 \
    -fSsL "https://storage.googleapis.com/tfjs-models/tfjs/speech-commands/v0.2/browser_fft/18w/group1-shard2of2"

popd
exit 0
