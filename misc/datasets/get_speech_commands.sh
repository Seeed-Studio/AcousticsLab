#!/bin/bash

set -e

pushd "$(dirname "$0")"

# check if directory or symbolic link "speech_commands*" exists
if [ -d "speech_commands*" ] || [ -L "speech_commands*" ]; then
    echo "Directory or symbolic link 'speech_commands*' already exists. Skipping download."
    exit 1
fi

# download and extract the dataset
curl -o speech_commands_v0.02.tar.gz \
    -fSsL "http://download.tensorflow.org/data/speech_commands_v0.02.tar.gz"
tar xzvf speech_commands_v0.02.tar.gz --directory speech_commands_v0.02 --strip-components=1

popd
exit 0
