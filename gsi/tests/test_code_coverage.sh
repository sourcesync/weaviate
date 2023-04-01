#!/bin/bash

set -e

# Check go is available and at right version
which go

CHECKVER="1.20"
CURVER=$(go version)
if [[ "$CURVER" == *"$CHECKVER"* ]]; then
    echo "$0: GO version check passed at $CURVER"
else
    echo "$0: Expecting GO version $CHECKVER but got $CURVER"
    exit 1
fi


# Make sure the test program can access the module
go get github.com/gsi/weaviate/gemini_plugin

# Build the test program with code coverage
go build -coverpkg=main,github.com/gsi/weaviate/gemini_plugin -o /tmp/test 

# Remove previous test artifacts directory if present
if [ -d /tmp/testdata ]; then
    echo "$0: Clearing last test artifacts directory..."
    rm -fr /tmp/testdata
fi

# make a test artifacts directory
echo "$0: Making test artifacts directory..."
mkdir /tmp/testdata

# Run the program and product test artifacts
echo "$0: Running the test program with code coverage..."
echo
GOCOVERDIR=/tmp/testdata /tmp/test
echo

# Output the code coverage
if [ -d /tmp/testdata ]; then
    echo "$0: Test coverage stats:"
    go tool covdata percent -i=/tmp/testdata
else
    echo "$0: No test coverage stats to show.  Did the test program fail?"
fi
