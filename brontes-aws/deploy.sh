#!/bin/bash
SCRIPT_DIR=$(cd $(dirname $0); pwd)

mvn -f $SCRIPT_DIR/../brontes-api package &&\
cdk deploy --method direct