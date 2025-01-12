#!/bin/bash
SCRIPT_DIR=$(cd $(dirname $0); pwd)

mvn -f $SCRIPT_DIR/../brontes-api spring-boot:build-image &&\
cdk deploy