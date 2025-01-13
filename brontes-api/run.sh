#!/bin/bash
ROLE_ARN="arn:aws:iam::346929044083:role/Developer"
TIMESTAMP=$(date +"%Y%m%dT%H%M%S")
ROLE_SESSION_NAME="local-execution-$TIMESTAMP"

ASSUME_ROLE_OUTPUT=$(aws sts assume-role --role-arn "$ROLE_ARN" --role-session-name "$ROLE_SESSION_NAME")

AWS_ACCESS_KEY_ID=$(echo "$ASSUME_ROLE_OUTPUT" | jq -r '.Credentials.AccessKeyId')
AWS_SECRET_ACCESS_KEY=$(echo "$ASSUME_ROLE_OUTPUT" | jq -r '.Credentials.SecretAccessKey')
AWS_SESSION_TOKEN=$(echo "$ASSUME_ROLE_OUTPUT" | jq -r '.Credentials.SessionToken')

export AWS_ACCESS_KEY_ID
export AWS_SECRET_ACCESS_KEY
export AWS_SESSION_TOKEN

DSQL_ENDPOINTS="4iabtwnq2j55iez4j4bkykghgm.dsql.us-east-1.on.aws,s4abtwnq2jebk7aj6vhlsb2coi.dsql.us-east-2.on.aws"
AWS_REGION="us-east-1"

export DSQL_ENDPOINTS
export AWS_REGION

mvn spring-boot:run