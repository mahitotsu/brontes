#!/usr/bin/env node
import { App } from "aws-cdk-lib";
import { BrontesStack } from "../lib/BrontesStack";

const app = new App();
new BrontesStack(app, 'BrontesStack', {
  env: { account: process.env.CDK_DEFAULT_ACCOUNT, region: 'us-east-1' },
});