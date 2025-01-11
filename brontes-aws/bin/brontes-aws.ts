#!/usr/bin/env node

import { App } from "aws-cdk-lib";
import { BrontesStack } from "../lib/BrontesStack";

const app = new App();
const account = process.env.CDK_DEFAULT_ACCOUNT;

new BrontesStack(app, 'BrontesStack', {
  env: { account, region: 'us-east-1' },
});