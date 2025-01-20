import { CfnOutput, Stack, StackProps } from "aws-cdk-lib";
import { AccountPrincipal, Effect, PolicyStatement, Role } from "aws-cdk-lib/aws-iam";
import { Construct } from "constructs";

export class BrontesStack extends Stack {
    constructor(scope: Construct, id: string, props: StackProps) {
        super(scope, id, props);

        const env = props.env!;

        const dsqlClusterId = '4iabtwnq2j55iez4j4bkykghgm';
        const dsqlEndpoint = `${dsqlClusterId}.dsql.us-east-1.on.aws`;
        const dsqlArn = `arn:aws:dsql:${env.region}:${env.account}:cluster/${dsqlClusterId}`;

        const developerRole = new Role(this, 'DeveloperRole', {
            assumedBy: new AccountPrincipal(env.account),
        });
        developerRole.addToPolicy(new PolicyStatement({
            effect: Effect.ALLOW,
            actions: [
                'dsql:DbConnectAdmin',
            ],
            resources: [
                dsqlArn,
            ],
        }));

        new CfnOutput(this, 'DsqlEndpoint', { value: dsqlEndpoint });
        new CfnOutput(this, 'DeveloperRoleArn', { value: developerRole.roleArn});
    }
}