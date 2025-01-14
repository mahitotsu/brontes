import { CfnOutput, Duration, RemovalPolicy, Stack, StackProps } from "aws-cdk-lib";
import { Effect, PolicyStatement } from "aws-cdk-lib/aws-iam";
import { DockerImageCode, DockerImageFunction, FunctionUrlAuthType, InvokeMode } from "aws-cdk-lib/aws-lambda";
import { LogGroup, RetentionDays } from "aws-cdk-lib/aws-logs";
import { Construct } from "constructs";

export class BrontesStack extends Stack {

    constructor(scope: Construct, id: string, props: StackProps) {
        super(scope, id, props);

        const env = props.env!;

        const dsqlClusterIds = {
            'us-east-1': '4iabtwnq2j55iez4j4bkykghgm',
            'us-east-2': 's4abtwnq2jebk7aj6vhlsb2coi'
        };
        const dsqlEndpoints = Object.entries(dsqlClusterIds).map(entry => `${entry[1]}.dsql.${entry[0]}.on.aws`);
        const dsqlArns = Object.entries(dsqlClusterIds).map(entry => `arn:aws:dsql:${entry[0]}:${env.account}:cluster/${entry[1]}`);

        const server = new DockerImageFunction(this, 'Server', {
            code: DockerImageCode.fromImageAsset(`${__dirname}/../../brontes-api`, {}),
            memorySize: 2048,
            timeout: Duration.seconds(10),
            environment: {
                DSQL_ENDPOINTS: dsqlEndpoints.join(','),
            },
        });
        server.role?.addToPrincipalPolicy(new PolicyStatement({
            effect: Effect.ALLOW,
            actions: ['dsql:DbConnectAdmin'],
            resources: dsqlArns,
        }));
        new LogGroup(this, 'LogGroup', {
            logGroupName: `/aws/lambda/${server.functionName}`,
            removalPolicy: RemovalPolicy.DESTROY,
            retention: RetentionDays.ONE_DAY,
        });

        const endpoint = server.addFunctionUrl({
            authType: FunctionUrlAuthType.NONE,
            invokeMode: InvokeMode.BUFFERED,
        });

        new CfnOutput(this, 'Endpoint', {value: `${endpoint.url}`})
    }
}