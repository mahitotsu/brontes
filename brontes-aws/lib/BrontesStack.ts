import { CfnOutput, Duration, RemovalPolicy, Stack, StackProps } from "aws-cdk-lib";
import { HttpApi } from "aws-cdk-lib/aws-apigatewayv2";
import { HttpLambdaIntegration } from "aws-cdk-lib/aws-apigatewayv2-integrations";
import { AttributeType, BillingMode, StreamViewType, Table } from "aws-cdk-lib/aws-dynamodb";
import { Effect, PolicyStatement } from "aws-cdk-lib/aws-iam";
import { AssetCode, Function, FunctionUrlAuthType, InvokeMode, Runtime } from "aws-cdk-lib/aws-lambda";
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

        const httpRequestTable = new Table(this, 'HttpRequestTable', {
            partitionKey: { name: 'group', type: AttributeType.STRING },
            sortKey: { name: 'request-id', type: AttributeType.STRING },
            billingMode: BillingMode.PAY_PER_REQUEST,
            // stream: StreamViewType.NEW_IMAGE,
            removalPolicy: RemovalPolicy.DESTROY,
        });

        const apiBaseDir = `${__dirname}/../../brontes-api`;
        const httpRequestHandler = new Function(this, 'HttpRequestHandler', {
            runtime: Runtime.JAVA_21,
            code: AssetCode.fromCustomCommand(`${apiBaseDir}/target/lambda-asset.zip`, ['mvn', '-f', apiBaseDir, 'clean', 'package']),
            handler: "com.mahitotsu.brontes.api.HttpRequestHandler::handleRequest",
            memorySize: 2048,
            timeout: Duration.seconds(10),
            environment: {
                DSQL_ENDPOINTS: dsqlEndpoints.join(','),
                HTTP_REQUEST_TABLE_NAME: httpRequestTable.tableName,
            },
        });
        new LogGroup(httpRequestHandler, 'LogGroup', {
            logGroupName: `/aws/lambda/${httpRequestHandler.functionName}`,
            removalPolicy: RemovalPolicy.DESTROY,
            retention: RetentionDays.ONE_DAY,
        });
        httpRequestHandler.role?.addToPrincipalPolicy(new PolicyStatement({
            effect: Effect.ALLOW,
            actions: ['dsql:DbConnectAdmin'],
            resources: dsqlArns,
        }));
        httpRequestTable.grantReadWriteData(httpRequestHandler);

        const endpoint = httpRequestHandler.addFunctionUrl({
            authType: FunctionUrlAuthType.AWS_IAM,
            invokeMode: InvokeMode.BUFFERED,
        });

        const apiGww = new HttpApi(this, 'HttpApi', {
            defaultIntegration: new HttpLambdaIntegration('DefaultIntegration', httpRequestHandler),
        });

        new CfnOutput(this, 'Endpoint', { value: `${apiGww.url}` })
    }
}