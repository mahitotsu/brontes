import { CfnOutput, Duration, RemovalPolicy, Stack, StackProps } from "aws-cdk-lib";
import { IpProtocol, SubnetType, Vpc } from "aws-cdk-lib/aws-ec2";
import { Cluster, ContainerImage, FargateService, FargateTaskDefinition, LogDriver } from "aws-cdk-lib/aws-ecs";
import { ApplicationLoadBalancer, ApplicationProtocol, ApplicationTargetGroup, TargetType } from "aws-cdk-lib/aws-elasticloadbalancingv2";
import { LogGroup, RetentionDays } from "aws-cdk-lib/aws-logs";
import { Construct } from "constructs";

export class BrontesStack extends Stack {

    constructor(scope: Construct, id: string, props: StackProps) {
        super(scope, id, props);

        const maxAzs = 2;
        const listenerPort = 80;
        const containerPort = 8080;

        const taskDefinition = new FargateTaskDefinition(this, 'TaskDefinition', { cpu: 512, memoryLimitMiB: 4096 });
        taskDefinition.addContainer('application', {
            image: ContainerImage.fromAsset(`${__dirname}/../../brontes-api`),
            portMappings: [{ containerPort }],
            logging: LogDriver.awsLogs({
                streamPrefix: 'application',
                logGroup: new LogGroup(this, 'ApplicationLogGroup', {
                    retention: RetentionDays.ONE_DAY,
                    removalPolicy: RemovalPolicy.DESTROY,
                })
            })
        });

        const vpc = new Vpc(this, 'Vpc', {
            ipProtocol: IpProtocol.DUAL_STACK,
            createInternetGateway: true,
            maxAzs,
        });

        const lb = new ApplicationLoadBalancer(vpc, 'LoadBalancer', {
            vpc, vpcSubnets: { subnetType: SubnetType.PUBLIC }, internetFacing: true,
        });
        const lister = lb.addListener('HttpListener', { protocol: ApplicationProtocol.HTTP, port: listenerPort });

        const cluster = new Cluster(vpc, 'ApplicationCluster', {
            vpc, enableFargateCapacityProviders: true,
        });
        cluster.addDefaultCapacityProviderStrategy([{ capacityProvider: 'FARGATE_SPOT', weight: 2 }]);

        const service = new FargateService(cluster, 'ApplicationService', {
            taskDefinition, cluster, vpcSubnets: { subnetType: SubnetType.PRIVATE_WITH_EGRESS }, assignPublicIp: false,
            capacityProviderStrategies: [{ capacityProvider: 'FARGATE_SPOT', weight: 1 }],
            minHealthyPercent: 0, maxHealthyPercent: 200, circuitBreaker: { enable: true, rollback: true },
            desiredCount: 1,
        });

        const targetGroup = new ApplicationTargetGroup(this, 'TargetGroup', {
            vpc, targetType: TargetType.IP, port: containerPort, protocol: ApplicationProtocol.HTTP,
            targets: [service],
            healthCheck: {
                path: '/actuator/health',
                interval: Duration.seconds(5), timeout: Duration.seconds(2),
                healthyThresholdCount: 2, unhealthyThresholdCount: 2,
                healthyHttpCodes: '200',
            },
            deregistrationDelay: Duration.seconds(0),
        });
        lister.addTargetGroups('TargetGroups', { targetGroups: [targetGroup] });

        new CfnOutput(this, 'ApplicationEndpoint', { value: `http://${lb.loadBalancerDnsName}:${listenerPort}` });
    }
}