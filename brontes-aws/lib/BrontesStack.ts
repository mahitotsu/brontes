import { Duration, Stack, StackProps } from "aws-cdk-lib";
import { IpProtocol, SubnetType, Vpc } from "aws-cdk-lib/aws-ec2";
import { Cluster, ContainerImage, FargateService, FargateTaskDefinition } from "aws-cdk-lib/aws-ecs";
import { ApplicationLoadBalancer, ApplicationProtocol, ApplicationTargetGroup, TargetType } from "aws-cdk-lib/aws-elasticloadbalancingv2";
import { Construct } from "constructs";

export class BrontesStack extends Stack {

    constructor(scope: Construct, id: string, props: StackProps) {
        super(scope, id, props);

        const maxAzs = 2;
        const listnerPort = 80;
        const containerPort = 8080;

        const taskDefinition = new FargateTaskDefinition(this, 'TaskDefinition', { cpu: 512, memoryLimitMiB: 4096 });
        taskDefinition.addContainer('application', {
            image: ContainerImage.fromAsset(`${__dirname}/../../brontes-api`),
            portMappings: [{ containerPort }],
        });

        const vpc = new Vpc(this, 'Vpc', {
            ipProtocol: IpProtocol.DUAL_STACK,
            createInternetGateway: true,
            maxAzs,
        });

        const lb = new ApplicationLoadBalancer(vpc, 'LoadBalancer', {
            vpc, vpcSubnets: { subnetType: SubnetType.PUBLIC }, internetFacing: true,
        });
        const lister = lb.addListener('HttpListener', { protocol: ApplicationProtocol.HTTP, port: listnerPort });

        const cluster = new Cluster(vpc, 'ApplicationCluster', {
            vpc, enableFargateCapacityProviders: true,
        });
        const service = new FargateService(cluster, 'ApplicationService', {
            taskDefinition, cluster, vpcSubnets: { subnetType: SubnetType.PRIVATE_WITH_EGRESS },
        });

        const targetGroup = new ApplicationTargetGroup(this, 'TargetGroup', {
            vpc, targetType: TargetType.IP, port: containerPort, protocol: ApplicationProtocol.HTTP,
            targets: [service],
            healthCheck: {
                path: '/actuator/health',
                interval: Duration.seconds(5),
                healthyThresholdCount: 2, unhealthyThresholdCount: 2,
            },
            deregistrationDelay: Duration.seconds(0),
        });
        lister.addTargetGroups('TargetGroups', { targetGroups: [targetGroup] });
    }
}