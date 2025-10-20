import * as cdk from 'aws-cdk-lib';
import { Architecture, DockerImageCode, DockerImageFunction, FunctionUrlAuthType } from 'aws-cdk-lib/aws-lambda';
import { Construct } from 'constructs';
import { ManagedPolicy } from "aws-cdk-lib/aws-iam";
import { AttributeType, BillingMode, Table } from "aws-cdk-lib/aws-dynamodb";
import * as sqs from 'aws-cdk-lib/aws-sqs';
import { SqsEventSource } from 'aws-cdk-lib/aws-lambda-event-sources';

export class RagCdkInfraStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props?: cdk.StackProps) {
    super(scope, id, props);

    const ragQueryTable = new Table(this, "RagQueryTable", {
      partitionKey: { name: "query_id", type: AttributeType.STRING },
      billingMode: BillingMode.PAY_PER_REQUEST,
    });

    const workerImageCode = DockerImageCode.fromImageAsset("../docker-image", {
      cmd: ["work_handler.handler"],
      buildArgs: {
        platform: "linux/amd64", 
      },
    });

    const workerFunction = new DockerImageFunction(this, "RagWorkerFunction", {
      code: workerImageCode,
      memorySize: 512,
      timeout: cdk.Duration.seconds(60), 
      architecture: Architecture.X86_64,
      environment: {
        TABLE_NAME: ragQueryTable.tableName,
      },
    });

    const apiImageCode = DockerImageCode.fromImageAsset("../docker-image", {
      cmd: ["api_handler.handler"],
      buildArgs: {
        platform: "linux/amd64",
      },
    });

    const apiFunction = new DockerImageFunction(this, "ApiFunc", {
      code: apiImageCode,
      memorySize: 256,
      timeout: cdk.Duration.seconds(29),
      architecture: Architecture.X86_64,
      environment: {
        IS_USING_IMAGE_RUNTIME: "1",  // Enable copying ChromaDB to /tmp
        CHROMA_PATH: "src/data/chroma",  // Source path in Docker image
        TABLE_NAME: ragQueryTable.tableName,
      }
    });

    const functionUrl = apiFunction.addFunctionUrl({
      authType: FunctionUrlAuthType.NONE
    })

    const jobsDlq = new sqs.Queue(this, 'JobsDLQ', {
      retentionPeriod: cdk.Duration.days(14),
      encryption: sqs.QueueEncryption.KMS_MANAGED,
    });

    const jobsQueue = new sqs.Queue(this, 'JobsQueue', {
      visibilityTimeout: cdk.Duration.seconds(120),
      retentionPeriod: cdk.Duration.days(4),
      encryption: sqs.QueueEncryption.KMS_MANAGED,
      deadLetterQueue: {
        maxReceiveCount: 3,
        queue: jobsDlq,
      },
    });

    jobsQueue.grantSendMessages(apiFunction);
    jobsQueue.grantConsumeMessages(workerFunction);

    workerFunction.addEventSource(
      new SqsEventSource(jobsQueue, {
        batchSize: 5,
        maxBatchingWindow: cdk.Duration.seconds(1),
      })
    );

    // API needs the queue URL to send jobs
    apiFunction.addEnvironment('JOBS_QUEUE_URL', jobsQueue.queueUrl);

    ragQueryTable.grantReadWriteData(workerFunction);
    ragQueryTable.grantReadWriteData(apiFunction);

    workerFunction.role?.addManagedPolicy(
      ManagedPolicy.fromAwsManagedPolicyName("AmazonBedrockFullAccess")
    );
    apiFunction.role?.addManagedPolicy(
      ManagedPolicy.fromAwsManagedPolicyName("AmazonBedrockFullAccess")
    );


    new cdk.CfnOutput(this, "FunctionUrl", {
      value: functionUrl.url
    });
  }
}
