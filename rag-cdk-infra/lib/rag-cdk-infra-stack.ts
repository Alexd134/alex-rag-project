import * as cdk from 'aws-cdk-lib';
import { Architecture, DockerImageCode, DockerImageFunction, FunctionUrlAuthType } from 'aws-cdk-lib/aws-lambda';
import { Construct } from 'constructs';
import { ManagedPolicy } from "aws-cdk-lib/aws-iam";

export class RagCdkInfraStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props?: cdk.StackProps) {
    super(scope, id, props);

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
        // CORS: Allow all origins by default (set to specific domains in production)
        // Example: ALLOWED_ORIGINS: "https://myapp.com,https://www.myapp.com"
        ALLOWED_ORIGINS: "*"  // Change to specific domains for production
      }
    });

    const functionUrl = apiFunction.addFunctionUrl({
      authType: FunctionUrlAuthType.NONE
    })

    apiFunction.role?.addManagedPolicy(
      ManagedPolicy.fromAwsManagedPolicyName("AmazonBedrockFullAccess")
    );


    new cdk.CfnOutput(this, "FunctionUrl", {
      value: functionUrl.url
    });
  }
}
