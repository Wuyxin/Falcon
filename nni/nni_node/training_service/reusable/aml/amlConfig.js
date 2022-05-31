"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.AMLEnvironmentInformation = exports.AMLTrialConfig = exports.AMLClusterConfig = void 0;
const trialConfig_1 = require("training_service/common/trialConfig");
const environment_1 = require("../environment");
class AMLClusterConfig {
    subscriptionId;
    resourceGroup;
    workspaceName;
    computeTarget;
    maxTrialNumPerGpu;
    constructor(subscriptionId, resourceGroup, workspaceName, computeTarget, maxTrialNumPerGpu) {
        this.subscriptionId = subscriptionId;
        this.resourceGroup = resourceGroup;
        this.workspaceName = workspaceName;
        this.computeTarget = computeTarget;
        this.maxTrialNumPerGpu = maxTrialNumPerGpu;
    }
}
exports.AMLClusterConfig = AMLClusterConfig;
class AMLTrialConfig extends trialConfig_1.TrialConfig {
    image;
    command;
    codeDir;
    constructor(codeDir, command, image) {
        super("", codeDir, 0);
        this.codeDir = codeDir;
        this.command = command;
        this.image = image;
    }
}
exports.AMLTrialConfig = AMLTrialConfig;
class AMLEnvironmentInformation extends environment_1.EnvironmentInformation {
    amlClient;
    currentMessageIndex = -1;
}
exports.AMLEnvironmentInformation = AMLEnvironmentInformation;
