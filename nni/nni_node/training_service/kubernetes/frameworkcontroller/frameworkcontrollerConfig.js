"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.FrameworkControllerClusterConfigFactory = exports.FrameworkControllerClusterConfigAzure = exports.FrameworkControllerClusterConfigNFS = exports.FrameworkControllerClusterConfig = exports.FrameworkControllerTrialConfig = exports.FrameworkControllerTrialConfigTemplate = exports.FrameworkAttemptCompletionPolicy = void 0;
const assert_1 = __importDefault(require("assert"));
const kubernetesConfig_1 = require("../kubernetesConfig");
class FrameworkAttemptCompletionPolicy {
    minFailedTaskCount;
    minSucceededTaskCount;
    constructor(minFailedTaskCount, minSucceededTaskCount) {
        this.minFailedTaskCount = minFailedTaskCount;
        this.minSucceededTaskCount = minSucceededTaskCount;
    }
}
exports.FrameworkAttemptCompletionPolicy = FrameworkAttemptCompletionPolicy;
class FrameworkControllerTrialConfigTemplate extends kubernetesConfig_1.KubernetesTrialConfigTemplate {
    frameworkAttemptCompletionPolicy;
    name;
    taskNum;
    constructor(name, taskNum, command, gpuNum, cpuNum, memoryMB, image, frameworkAttemptCompletionPolicy, privateRegistryFilePath) {
        super(command, gpuNum, cpuNum, memoryMB, image, privateRegistryFilePath);
        this.frameworkAttemptCompletionPolicy = frameworkAttemptCompletionPolicy;
        this.name = name;
        this.taskNum = taskNum;
    }
}
exports.FrameworkControllerTrialConfigTemplate = FrameworkControllerTrialConfigTemplate;
class FrameworkControllerTrialConfig extends kubernetesConfig_1.KubernetesTrialConfig {
    taskRoles;
    codeDir;
    constructor(codeDir, taskRoles) {
        super(codeDir);
        this.taskRoles = taskRoles;
        this.codeDir = codeDir;
    }
}
exports.FrameworkControllerTrialConfig = FrameworkControllerTrialConfig;
class FrameworkControllerClusterConfig extends kubernetesConfig_1.KubernetesClusterConfig {
    serviceAccountName;
    constructor(apiVersion, serviceAccountName, _configPath, namespace) {
        super(apiVersion, undefined, namespace);
        this.serviceAccountName = serviceAccountName;
    }
}
exports.FrameworkControllerClusterConfig = FrameworkControllerClusterConfig;
class FrameworkControllerClusterConfigNFS extends kubernetesConfig_1.KubernetesClusterConfigNFS {
    serviceAccountName;
    configPath;
    constructor(serviceAccountName, apiVersion, nfs, storage, namespace, configPath) {
        super(apiVersion, nfs, storage, namespace);
        this.serviceAccountName = serviceAccountName;
        this.configPath = configPath;
    }
    static getInstance(jsonObject) {
        const kubernetesClusterConfigObjectNFS = jsonObject;
        assert_1.default(kubernetesClusterConfigObjectNFS !== undefined);
        return new FrameworkControllerClusterConfigNFS(kubernetesClusterConfigObjectNFS.serviceAccountName, kubernetesClusterConfigObjectNFS.apiVersion, kubernetesClusterConfigObjectNFS.nfs, kubernetesClusterConfigObjectNFS.storage, kubernetesClusterConfigObjectNFS.namespace);
    }
}
exports.FrameworkControllerClusterConfigNFS = FrameworkControllerClusterConfigNFS;
class FrameworkControllerClusterConfigAzure extends kubernetesConfig_1.KubernetesClusterConfigAzure {
    serviceAccountName;
    configPath;
    constructor(serviceAccountName, apiVersion, keyVault, azureStorage, storage, uploadRetryCount, namespace, configPath) {
        super(apiVersion, keyVault, azureStorage, storage, uploadRetryCount, namespace);
        this.serviceAccountName = serviceAccountName;
        this.configPath = configPath;
    }
    static getInstance(jsonObject) {
        const kubernetesClusterConfigObjectAzure = jsonObject;
        return new FrameworkControllerClusterConfigAzure(kubernetesClusterConfigObjectAzure.serviceAccountName, kubernetesClusterConfigObjectAzure.apiVersion, kubernetesClusterConfigObjectAzure.keyVault, kubernetesClusterConfigObjectAzure.azureStorage, kubernetesClusterConfigObjectAzure.storage, kubernetesClusterConfigObjectAzure.uploadRetryCount, kubernetesClusterConfigObjectAzure.namespace);
    }
}
exports.FrameworkControllerClusterConfigAzure = FrameworkControllerClusterConfigAzure;
class FrameworkControllerClusterConfigFactory {
    static generateFrameworkControllerClusterConfig(jsonObject) {
        const storageConfig = jsonObject;
        if (storageConfig === undefined) {
            throw new Error('Invalid json object as a StorageConfig instance');
        }
        if (storageConfig.storage !== undefined && storageConfig.storage === 'azureStorage') {
            return FrameworkControllerClusterConfigAzure.getInstance(jsonObject);
        }
        else if (storageConfig.storage === undefined || storageConfig.storage === 'nfs') {
            return FrameworkControllerClusterConfigNFS.getInstance(jsonObject);
        }
        throw new Error(`Invalid json object ${jsonObject}`);
    }
}
exports.FrameworkControllerClusterConfigFactory = FrameworkControllerClusterConfigFactory;
