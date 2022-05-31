"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.KubeflowTrialConfigFactory = exports.KubeflowTrialConfigPytorch = exports.KubeflowTrialConfigTensorflow = exports.KubeflowTrialConfigTemplate = exports.KubeflowTrialConfig = exports.KubeflowClusterConfigFactory = exports.KubeflowClusterConfigAzure = exports.KubeflowClusterConfigNFS = exports.KubeflowClusterConfig = void 0;
const assert_1 = __importDefault(require("assert"));
const errors_1 = require("common/errors");
const kubernetesConfig_1 = require("../kubernetesConfig");
class KubeflowClusterConfig extends kubernetesConfig_1.KubernetesClusterConfig {
    operator;
    constructor(apiVersion, operator, namespace) {
        super(apiVersion, undefined, namespace);
        this.operator = operator;
    }
}
exports.KubeflowClusterConfig = KubeflowClusterConfig;
class KubeflowClusterConfigNFS extends kubernetesConfig_1.KubernetesClusterConfigNFS {
    operator;
    constructor(operator, apiVersion, nfs, storage, namespace) {
        super(apiVersion, nfs, storage, namespace);
        this.operator = operator;
    }
    get storageType() {
        return 'nfs';
    }
    static getInstance(jsonObject) {
        const kubeflowClusterConfigObjectNFS = jsonObject;
        assert_1.default(kubeflowClusterConfigObjectNFS !== undefined);
        return new KubeflowClusterConfigNFS(kubeflowClusterConfigObjectNFS.operator, kubeflowClusterConfigObjectNFS.apiVersion, kubeflowClusterConfigObjectNFS.nfs, kubeflowClusterConfigObjectNFS.storage, kubeflowClusterConfigObjectNFS.namespace);
    }
}
exports.KubeflowClusterConfigNFS = KubeflowClusterConfigNFS;
class KubeflowClusterConfigAzure extends kubernetesConfig_1.KubernetesClusterConfigAzure {
    operator;
    constructor(operator, apiVersion, keyVault, azureStorage, storage, namespace) {
        super(apiVersion, keyVault, azureStorage, storage, undefined, namespace);
        this.operator = operator;
    }
    get storageType() {
        return 'azureStorage';
    }
    static getInstance(jsonObject) {
        const kubeflowClusterConfigObjectAzure = jsonObject;
        return new KubeflowClusterConfigAzure(kubeflowClusterConfigObjectAzure.operator, kubeflowClusterConfigObjectAzure.apiVersion, kubeflowClusterConfigObjectAzure.keyVault, kubeflowClusterConfigObjectAzure.azureStorage, kubeflowClusterConfigObjectAzure.storage, kubeflowClusterConfigObjectAzure.namespace);
    }
}
exports.KubeflowClusterConfigAzure = KubeflowClusterConfigAzure;
class KubeflowClusterConfigFactory {
    static generateKubeflowClusterConfig(jsonObject) {
        const storageConfig = jsonObject;
        if (storageConfig === undefined) {
            throw new Error('Invalid json object as a StorageConfig instance');
        }
        if (storageConfig.storage !== undefined && storageConfig.storage === 'azureStorage') {
            return KubeflowClusterConfigAzure.getInstance(jsonObject);
        }
        else if (storageConfig.storage === undefined || storageConfig.storage === 'nfs') {
            return KubeflowClusterConfigNFS.getInstance(jsonObject);
        }
        throw new Error(`Invalid json object ${jsonObject}`);
    }
}
exports.KubeflowClusterConfigFactory = KubeflowClusterConfigFactory;
class KubeflowTrialConfig extends kubernetesConfig_1.KubernetesTrialConfig {
    constructor(codeDir) {
        super(codeDir);
    }
    get operatorType() {
        throw new errors_1.MethodNotImplementedError();
    }
}
exports.KubeflowTrialConfig = KubeflowTrialConfig;
class KubeflowTrialConfigTemplate extends kubernetesConfig_1.KubernetesTrialConfigTemplate {
    replicas;
    constructor(replicas, command, gpuNum, cpuNum, memoryMB, image, privateRegistryAuthPath) {
        super(command, gpuNum, cpuNum, memoryMB, image, privateRegistryAuthPath);
        this.replicas = replicas;
    }
}
exports.KubeflowTrialConfigTemplate = KubeflowTrialConfigTemplate;
class KubeflowTrialConfigTensorflow extends KubeflowTrialConfig {
    ps;
    worker;
    constructor(codeDir, worker, ps) {
        super(codeDir);
        this.ps = ps;
        this.worker = worker;
    }
    get operatorType() {
        return 'tf-operator';
    }
}
exports.KubeflowTrialConfigTensorflow = KubeflowTrialConfigTensorflow;
class KubeflowTrialConfigPytorch extends KubeflowTrialConfig {
    master;
    worker;
    constructor(codeDir, master, worker) {
        super(codeDir);
        this.master = master;
        this.worker = worker;
    }
    get operatorType() {
        return 'pytorch-operator';
    }
}
exports.KubeflowTrialConfigPytorch = KubeflowTrialConfigPytorch;
class KubeflowTrialConfigFactory {
    static generateKubeflowTrialConfig(jsonObject, operator) {
        if (operator === 'tf-operator') {
            const kubeflowTrialConfigObject = jsonObject;
            return new KubeflowTrialConfigTensorflow(kubeflowTrialConfigObject.codeDir, kubeflowTrialConfigObject.worker, kubeflowTrialConfigObject.ps);
        }
        else if (operator === 'pytorch-operator') {
            const kubeflowTrialConfigObject = jsonObject;
            return new KubeflowTrialConfigPytorch(kubeflowTrialConfigObject.codeDir, kubeflowTrialConfigObject.master, kubeflowTrialConfigObject.worker);
        }
        throw new Error(`Invalid json object ${jsonObject}`);
    }
}
exports.KubeflowTrialConfigFactory = KubeflowTrialConfigFactory;
