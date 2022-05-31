"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.KubernetesTrialConfig = exports.KubernetesTrialConfigTemplate = exports.AzureStorage = exports.KeyVaultConfig = exports.PVCConfig = exports.NFSConfig = exports.KubernetesClusterConfigFactory = exports.KubernetesClusterConfigPVC = exports.KubernetesClusterConfigAzure = exports.KubernetesClusterConfigNFS = exports.StorageConfig = exports.KubernetesClusterConfig = void 0;
const errors_1 = require("common/errors");
class KubernetesClusterConfig {
    storage;
    apiVersion;
    namespace;
    constructor(apiVersion, storage, namespace) {
        this.storage = storage;
        this.apiVersion = apiVersion;
        this.namespace = namespace;
    }
    get storageType() {
        throw new errors_1.MethodNotImplementedError();
    }
}
exports.KubernetesClusterConfig = KubernetesClusterConfig;
class StorageConfig {
    storage;
    constructor(storage) {
        this.storage = storage;
    }
}
exports.StorageConfig = StorageConfig;
class KubernetesClusterConfigNFS extends KubernetesClusterConfig {
    nfs;
    constructor(apiVersion, nfs, storage, namespace) {
        super(apiVersion, storage, namespace);
        this.nfs = nfs;
    }
    get storageType() {
        return 'nfs';
    }
    static getInstance(jsonObject) {
        const kubernetesClusterConfigObjectNFS = jsonObject;
        return new KubernetesClusterConfigNFS(kubernetesClusterConfigObjectNFS.apiVersion, kubernetesClusterConfigObjectNFS.nfs, kubernetesClusterConfigObjectNFS.storage, kubernetesClusterConfigObjectNFS.namespace);
    }
}
exports.KubernetesClusterConfigNFS = KubernetesClusterConfigNFS;
class KubernetesClusterConfigAzure extends KubernetesClusterConfig {
    keyVault;
    azureStorage;
    uploadRetryCount;
    constructor(apiVersion, keyVault, azureStorage, storage, uploadRetryCount, namespace) {
        super(apiVersion, storage, namespace);
        this.keyVault = keyVault;
        this.azureStorage = azureStorage;
        this.uploadRetryCount = uploadRetryCount;
    }
    get storageType() {
        return 'azureStorage';
    }
    static getInstance(jsonObject) {
        const kubernetesClusterConfigObjectAzure = jsonObject;
        return new KubernetesClusterConfigAzure(kubernetesClusterConfigObjectAzure.apiVersion, kubernetesClusterConfigObjectAzure.keyVault, kubernetesClusterConfigObjectAzure.azureStorage, kubernetesClusterConfigObjectAzure.storage, kubernetesClusterConfigObjectAzure.uploadRetryCount, kubernetesClusterConfigObjectAzure.namespace);
    }
}
exports.KubernetesClusterConfigAzure = KubernetesClusterConfigAzure;
class KubernetesClusterConfigPVC extends KubernetesClusterConfig {
    pvc;
    constructor(apiVersion, pvc, storage, namespace) {
        super(apiVersion, storage, namespace);
        this.pvc = pvc;
    }
    get storageType() {
        return 'pvc';
    }
    static getInstance(jsonObject) {
        const kubernetesClusterConfigObjectPVC = jsonObject;
        return new KubernetesClusterConfigPVC(kubernetesClusterConfigObjectPVC.apiVersion, kubernetesClusterConfigObjectPVC.pvc, kubernetesClusterConfigObjectPVC.storage, kubernetesClusterConfigObjectPVC.namespace);
    }
}
exports.KubernetesClusterConfigPVC = KubernetesClusterConfigPVC;
class KubernetesClusterConfigFactory {
    static generateKubernetesClusterConfig(jsonObject) {
        const storageConfig = jsonObject;
        switch (storageConfig.storage) {
            case 'azureStorage':
                return KubernetesClusterConfigAzure.getInstance(jsonObject);
            case 'pvc':
                return KubernetesClusterConfigPVC.getInstance(jsonObject);
            case 'nfs':
            case undefined:
                return KubernetesClusterConfigNFS.getInstance(jsonObject);
            default:
                throw new Error(`Invalid json object ${jsonObject}`);
        }
    }
}
exports.KubernetesClusterConfigFactory = KubernetesClusterConfigFactory;
class NFSConfig {
    server;
    path;
    constructor(server, path) {
        this.server = server;
        this.path = path;
    }
}
exports.NFSConfig = NFSConfig;
class PVCConfig {
    path;
    constructor(path) {
        this.path = path;
    }
}
exports.PVCConfig = PVCConfig;
class KeyVaultConfig {
    vaultName;
    name;
    constructor(vaultName, name) {
        this.vaultName = vaultName;
        this.name = name;
    }
}
exports.KeyVaultConfig = KeyVaultConfig;
class AzureStorage {
    azureShare;
    accountName;
    constructor(azureShare, accountName) {
        this.azureShare = azureShare;
        this.accountName = accountName;
    }
}
exports.AzureStorage = AzureStorage;
class KubernetesTrialConfigTemplate {
    cpuNum;
    memoryMB;
    image;
    privateRegistryAuthPath;
    command;
    gpuNum;
    constructor(command, gpuNum, cpuNum, memoryMB, image, privateRegistryAuthPath) {
        this.command = command;
        this.gpuNum = gpuNum;
        this.cpuNum = cpuNum;
        this.memoryMB = memoryMB;
        this.image = image;
        this.privateRegistryAuthPath = privateRegistryAuthPath;
    }
}
exports.KubernetesTrialConfigTemplate = KubernetesTrialConfigTemplate;
class KubernetesTrialConfig {
    codeDir;
    constructor(codeDir) {
        this.codeDir = codeDir;
    }
}
exports.KubernetesTrialConfig = KubernetesTrialConfig;
