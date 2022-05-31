"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    Object.defineProperty(o, k2, { enumerable: true, get: function() { return m[k]; } });
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __decorate = (this && this.__decorate) || function (decorators, target, key, desc) {
    var c = arguments.length, r = c < 3 ? target : desc === null ? desc = Object.getOwnPropertyDescriptor(target, key) : desc, d;
    if (typeof Reflect === "object" && typeof Reflect.decorate === "function") r = Reflect.decorate(decorators, target, key, desc);
    else for (var i = decorators.length - 1; i >= 0; i--) if (d = decorators[i]) r = (c < 3 ? d(r) : c > 3 ? d(target, key, r) : d(target, key)) || r;
    return c > 3 && r && Object.defineProperty(target, key, r), r;
};
var __importStar = (this && this.__importStar) || function (mod) {
    if (mod && mod.__esModule) return mod;
    var result = {};
    if (mod != null) for (var k in mod) if (k !== "default" && Object.prototype.hasOwnProperty.call(mod, k)) __createBinding(result, mod, k);
    __setModuleDefault(result, mod);
    return result;
};
var __metadata = (this && this.__metadata) || function (k, v) {
    if (typeof Reflect === "object" && typeof Reflect.metadata === "function") return Reflect.metadata(k, v);
};
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.KubeflowTrainingService = void 0;
const assert_1 = __importDefault(require("assert"));
const child_process_promise_1 = __importDefault(require("child-process-promise"));
const fs_1 = __importDefault(require("fs"));
const path_1 = __importDefault(require("path"));
const component = __importStar(require("common/component"));
const experimentStartupInfo_1 = require("common/experimentStartupInfo");
const utils_1 = require("common/utils");
const containerJobData_1 = require("training_service/common/containerJobData");
const trialConfigMetadataKey_1 = require("training_service/common/trialConfigMetadataKey");
const util_1 = require("training_service/common/util");
const kubernetesData_1 = require("../kubernetesData");
const kubernetesJobRestServer_1 = require("../kubernetesJobRestServer");
const kubernetesTrainingService_1 = require("../kubernetesTrainingService");
const kubeflowApiClient_1 = require("./kubeflowApiClient");
const kubeflowConfig_1 = require("./kubeflowConfig");
const kubeflowJobInfoCollector_1 = require("./kubeflowJobInfoCollector");
const kubeflowJobRestServer_1 = require("./kubeflowJobRestServer");
let KubeflowTrainingService = class KubeflowTrainingService extends kubernetesTrainingService_1.KubernetesTrainingService {
    kubeflowClusterConfig;
    kubeflowTrialConfig;
    kubeflowJobInfoCollector;
    constructor() {
        super();
        this.kubeflowJobInfoCollector = new kubeflowJobInfoCollector_1.KubeflowJobInfoCollector(this.trialJobsMap);
        this.experimentId = experimentStartupInfo_1.getExperimentId();
        this.log.info('Construct Kubeflow training service.');
    }
    async run() {
        this.log.info('Run Kubeflow training service.');
        this.kubernetesJobRestServer = new kubernetesJobRestServer_1.KubernetesJobRestServer(this);
        if (this.kubernetesJobRestServer === undefined) {
            throw new Error('kubernetesJobRestServer not initialized!');
        }
        await this.kubernetesJobRestServer.start();
        this.kubernetesJobRestServer.setEnableVersionCheck = this.versionCheck;
        this.log.info(`Kubeflow Training service rest server listening on: ${this.kubernetesJobRestServer.endPoint}`);
        while (!this.stopping) {
            await utils_1.delay(3000);
            await this.kubeflowJobInfoCollector.retrieveTrialStatus(this.kubernetesCRDClient);
            if (this.kubernetesJobRestServer.getErrorMessage !== undefined) {
                throw new Error(this.kubernetesJobRestServer.getErrorMessage);
            }
        }
        this.log.info('Kubeflow training service exit.');
    }
    async submitTrialJob(form) {
        if (this.kubernetesCRDClient === undefined) {
            throw new Error('Kubeflow job operator client is undefined');
        }
        if (this.kubernetesRestServerPort === undefined) {
            const restServer = new kubeflowJobRestServer_1.KubeflowJobRestServer(this);
            this.kubernetesRestServerPort = restServer.clusterRestServerPort;
        }
        if (this.copyExpCodeDirPromise !== undefined) {
            await this.copyExpCodeDirPromise;
        }
        const trialJobId = utils_1.uniqueString(5);
        const trialWorkingFolder = path_1.default.join(this.CONTAINER_MOUNT_PATH, 'nni', experimentStartupInfo_1.getExperimentId(), trialJobId);
        const kubeflowJobName = `nni-exp-${this.experimentId}-trial-${trialJobId}`.toLowerCase();
        const trialLocalTempFolder = path_1.default.join(utils_1.getExperimentRootDir(), 'trials', trialJobId);
        await this.prepareRunScript(trialLocalTempFolder, trialJobId, trialWorkingFolder, form);
        const trialJobOutputUrl = await this.uploadFolder(trialLocalTempFolder, `nni/${experimentStartupInfo_1.getExperimentId()}/${trialJobId}`);
        let initStatus = 'WAITING';
        if (!trialJobOutputUrl) {
            initStatus = 'FAILED';
        }
        const trialJobDetail = new kubernetesData_1.KubernetesTrialJobDetail(trialJobId, initStatus, Date.now(), trialWorkingFolder, form, kubeflowJobName, trialJobOutputUrl);
        const kubeflowJobConfig = await this.prepareKubeflowConfig(trialJobId, trialWorkingFolder, kubeflowJobName);
        await this.kubernetesCRDClient.createKubernetesJob(kubeflowJobConfig);
        this.trialJobsMap.set(trialJobId, trialJobDetail);
        return Promise.resolve(trialJobDetail);
    }
    async setClusterMetadata(key, value) {
        switch (key) {
            case trialConfigMetadataKey_1.TrialConfigMetadataKey.NNI_MANAGER_IP:
                this.nniManagerIpConfig = JSON.parse(value);
                break;
            case trialConfigMetadataKey_1.TrialConfigMetadataKey.KUBEFLOW_CLUSTER_CONFIG: {
                const kubeflowClusterJsonObject = JSON.parse(value);
                this.kubeflowClusterConfig = kubeflowConfig_1.KubeflowClusterConfigFactory.generateKubeflowClusterConfig(kubeflowClusterJsonObject);
                this.genericK8sClient.setNamespace = this.kubeflowClusterConfig.namespace ?? "default";
                if (this.kubeflowClusterConfig.storageType === 'azureStorage') {
                    const azureKubeflowClusterConfig = this.kubeflowClusterConfig;
                    this.azureStorageAccountName = azureKubeflowClusterConfig.azureStorage.accountName;
                    this.azureStorageShare = azureKubeflowClusterConfig.azureStorage.azureShare;
                    await this.createAzureStorage(azureKubeflowClusterConfig.keyVault.vaultName, azureKubeflowClusterConfig.keyVault.name);
                }
                else if (this.kubeflowClusterConfig.storageType === 'nfs') {
                    const nfsKubeflowClusterConfig = this.kubeflowClusterConfig;
                    await this.createNFSStorage(nfsKubeflowClusterConfig.nfs.server, nfsKubeflowClusterConfig.nfs.path);
                }
                this.kubernetesCRDClient = kubeflowApiClient_1.KubeflowOperatorClientFactory.createClient(this.kubeflowClusterConfig.operator, this.kubeflowClusterConfig.apiVersion);
                this.kubernetesCRDClient.namespace = this.kubeflowClusterConfig.namespace ?? "default";
                break;
            }
            case trialConfigMetadataKey_1.TrialConfigMetadataKey.TRIAL_CONFIG: {
                if (this.kubeflowClusterConfig === undefined) {
                    this.log.error('kubeflow cluster config is not initialized');
                    return Promise.reject(new Error('kubeflow cluster config is not initialized'));
                }
                assert_1.default(this.kubeflowClusterConfig !== undefined);
                const kubeflowTrialJsonObjsect = JSON.parse(value);
                this.kubeflowTrialConfig = kubeflowConfig_1.KubeflowTrialConfigFactory.generateKubeflowTrialConfig(kubeflowTrialJsonObjsect, this.kubeflowClusterConfig.operator);
                try {
                    await util_1.validateCodeDir(this.kubeflowTrialConfig.codeDir);
                    this.copyExpCodeDirPromise = this.uploadFolder(this.kubeflowTrialConfig.codeDir, `nni/${experimentStartupInfo_1.getExperimentId()}/nni-code`);
                }
                catch (error) {
                    this.log.error(error);
                    return Promise.reject(new Error(error));
                }
                break;
            }
            case trialConfigMetadataKey_1.TrialConfigMetadataKey.VERSION_CHECK:
                this.versionCheck = (value === 'true' || value === 'True');
                break;
            case trialConfigMetadataKey_1.TrialConfigMetadataKey.LOG_COLLECTION:
                this.logCollection = value;
                break;
            default:
        }
        return Promise.resolve();
    }
    async uploadFolder(srcDirectory, destDirectory) {
        if (this.kubeflowClusterConfig === undefined) {
            throw new Error('Kubeflow Cluster config is not initialized');
        }
        if (this.kubeflowTrialConfig === undefined) {
            throw new Error('Kubeflow Trial config is not initialized');
        }
        assert_1.default(this.kubeflowClusterConfig.storage === undefined
            || this.kubeflowClusterConfig.storage === 'azureStorage'
            || this.kubeflowClusterConfig.storage === 'nfs');
        if (this.kubeflowClusterConfig.storage === 'azureStorage') {
            if (this.azureStorageClient === undefined) {
                throw new Error('azureStorageClient is not initialized');
            }
            const azureKubeflowClusterConfig = this.kubeflowClusterConfig;
            return await this.uploadFolderToAzureStorage(srcDirectory, destDirectory, azureKubeflowClusterConfig.uploadRetryCount);
        }
        else if (this.kubeflowClusterConfig.storage === 'nfs' || this.kubeflowClusterConfig.storage === undefined) {
            await child_process_promise_1.default.exec(`mkdir -p ${this.trialLocalTempFolder}/${destDirectory}`);
            await child_process_promise_1.default.exec(`cp -r ${srcDirectory}/* ${this.trialLocalTempFolder}/${destDirectory}/.`);
            const nfsKubeflowClusterConfig = this.kubeflowClusterConfig;
            const nfsConfig = nfsKubeflowClusterConfig.nfs;
            return `nfs://${nfsConfig.server}:${destDirectory}`;
        }
        return '';
    }
    async prepareRunScript(trialLocalTempFolder, trialJobId, trialWorkingFolder, form) {
        if (this.kubeflowClusterConfig === undefined) {
            throw new Error('Kubeflow Cluster config is not initialized');
        }
        let kubeflowTrialConfig;
        if (this.kubeflowClusterConfig.operator === 'tf-operator') {
            kubeflowTrialConfig = this.kubeflowTrialConfig;
        }
        else if (this.kubeflowClusterConfig.operator === 'pytorch-operator') {
            kubeflowTrialConfig = this.kubeflowTrialConfig;
        }
        else {
            throw Error(`operator ${this.kubeflowClusterConfig.operator} is invalid`);
        }
        await child_process_promise_1.default.exec(`mkdir -p ${trialLocalTempFolder}`);
        const runScriptContent = containerJobData_1.CONTAINER_INSTALL_NNI_SHELL_FORMAT;
        await fs_1.default.promises.writeFile(path_1.default.join(trialLocalTempFolder, 'install_nni.sh'), runScriptContent, { encoding: 'utf8' });
        if (kubeflowTrialConfig.worker !== undefined) {
            const workerRunScriptContent = await this.generateRunScript('kubeflow', trialJobId, trialWorkingFolder, kubeflowTrialConfig.worker.command, form.sequenceId.toString(), 'worker', kubeflowTrialConfig.worker.gpuNum);
            await fs_1.default.promises.writeFile(path_1.default.join(trialLocalTempFolder, 'run_worker.sh'), workerRunScriptContent, { encoding: 'utf8' });
        }
        if (this.kubeflowClusterConfig.operator === 'tf-operator') {
            const tensorflowTrialConfig = this.kubeflowTrialConfig;
            if (tensorflowTrialConfig.ps !== undefined) {
                const psRunScriptContent = await this.generateRunScript('kubeflow', trialJobId, trialWorkingFolder, tensorflowTrialConfig.ps.command, form.sequenceId.toString(), 'ps', tensorflowTrialConfig.ps.gpuNum);
                await fs_1.default.promises.writeFile(path_1.default.join(trialLocalTempFolder, 'run_ps.sh'), psRunScriptContent, { encoding: 'utf8' });
            }
        }
        else if (this.kubeflowClusterConfig.operator === 'pytorch-operator') {
            const pytorchTrialConfig = this.kubeflowTrialConfig;
            if (pytorchTrialConfig.master !== undefined) {
                const masterRunScriptContent = await this.generateRunScript('kubeflow', trialJobId, trialWorkingFolder, pytorchTrialConfig.master.command, form.sequenceId.toString(), 'master', pytorchTrialConfig.master.gpuNum);
                await fs_1.default.promises.writeFile(path_1.default.join(trialLocalTempFolder, 'run_master.sh'), masterRunScriptContent, { encoding: 'utf8' });
            }
        }
        if (form !== undefined) {
            await fs_1.default.promises.writeFile(path_1.default.join(trialLocalTempFolder, utils_1.generateParamFileName(form.hyperParameters)), form.hyperParameters.value, { encoding: 'utf8' });
        }
    }
    async prepareKubeflowConfig(trialJobId, trialWorkingFolder, kubeflowJobName) {
        if (this.kubeflowClusterConfig === undefined) {
            throw new Error('Kubeflow Cluster config is not initialized');
        }
        if (this.kubeflowTrialConfig === undefined) {
            throw new Error('Kubeflow trial config is not initialized');
        }
        let kubeflowTrialConfig;
        if (this.kubeflowClusterConfig.operator === 'tf-operator') {
            kubeflowTrialConfig = this.kubeflowTrialConfig;
        }
        else if (this.kubeflowClusterConfig.operator === 'pytorch-operator') {
            kubeflowTrialConfig = this.kubeflowTrialConfig;
        }
        else {
            throw Error(`operator ${this.kubeflowClusterConfig.operator} is invalid`);
        }
        const workerPodResources = {};
        if (kubeflowTrialConfig.worker !== undefined) {
            workerPodResources.requests = this.generatePodResource(kubeflowTrialConfig.worker.memoryMB, kubeflowTrialConfig.worker.cpuNum, kubeflowTrialConfig.worker.gpuNum);
        }
        workerPodResources.limits = { ...workerPodResources.requests };
        const nonWorkerResources = {};
        if (this.kubeflowClusterConfig.operator === 'tf-operator') {
            const tensorflowTrialConfig = this.kubeflowTrialConfig;
            if (tensorflowTrialConfig.ps !== undefined) {
                nonWorkerResources.requests = this.generatePodResource(tensorflowTrialConfig.ps.memoryMB, tensorflowTrialConfig.ps.cpuNum, tensorflowTrialConfig.ps.gpuNum);
                nonWorkerResources.limits = { ...nonWorkerResources.requests };
            }
        }
        else if (this.kubeflowClusterConfig.operator === 'pytorch-operator') {
            const pyTorchTrialConfig = this.kubeflowTrialConfig;
            nonWorkerResources.requests = this.generatePodResource(pyTorchTrialConfig.master.memoryMB, pyTorchTrialConfig.master.cpuNum, pyTorchTrialConfig.master.gpuNum);
            nonWorkerResources.limits = { ...nonWorkerResources.requests };
        }
        const kubeflowJobConfig = await this.generateKubeflowJobConfig(trialJobId, trialWorkingFolder, kubeflowJobName, workerPodResources, nonWorkerResources);
        this.log.info('kubeflowJobConfig:', kubeflowJobConfig);
        return Promise.resolve(kubeflowJobConfig);
    }
    async generateKubeflowJobConfig(trialJobId, trialWorkingFolder, kubeflowJobName, workerPodResources, nonWorkerPodResources) {
        if (this.kubeflowClusterConfig === undefined) {
            throw new Error('Kubeflow Cluster config is not initialized');
        }
        if (this.kubeflowTrialConfig === undefined) {
            throw new Error('Kubeflow trial config is not initialized');
        }
        if (this.kubernetesCRDClient === undefined) {
            throw new Error('Kubeflow operator client is not initialized');
        }
        const replicaSpecsObj = {};
        const replicaSpecsObjMap = new Map();
        if (this.kubeflowTrialConfig.operatorType === 'tf-operator') {
            const tensorflowTrialConfig = this.kubeflowTrialConfig;
            const privateRegistrySecretName = await this.createRegistrySecret(tensorflowTrialConfig.worker.privateRegistryAuthPath);
            replicaSpecsObj.Worker = this.generateReplicaConfig(trialWorkingFolder, tensorflowTrialConfig.worker.replicas, tensorflowTrialConfig.worker.image, 'run_worker.sh', workerPodResources, privateRegistrySecretName);
            if (tensorflowTrialConfig.ps !== undefined) {
                const privateRegistrySecretName = await this.createRegistrySecret(tensorflowTrialConfig.ps.privateRegistryAuthPath);
                replicaSpecsObj.Ps = this.generateReplicaConfig(trialWorkingFolder, tensorflowTrialConfig.ps.replicas, tensorflowTrialConfig.ps.image, 'run_ps.sh', nonWorkerPodResources, privateRegistrySecretName);
            }
            replicaSpecsObjMap.set(this.kubernetesCRDClient.jobKind, { tfReplicaSpecs: replicaSpecsObj });
        }
        else if (this.kubeflowTrialConfig.operatorType === 'pytorch-operator') {
            const pytorchTrialConfig = this.kubeflowTrialConfig;
            if (pytorchTrialConfig.worker !== undefined) {
                const privateRegistrySecretName = await this.createRegistrySecret(pytorchTrialConfig.worker.privateRegistryAuthPath);
                replicaSpecsObj.Worker = this.generateReplicaConfig(trialWorkingFolder, pytorchTrialConfig.worker.replicas, pytorchTrialConfig.worker.image, 'run_worker.sh', workerPodResources, privateRegistrySecretName);
            }
            const privateRegistrySecretName = await this.createRegistrySecret(pytorchTrialConfig.master.privateRegistryAuthPath);
            replicaSpecsObj.Master = this.generateReplicaConfig(trialWorkingFolder, pytorchTrialConfig.master.replicas, pytorchTrialConfig.master.image, 'run_master.sh', nonWorkerPodResources, privateRegistrySecretName);
            replicaSpecsObjMap.set(this.kubernetesCRDClient.jobKind, { pytorchReplicaSpecs: replicaSpecsObj });
        }
        return Promise.resolve({
            apiVersion: `kubeflow.org/${this.kubernetesCRDClient.apiVersion}`,
            kind: this.kubernetesCRDClient.jobKind,
            metadata: {
                name: kubeflowJobName,
                namespace: this.kubernetesCRDClient.namespace,
                labels: {
                    app: this.NNI_KUBERNETES_TRIAL_LABEL,
                    expId: experimentStartupInfo_1.getExperimentId(),
                    trialId: trialJobId
                }
            },
            spec: replicaSpecsObjMap.get(this.kubernetesCRDClient.jobKind)
        });
    }
    generateReplicaConfig(trialWorkingFolder, replicaNumber, replicaImage, runScriptFile, podResources, privateRegistrySecretName) {
        if (this.kubeflowClusterConfig === undefined) {
            throw new Error('Kubeflow Cluster config is not initialized');
        }
        if (this.kubeflowTrialConfig === undefined) {
            throw new Error('Kubeflow trial config is not initialized');
        }
        if (this.kubernetesCRDClient === undefined) {
            throw new Error('Kubeflow operator client is not initialized');
        }
        const volumeSpecMap = new Map();
        if (this.kubeflowClusterConfig.storageType === 'azureStorage') {
            volumeSpecMap.set('nniVolumes', [
                {
                    name: 'nni-vol',
                    azureFile: {
                        secretName: `${this.azureStorageSecretName}`,
                        shareName: `${this.azureStorageShare}`,
                        readonly: false
                    }
                }
            ]);
        }
        else {
            const nfsKubeflowClusterConfig = this.kubeflowClusterConfig;
            volumeSpecMap.set('nniVolumes', [
                {
                    name: 'nni-vol',
                    nfs: {
                        server: `${nfsKubeflowClusterConfig.nfs.server}`,
                        path: `${nfsKubeflowClusterConfig.nfs.path}`
                    }
                }
            ]);
        }
        const containersSpecMap = new Map();
        containersSpecMap.set('containers', [
            {
                name: this.kubernetesCRDClient.containerName,
                image: replicaImage,
                args: ['sh', `${path_1.default.join(trialWorkingFolder, runScriptFile)}`],
                volumeMounts: [
                    {
                        name: 'nni-vol',
                        mountPath: this.CONTAINER_MOUNT_PATH
                    }
                ],
                resources: podResources
            }
        ]);
        const spec = {
            containers: containersSpecMap.get('containers'),
            restartPolicy: 'ExitCode',
            volumes: volumeSpecMap.get('nniVolumes')
        };
        if (privateRegistrySecretName) {
            spec.imagePullSecrets = [
                {
                    name: privateRegistrySecretName
                }
            ];
        }
        return {
            replicas: replicaNumber,
            template: {
                metadata: {
                    creationTimestamp: null
                },
                spec: spec
            }
        };
    }
    async updateTrialJob(_1, _2) {
        throw new Error('not supported');
    }
};
KubeflowTrainingService = __decorate([
    component.Singleton,
    __metadata("design:paramtypes", [])
], KubeflowTrainingService);
exports.KubeflowTrainingService = KubeflowTrainingService;
