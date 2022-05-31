"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.KubernetesEnvironmentService = void 0;
const child_process_promise_1 = __importDefault(require("child-process-promise"));
const path_1 = __importDefault(require("path"));
const azure_storage_1 = __importDefault(require("azure-storage"));
const js_base64_1 = require("js-base64");
const typescript_string_operations_1 = require("typescript-string-operations");
const log_1 = require("common/log");
const environment_1 = require("training_service/reusable/environment");
const kubernetesApiClient_1 = require("training_service/kubernetes/kubernetesApiClient");
const azureStorageClientUtils_1 = require("training_service/kubernetes/azureStorageClientUtils");
const utils_1 = require("common/utils");
const fs = require('fs');
class KubernetesEnvironmentService extends environment_1.EnvironmentService {
    azureStorageClient;
    azureStorageShare;
    azureStorageSecretName;
    azureStorageAccountName;
    genericK8sClient;
    kubernetesCRDClient;
    experimentRootDir;
    experimentId;
    environmentLocalTempFolder;
    NNI_KUBERNETES_TRIAL_LABEL = 'nni-kubernetes-trial';
    CONTAINER_MOUNT_PATH;
    log = log_1.getLogger('KubernetesEnvironmentService');
    environmentWorkingFolder;
    constructor(_config, info) {
        super();
        this.CONTAINER_MOUNT_PATH = '/tmp/mount';
        this.genericK8sClient = new kubernetesApiClient_1.GeneralK8sClient();
        this.experimentRootDir = info.logDir;
        this.environmentLocalTempFolder = path_1.default.join(this.experimentRootDir, "environment-temp");
        this.experimentId = info.experimentId;
        this.environmentWorkingFolder = path_1.default.join(this.CONTAINER_MOUNT_PATH, 'nni', this.experimentId);
    }
    get environmentMaintenceLoopInterval() {
        return 5000;
    }
    get hasStorageService() {
        return false;
    }
    get getName() {
        return 'kubernetes';
    }
    async createAzureStorage(vaultName, valutKeyName) {
        try {
            const result = await child_process_promise_1.default.exec(`az keyvault secret show --name ${valutKeyName} --vault-name ${vaultName}`);
            if (result.stderr) {
                const errorMessage = result.stderr;
                this.log.error(errorMessage);
                return Promise.reject(errorMessage);
            }
            const storageAccountKey = JSON.parse(result.stdout).value;
            if (this.azureStorageAccountName === undefined) {
                throw new Error('azureStorageAccountName not initialized!');
            }
            this.azureStorageClient = azure_storage_1.default.createFileService(this.azureStorageAccountName, storageAccountKey);
            await azureStorageClientUtils_1.AzureStorageClientUtility.createShare(this.azureStorageClient, this.azureStorageShare);
            this.azureStorageSecretName = typescript_string_operations_1.String.Format('nni-secret-{0}', utils_1.uniqueString(8)
                .toLowerCase());
            if (this.genericK8sClient === undefined) {
                throw new Error("genericK8sClient undefined!");
            }
            const namespace = this.genericK8sClient.getNamespace ?? "default";
            await this.genericK8sClient.createSecret({
                apiVersion: 'v1',
                kind: 'Secret',
                metadata: {
                    name: this.azureStorageSecretName,
                    namespace: namespace,
                    labels: {
                        app: this.NNI_KUBERNETES_TRIAL_LABEL,
                        expId: this.experimentId
                    }
                },
                type: 'Opaque',
                data: {
                    azurestorageaccountname: js_base64_1.Base64.encode(this.azureStorageAccountName),
                    azurestorageaccountkey: js_base64_1.Base64.encode(storageAccountKey)
                }
            });
        }
        catch (error) {
            this.log.error(error);
            return Promise.reject(error);
        }
        return Promise.resolve();
    }
    async uploadFolderToAzureStorage(srcDirectory, destDirectory, uploadRetryCount) {
        if (this.azureStorageClient === undefined) {
            throw new Error('azureStorageClient is not initialized');
        }
        let retryCount = 1;
        if (uploadRetryCount) {
            retryCount = uploadRetryCount;
        }
        let uploadSuccess = false;
        let folderUriInAzure = '';
        try {
            do {
                uploadSuccess = await azureStorageClientUtils_1.AzureStorageClientUtility.uploadDirectory(this.azureStorageClient, `${destDirectory}`, this.azureStorageShare, `${srcDirectory}`);
                if (!uploadSuccess) {
                    await utils_1.delay(5000);
                    this.log.info('Upload failed, Retry: upload files to azure-storage');
                }
                else {
                    folderUriInAzure = `https://${this.azureStorageAccountName}.file.core.windows.net/${this.azureStorageShare}/${destDirectory}`;
                    break;
                }
            } while (retryCount-- >= 0);
        }
        catch (error) {
            this.log.error(error);
            return Promise.resolve('');
        }
        return Promise.resolve(folderUriInAzure);
    }
    async createNFSStorage(nfsServer, nfsPath) {
        await child_process_promise_1.default.exec(`mkdir -p ${this.environmentLocalTempFolder}`);
        try {
            await child_process_promise_1.default.exec(`sudo mount ${nfsServer}:${nfsPath} ${this.environmentLocalTempFolder}`);
        }
        catch (error) {
            const mountError = `Mount NFS ${nfsServer}:${nfsPath} to ${this.environmentLocalTempFolder} failed, error is ${error}`;
            this.log.error(mountError);
            return Promise.reject(mountError);
        }
        return Promise.resolve();
    }
    async createPVCStorage(pvcPath) {
        try {
            await child_process_promise_1.default.exec(`mkdir -p ${pvcPath}`);
            await child_process_promise_1.default.exec(`sudo ln -s ${pvcPath} ${this.environmentLocalTempFolder}`);
        }
        catch (error) {
            const linkError = `Linking ${pvcPath} to ${this.environmentLocalTempFolder} failed, error is ${error}`;
            this.log.error(linkError);
            return Promise.reject(linkError);
        }
        return Promise.resolve();
    }
    async createRegistrySecret(filePath) {
        if (filePath === undefined || filePath === '') {
            return undefined;
        }
        const body = fs.readFileSync(filePath).toString('base64');
        const registrySecretName = typescript_string_operations_1.String.Format('nni-secret-{0}', utils_1.uniqueString(8)
            .toLowerCase());
        const namespace = this.genericK8sClient.getNamespace ?? "default";
        await this.genericK8sClient.createSecret({
            apiVersion: 'v1',
            kind: 'Secret',
            metadata: {
                name: registrySecretName,
                namespace: namespace,
                labels: {
                    app: this.NNI_KUBERNETES_TRIAL_LABEL,
                    expId: this.experimentId
                }
            },
            type: 'kubernetes.io/dockerconfigjson',
            data: {
                '.dockerconfigjson': body
            }
        });
        return registrySecretName;
    }
    async refreshEnvironmentsStatus(environments) {
        environments.forEach(async (environment) => {
            if (this.kubernetesCRDClient === undefined) {
                throw new Error("kubernetesCRDClient undefined");
            }
            const kubeflowJobName = `nniexp${this.experimentId}env${environment.id}`.toLowerCase();
            const kubernetesJobInfo = await this.kubernetesCRDClient.getKubernetesJob(kubeflowJobName);
            if (kubernetesJobInfo.status && kubernetesJobInfo.status.conditions) {
                const latestCondition = kubernetesJobInfo.status.conditions[kubernetesJobInfo.status.conditions.length - 1];
                const tfJobType = latestCondition.type;
                switch (tfJobType) {
                    case 'Created':
                        environment.setStatus('WAITING');
                        break;
                    case 'Running':
                        environment.setStatus('RUNNING');
                        break;
                    case 'Failed':
                        environment.setStatus('FAILED');
                        break;
                    case 'Succeeded':
                        environment.setStatus('SUCCEEDED');
                        break;
                    default:
                }
            }
        });
    }
    async startEnvironment(_environment) {
        throw new Error("Not implemented");
    }
    async stopEnvironment(environment) {
        if (this.kubernetesCRDClient === undefined) {
            throw new Error('kubernetesCRDClient not initialized!');
        }
        try {
            await this.kubernetesCRDClient.deleteKubernetesJob(new Map([
                ['app', this.NNI_KUBERNETES_TRIAL_LABEL],
                ['expId', this.experimentId],
                ['envId', environment.id]
            ]));
        }
        catch (err) {
            const errorMessage = `Delete env ${environment.id} failed: ${err}`;
            this.log.error(errorMessage);
            return Promise.reject(errorMessage);
        }
    }
    generatePodResource(memory, cpuNum, gpuNum) {
        const resources = {
            memory: `${memory}Mi`,
            cpu: `${cpuNum}`
        };
        if (gpuNum !== 0) {
            resources['nvidia.com/gpu'] = `${gpuNum}`;
        }
        return resources;
    }
}
exports.KubernetesEnvironmentService = KubernetesEnvironmentService;
