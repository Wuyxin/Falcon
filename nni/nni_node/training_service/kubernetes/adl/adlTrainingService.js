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
exports.AdlTrainingService = void 0;
const fs_1 = __importDefault(require("fs"));
const component = __importStar(require("common/component"));
const typescript_string_operations_1 = require("typescript-string-operations");
const experimentStartupInfo_1 = require("common/experimentStartupInfo");
const utils_1 = require("common/utils");
const trialConfigMetadataKey_1 = require("training_service/common/trialConfigMetadataKey");
const kubernetesData_1 = require("../kubernetesData");
const kubernetesTrainingService_1 = require("../kubernetesTrainingService");
const adlApiClient_1 = require("./adlApiClient");
const adlJobInfoCollector_1 = require("./adlJobInfoCollector");
const adlJobRestServer_1 = require("./adlJobRestServer");
let AdlTrainingService = class AdlTrainingService extends kubernetesTrainingService_1.KubernetesTrainingService {
    adlTrialConfig;
    adlJobInfoCollector;
    configmapTemplateStr;
    jobTemplateStr;
    pvcTemplateStr;
    tensorboardPvcTemplate;
    tensorboardDeploymentTemplate;
    tensorboardName = "adaptdl-tensorboard-" + experimentStartupInfo_1.getExperimentId().toLowerCase();
    constructor() {
        super();
        this.adlJobInfoCollector = new adlJobInfoCollector_1.AdlJobInfoCollector(this.trialJobsMap);
        this.experimentId = experimentStartupInfo_1.getExperimentId();
        this.configmapTemplateStr = fs_1.default.readFileSync('./config/adl/adaptdl-nni-configmap-template.json', 'utf8');
        this.jobTemplateStr = fs_1.default.readFileSync('./config/adl/adaptdljob-template.json', 'utf8');
        this.pvcTemplateStr = fs_1.default.readFileSync('./config/adl/adaptdl-pvc-template.json', 'utf8');
        this.tensorboardPvcTemplate = JSON.parse(fs_1.default.readFileSync('./config/adl/adaptdl-tensorboard-pvc-template.json', 'utf8'));
        this.tensorboardDeploymentTemplate = JSON.parse(fs_1.default.readFileSync('./config/adl/adaptdl-tensorboard-deployment-template.json', 'utf8'));
        this.log.info('Construct Adl training service.');
    }
    async run() {
        this.log.info(this.tensorboardName);
        this.log.info('Start tensorboard deployment.');
        await this.launchTensorboard();
        this.log.info('Run Adl training service.');
        this.kubernetesJobRestServer = component.get(adlJobRestServer_1.AdlJobRestServer);
        if (this.kubernetesJobRestServer === undefined) {
            throw new Error('kubernetesJobRestServer not initialized!');
        }
        await this.kubernetesJobRestServer.start();
        this.kubernetesJobRestServer.setEnableVersionCheck = this.versionCheck;
        this.log.info(`Adl Training service rest server listening on: ${this.kubernetesJobRestServer.endPoint}`);
        while (!this.stopping) {
            await utils_1.delay(3000);
            await this.adlJobInfoCollector.retrieveTrialStatus(this.kubernetesCRDClient);
            if (this.kubernetesJobRestServer.getErrorMessage !== undefined) {
                throw new Error(this.kubernetesJobRestServer.getErrorMessage);
            }
        }
        this.log.info('Adl training service exit.');
    }
    async launchTensorboard() {
        if (this.adlTrialConfig === undefined) {
            throw new Error('Adl trial config is undefined');
        }
        this.tensorboardDeploymentTemplate.metadata.name = this.tensorboardName;
        this.tensorboardDeploymentTemplate.metadata.labels.expId = this.experimentId;
        this.tensorboardDeploymentTemplate.spec.selector.matchLabels.app = this.tensorboardName;
        this.tensorboardDeploymentTemplate.spec.template.metadata.labels.app = this.tensorboardName;
        this.tensorboardDeploymentTemplate.spec.template.spec.volumes[0]
            .persistentVolumeClaim.claimName = this.tensorboardName;
        const deploymentUid = await this.genericK8sClient.createDeployment(this.tensorboardDeploymentTemplate);
        this.tensorboardPvcTemplate.metadata.name = this.tensorboardName;
        this.tensorboardPvcTemplate.metadata.ownerReferences[0].name = this.tensorboardName;
        this.tensorboardPvcTemplate.metadata.ownerReferences[0].uid = deploymentUid;
        if (this.adlTrialConfig.checkpoint != undefined) {
            this.tensorboardPvcTemplate.spec.resources.requests.storage = this.adlTrialConfig.checkpoint.storageSize;
            this.tensorboardPvcTemplate.spec.storageClassName = this.adlTrialConfig.checkpoint.storageClass;
        }
        else {
            this.tensorboardPvcTemplate.spec.resources.requests.storage = "1Gi";
            this.tensorboardPvcTemplate.spec.storageClassName = await this.genericK8sClient.getStorageClass();
        }
        await this.genericK8sClient.createPersistentVolumeClaim(this.tensorboardPvcTemplate);
        return Promise.resolve();
    }
    async submitTrialJob(form) {
        if (this.kubernetesCRDClient === undefined) {
            throw new Error('Adl job operator client is undefined');
        }
        if (this.adlTrialConfig === undefined) {
            throw new Error('Adl trial config is undefined');
        }
        if (this.kubernetesRestServerPort === undefined) {
            const restServer = component.get(adlJobRestServer_1.AdlJobRestServer);
            this.kubernetesRestServerPort = restServer.clusterRestServerPort;
        }
        const trialJobId = utils_1.uniqueString(5);
        const adlJobName = `nni-exp-${this.experimentId}-trial-${trialJobId}`.toLowerCase();
        const initStatus = 'WAITING';
        const codeDir = this.adlTrialConfig.codeDir;
        const outputDir = "output";
        const trialJobDetail = new kubernetesData_1.KubernetesTrialJobDetail(trialJobId, initStatus, Date.now(), codeDir, form, adlJobName, outputDir);
        const job = JSON.parse(this.jobTemplateStr);
        job.metadata.name = adlJobName;
        job.metadata.labels.app = this.NNI_KUBERNETES_TRIAL_LABEL;
        job.metadata.labels.expId = this.experimentId;
        job.metadata.labels.trialId = trialJobId;
        if (this.adlTrialConfig.adaptive !== undefined) {
            job.spec.preemptible = this.adlTrialConfig.adaptive;
        }
        job.spec.template.spec.containers[0]
            .image = this.adlTrialConfig.image;
        job.spec.template.spec.volumes[0]
            .persistentVolumeClaim.claimName = adlJobName;
        job.spec.template.spec.volumes[1]
            .persistentVolumeClaim.claimName = this.tensorboardName;
        job.spec.template.spec.volumes[2]
            .configMap.name = adlJobName;
        let cpu = 1;
        let memory = "1Gi";
        if (this.adlTrialConfig.cpuNum !== undefined) {
            cpu = this.adlTrialConfig.cpuNum;
        }
        if (this.adlTrialConfig.memorySize !== undefined) {
            memory = this.adlTrialConfig.memorySize;
        }
        job.spec.template.spec.containers[0]
            .resources.requests.memory = memory;
        job.spec.template.spec.containers[0]
            .resources.requests.cpu = cpu;
        job.spec.template.spec.containers[0]
            .resources.limits["nvidia.com/gpu"] = this.adlTrialConfig.gpuNum;
        if (this.adlTrialConfig.imagePullSecrets !== undefined) {
            job.spec.template.spec.imagePullSecrets = job.spec.template.spec
                .imagePullSecrets.concat(this.adlTrialConfig.imagePullSecrets);
        }
        if (this.adlTrialConfig.nfs !== undefined) {
            job.spec.template.spec.volumes.push({
                "name": "nfs",
                "nfs": {
                    "server": this.adlTrialConfig.nfs.server,
                    "path": this.adlTrialConfig.nfs.path,
                    "readOnly": false
                }
            });
            job.spec.template.spec.containers[0].volumeMounts.push({
                "name": "nfs",
                "mountPath": this.adlTrialConfig.nfs.containerMountPath
            });
        }
        await this.kubernetesCRDClient.createKubernetesJob(job);
        const k8sadlJob = await this.kubernetesCRDClient.getKubernetesJob(adlJobName);
        const pvc = JSON.parse(this.pvcTemplateStr);
        pvc.metadata.name = adlJobName;
        pvc.metadata.ownerReferences[0].name = adlJobName;
        pvc.metadata.ownerReferences[0].uid = k8sadlJob.metadata.uid;
        if (this.adlTrialConfig.checkpoint != undefined) {
            pvc.spec.resources.requests.storage = this.adlTrialConfig
                .checkpoint.storageSize;
            pvc.spec.storageClassName = this.adlTrialConfig.checkpoint.storageClass;
        }
        else {
            pvc.spec.resources.requests.storage = "1Gi";
            pvc.spec.storageClassName = await this.genericK8sClient.getStorageClass();
        }
        await this.genericK8sClient.createPersistentVolumeClaim(pvc);
        const configmap = JSON.parse(this.configmapTemplateStr);
        configmap.metadata.name = adlJobName;
        configmap.metadata.ownerReferences[0].name = adlJobName;
        configmap.metadata.ownerReferences[0].uid = k8sadlJob.metadata.uid;
        configmap.data["run.sh"] = await this.prepareRunScript(trialJobId, form, codeDir, outputDir);
        const cleanupScriptTemplate = `#!/bin/bash
ps aux | grep "python3 -m nni.tools.trial_tool.trial_keeper" | awk '{print $2}' | xargs kill -2
while true;
do
    proc=\`ps aux | grep "python3 -m nni.tools.trial_tool.trial_keeper" | awk '{print $2}' | grep "" -c\`
    if (( $proc == 1  )); then
        exit 0
    else
        echo "waiting"
    fi
    sleep 1
done
`;
        configmap.data["cleanup.sh"] = cleanupScriptTemplate;
        await this.genericK8sClient.createConfigMap(configmap);
        this.trialJobsMap.set(trialJobId, trialJobDetail);
        return Promise.resolve(trialJobDetail);
    }
    async prepareRunScript(jobId, form, codeDir, outputDir) {
        if (this.adlTrialConfig === undefined) {
            throw new Error('Adl trial config is undefined');
        }
        if (this.kubernetesRestServerPort === undefined) {
            throw new Error('Adl rest server port is undefined');
        }
        if (this.nniManagerIpConfig === undefined) {
            throw new Error('Adl nniManager ip config is undefined');
        }
        const expId = this.experimentId;
        const seqId = form.sequenceId.toString();
        const command = this.adlTrialConfig.command;
        const hyperParameters = form.hyperParameters.value;
        const hyperParametersFile = utils_1.generateParamFileName(form.hyperParameters);
        const nniManagerPort = this.kubernetesRestServerPort.toString();
        const nniManagerIp = this.nniManagerIpConfig.nniManagerIp;
        let nniManagerVersion = '';
        if (this.versionCheck) {
            nniManagerVersion = await utils_1.getVersion();
        }
        let nvidiaScript = '';
        if (this.adlTrialConfig.gpuNum == 0) {
            nvidiaScript = 'export CUDA_VISIBLE_DEVICES=';
        }
        const runScriptTemplate = `#!/bin/bash
export NNI_PLATFORM=adl
export MULTI_PHASE=false
export NNI_SYS_DIR={0}
export NNI_CODE_DIR={0}
export NNI_OUTPUT_DIR={1}
export NNI_TRIAL_JOB_ID={2}
export NNI_EXP_ID={3}
export NNI_TRIAL_SEQ_ID={4}
mkdir -p $NNI_OUTPUT_DIR
{5}
echo '{6}' > $NNI_CODE_DIR/{7}
python3 -m nni.tools.trial_tool.trial_keeper --trial_command '{8}' \
--nnimanager_ip {9} --nnimanager_port {10} \
--nni_manager_version '{11}' --log_collection '{12}'
`;
        const runScript = typescript_string_operations_1.String.Format(runScriptTemplate, codeDir, outputDir, jobId, expId, seqId, nvidiaScript, hyperParameters, hyperParametersFile, command, nniManagerIp, nniManagerPort, nniManagerVersion, this.logCollection);
        return Promise.resolve(runScript);
    }
    async cleanUp() {
        super.cleanUp();
        try {
            await this.genericK8sClient.deleteDeployment("adaptdl-tensorboard-" + this.experimentId.toLowerCase());
            this.log.info('tensorboard deployment deleted');
        }
        catch (error) {
            this.log.error(`tensorboard deployment deletion failed: ${error.message}`);
        }
    }
    async setClusterMetadata(key, value) {
        this.log.info('SetCluster ' + key + ', ' + value);
        switch (key) {
            case trialConfigMetadataKey_1.TrialConfigMetadataKey.NNI_MANAGER_IP:
                this.nniManagerIpConfig = JSON.parse(value);
                break;
            case trialConfigMetadataKey_1.TrialConfigMetadataKey.TRIAL_CONFIG: {
                this.adlTrialConfig = JSON.parse(value);
                let namespace = 'default';
                if (this.adlTrialConfig.namespace !== undefined) {
                    namespace = this.adlTrialConfig.namespace;
                }
                this.genericK8sClient.setNamespace = namespace;
                this.kubernetesCRDClient = adlApiClient_1.AdlClientFactory.createClient(namespace);
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
    getClusterMetadata(key) {
        let result;
        switch (key) {
            case trialConfigMetadataKey_1.TrialConfigMetadataKey.TRIAL_CONFIG:
                if (this.adlTrialConfig === undefined) {
                    return Promise.reject(`${key} is not set yet`);
                }
                result = JSON.stringify(this.adlTrialConfig);
                break;
            case trialConfigMetadataKey_1.TrialConfigMetadataKey.NNI_MANAGER_IP:
                if (this.nniManagerIpConfig === undefined) {
                    return Promise.reject(`${key} is not set yet`);
                }
                result = JSON.stringify(this.nniManagerIpConfig);
                break;
            default:
                return Promise.reject(`${key} not set`);
        }
        return Promise.resolve(result);
    }
    async updateTrialJob(_1, _2) {
        throw new Error('not supported');
    }
};
AdlTrainingService = __decorate([
    component.Singleton,
    __metadata("design:paramtypes", [])
], AdlTrainingService);
exports.AdlTrainingService = AdlTrainingService;
