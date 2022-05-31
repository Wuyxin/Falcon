"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.NNIPAITrialConfig = exports.PAI_TRIAL_COMMAND_FORMAT = exports.PAITrialJobDetail = exports.PAIClusterConfig = void 0;
const trialConfig_1 = require("../common/trialConfig");
class PAIClusterConfig {
    userName;
    passWord;
    host;
    token;
    reuse;
    cpuNum;
    memoryMB;
    gpuNum;
    useActiveGpu;
    maxTrialNumPerGpu;
    constructor(userName, host, passWord, token, reuse, cpuNum, memoryMB, gpuNum) {
        this.userName = userName;
        this.passWord = passWord;
        this.host = host;
        this.token = token;
        this.reuse = reuse;
        this.cpuNum = cpuNum;
        this.memoryMB = memoryMB;
        this.gpuNum = gpuNum;
    }
}
exports.PAIClusterConfig = PAIClusterConfig;
class PAITrialJobDetail {
    id;
    status;
    paiJobName;
    submitTime;
    startTime;
    endTime;
    tags;
    url;
    workingDirectory;
    form;
    logPath;
    isEarlyStopped;
    paiJobDetailUrl;
    constructor(id, status, paiJobName, submitTime, workingDirectory, form, logPath, paiJobDetailUrl) {
        this.id = id;
        this.status = status;
        this.paiJobName = paiJobName;
        this.submitTime = submitTime;
        this.workingDirectory = workingDirectory;
        this.form = form;
        this.tags = [];
        this.logPath = logPath;
        this.paiJobDetailUrl = paiJobDetailUrl;
    }
}
exports.PAITrialJobDetail = PAITrialJobDetail;
exports.PAI_TRIAL_COMMAND_FORMAT = `export NNI_PLATFORM=pai NNI_SYS_DIR={0} NNI_OUTPUT_DIR={1} NNI_TRIAL_JOB_ID={2} NNI_EXP_ID={3} NNI_TRIAL_SEQ_ID={4} MULTI_PHASE={5} \
&& NNI_CODE_DIR={6} && mkdir -p $NNI_SYS_DIR/code && cp -r $NNI_CODE_DIR/. $NNI_SYS_DIR/code && sh $NNI_SYS_DIR/install_nni.sh \
&& cd $NNI_SYS_DIR/code && python3 -m nni.tools.trial_tool.trial_keeper --trial_command '{7}' --nnimanager_ip '{8}' --nnimanager_port '{9}' \
--nni_manager_version '{10}' --log_collection '{11}' | tee $NNI_OUTPUT_DIR/trial.log`;
class NNIPAITrialConfig extends trialConfig_1.TrialConfig {
    cpuNum;
    memoryMB;
    image;
    virtualCluster;
    nniManagerNFSMountPath;
    containerNFSMountPath;
    paiStorageConfigName;
    paiConfigPath;
    constructor(command, codeDir, gpuNum, cpuNum, memoryMB, image, nniManagerNFSMountPath, containerNFSMountPath, paiStorageConfigName, virtualCluster, paiConfigPath) {
        super(command, codeDir, gpuNum);
        this.cpuNum = cpuNum;
        this.memoryMB = memoryMB;
        this.image = image;
        this.virtualCluster = virtualCluster;
        this.nniManagerNFSMountPath = nniManagerNFSMountPath;
        this.containerNFSMountPath = containerNFSMountPath;
        this.paiStorageConfigName = paiStorageConfigName;
        this.paiConfigPath = paiConfigPath;
    }
}
exports.NNIPAITrialConfig = NNIPAITrialConfig;
