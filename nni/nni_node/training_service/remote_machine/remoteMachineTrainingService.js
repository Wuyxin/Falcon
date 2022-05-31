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
exports.RemoteMachineTrainingService = void 0;
const assert_1 = __importDefault(require("assert"));
const events_1 = require("events");
const fs_1 = __importDefault(require("fs"));
const path_1 = __importDefault(require("path"));
const ts_deferred_1 = require("ts-deferred");
const component = __importStar(require("common/component"));
const errors_1 = require("common/errors");
const experimentStartupInfo_1 = require("common/experimentStartupInfo");
const log_1 = require("common/log");
const observableTimer_1 = require("common/observableTimer");
const utils_1 = require("common/utils");
const containerJobData_1 = require("../common/containerJobData");
const gpuData_1 = require("../common/gpuData");
const util_1 = require("../common/util");
const gpuScheduler_1 = require("./gpuScheduler");
const remoteMachineData_1 = require("./remoteMachineData");
const remoteMachineJobRestServer_1 = require("./remoteMachineJobRestServer");
let RemoteMachineTrainingService = class RemoteMachineTrainingService {
    initExecutorId = "initConnection";
    machineExecutorManagerMap;
    machineCopyExpCodeDirPromiseMap;
    trialExecutorManagerMap;
    trialJobsMap;
    expRootDir;
    gpuScheduler;
    jobQueue;
    timer;
    stopping = false;
    metricsEmitter;
    log;
    remoteRestServerPort;
    versionCheck = true;
    logCollection = 'none';
    sshConnectionPromises;
    config;
    constructor(config) {
        this.metricsEmitter = new events_1.EventEmitter();
        this.trialJobsMap = new Map();
        this.trialExecutorManagerMap = new Map();
        this.machineCopyExpCodeDirPromiseMap = new Map();
        this.machineExecutorManagerMap = new Map();
        this.jobQueue = [];
        this.sshConnectionPromises = [];
        this.expRootDir = utils_1.getExperimentRootDir();
        this.timer = component.get(observableTimer_1.ObservableTimer);
        this.log = log_1.getLogger('RemoteMachineTrainingService');
        this.log.info('Construct remote machine training service.');
        this.config = config;
        if (!fs_1.default.lstatSync(this.config.trialCodeDirectory).isDirectory()) {
            throw new Error(`codeDir ${this.config.trialCodeDirectory} is not a directory`);
        }
        util_1.validateCodeDir(this.config.trialCodeDirectory);
        this.sshConnectionPromises = this.config.machineList.map(machine => this.initRemoteMachineOnConnected(machine));
    }
    async run() {
        const restServer = new remoteMachineJobRestServer_1.RemoteMachineJobRestServer(this);
        await restServer.start();
        restServer.setEnableVersionCheck = this.versionCheck;
        this.log.info('Run remote machine training service.');
        if (this.sshConnectionPromises.length > 0) {
            await Promise.all(this.sshConnectionPromises);
            this.log.info('ssh connection initialized!');
            this.sshConnectionPromises = [];
            this.gpuScheduler = new gpuScheduler_1.GPUScheduler(this.machineExecutorManagerMap);
            for (const [machineConfig, executorManager] of this.machineExecutorManagerMap.entries()) {
                const executor = await executorManager.getExecutor(this.initExecutorId);
                if (executor !== undefined) {
                    this.machineCopyExpCodeDirPromiseMap.set(machineConfig, executor.copyDirectoryToRemote(this.config.trialCodeDirectory, executor.getRemoteCodePath(experimentStartupInfo_1.getExperimentId())));
                }
            }
        }
        while (!this.stopping) {
            while (this.jobQueue.length > 0) {
                this.updateGpuReservation();
                const trialJobId = this.jobQueue[0];
                const prepareResult = await this.prepareTrialJob(trialJobId);
                if (prepareResult) {
                    this.jobQueue.shift();
                }
                else {
                    break;
                }
            }
            if (restServer.getErrorMessage !== undefined) {
                this.stopping = true;
                throw new Error(restServer.getErrorMessage);
            }
            await utils_1.delay(3000);
        }
        this.log.info('RemoteMachineTrainingService run loop exited.');
    }
    allocateExecutorManagerForTrial(trial) {
        if (trial.rmMeta === undefined) {
            throw new Error(`rmMeta not set in trial ${trial.id}`);
        }
        const executorManager = this.machineExecutorManagerMap.get(trial.rmMeta.config);
        if (executorManager === undefined) {
            throw new Error(`executorManager not initialized`);
        }
        this.trialExecutorManagerMap.set(trial.id, executorManager);
    }
    releaseTrialResource(trial) {
        if (trial.rmMeta === undefined) {
            throw new Error(`rmMeta not set in trial ${trial.id}`);
        }
        const executorManager = this.trialExecutorManagerMap.get(trial.id);
        if (executorManager === undefined) {
            throw new Error(`ExecutorManager is not assigned for trial ${trial.id}`);
        }
        executorManager.releaseExecutor(trial.id);
    }
    async listTrialJobs() {
        const jobs = [];
        const deferred = new ts_deferred_1.Deferred();
        for (const [key,] of this.trialJobsMap) {
            jobs.push(await this.getTrialJob(key));
        }
        deferred.resolve(jobs);
        return deferred.promise;
    }
    async getTrialJob(trialJobId) {
        const trialJob = this.trialJobsMap.get(trialJobId);
        if (trialJob === undefined) {
            throw new errors_1.NNIError(errors_1.NNIErrorNames.NOT_FOUND, `trial job id ${trialJobId} not found`);
        }
        if (trialJob.status === 'RUNNING' || trialJob.status === 'UNKNOWN') {
            if (trialJob.rmMeta === undefined) {
                throw new Error(`rmMeta not set for submitted job ${trialJobId}`);
            }
            const executor = await this.getExecutor(trialJob.id);
            return this.updateTrialJobStatus(trialJob, executor);
        }
        else {
            return trialJob;
        }
    }
    async getTrialFile(_trialJobId, _fileName) {
        throw new errors_1.MethodNotImplementedError();
    }
    addTrialJobMetricListener(listener) {
        this.metricsEmitter.on('metric', listener);
    }
    removeTrialJobMetricListener(listener) {
        this.metricsEmitter.off('metric', listener);
    }
    async submitTrialJob(form) {
        const trialJobId = utils_1.uniqueString(5);
        const trialJobDetail = new remoteMachineData_1.RemoteMachineTrialJobDetail(trialJobId, 'WAITING', Date.now(), "unset", form);
        this.jobQueue.push(trialJobId);
        this.trialJobsMap.set(trialJobId, trialJobDetail);
        return Promise.resolve(trialJobDetail);
    }
    async updateTrialJob(trialJobId, form) {
        const trialJobDetail = this.trialJobsMap.get(trialJobId);
        if (trialJobDetail === undefined) {
            throw new Error(`updateTrialJob failed: ${trialJobId} not found`);
        }
        await this.writeParameterFile(trialJobId, form.hyperParameters);
        return trialJobDetail;
    }
    async cancelTrialJob(trialJobId, isEarlyStopped = false) {
        const trialJob = this.trialJobsMap.get(trialJobId);
        if (trialJob === undefined) {
            throw new Error(`trial job id ${trialJobId} not found`);
        }
        const index = this.jobQueue.indexOf(trialJobId);
        if (index >= 0) {
            this.jobQueue.splice(index, 1);
        }
        if (trialJob.rmMeta !== undefined) {
            const executor = await this.getExecutor(trialJob.id);
            if (trialJob.status === 'UNKNOWN') {
                trialJob.status = 'USER_CANCELED';
                this.releaseTrialResource(trialJob);
                return;
            }
            const jobpidPath = this.getJobPidPath(executor, trialJob.id);
            try {
                trialJob.isEarlyStopped = isEarlyStopped;
                await executor.killChildProcesses(jobpidPath);
                this.releaseTrialResource(trialJob);
            }
            catch (error) {
                this.log.error(`remoteTrainingService.cancelTrialJob: ${error}`);
            }
        }
        else {
            assert_1.default(isEarlyStopped === false, 'isEarlyStopped is not supposed to be true here.');
            trialJob.status = utils_1.getJobCancelStatus(isEarlyStopped);
        }
    }
    async setClusterMetadata(_key, _value) { return; }
    async getClusterMetadata(_key) { return ''; }
    async cleanUp() {
        this.log.info('Stopping remote machine training service...');
        this.stopping = true;
        await this.cleanupConnections();
    }
    async getExecutor(trialId) {
        const executorManager = this.trialExecutorManagerMap.get(trialId);
        if (executorManager === undefined) {
            throw new Error(`ExecutorManager is not assigned for trial ${trialId}`);
        }
        return await executorManager.getExecutor(trialId);
    }
    updateGpuReservation() {
        if (this.gpuScheduler) {
            for (const [key, value] of this.trialJobsMap) {
                if (!['WAITING', 'RUNNING'].includes(value.status)) {
                    this.gpuScheduler.removeGpuReservation(key, this.trialJobsMap);
                }
            }
        }
    }
    async cleanupConnections() {
        try {
            for (const executorManager of this.machineExecutorManagerMap.values()) {
                const executor = await executorManager.getExecutor(this.initExecutorId);
                if (executor !== undefined) {
                    this.log.info(`killing gpu metric collector on ${executor.name}`);
                    const gpuJobPidPath = executor.joinPath(executor.getRemoteScriptsPath(experimentStartupInfo_1.getExperimentId()), 'pid');
                    await executor.killChildProcesses(gpuJobPidPath, true);
                }
                executorManager.releaseAllExecutor();
            }
        }
        catch (error) {
            this.log.error(`Cleanup connection exception, error is ${error}`);
        }
    }
    async initRemoteMachineOnConnected(machineConfig) {
        const executorManager = new remoteMachineData_1.ExecutorManager(machineConfig);
        this.log.info(`connecting to ${machineConfig.user}@${machineConfig.host}:${machineConfig.port}`);
        const executor = await executorManager.getExecutor(this.initExecutorId);
        this.log.debug(`reached ${executor.name}`);
        this.machineExecutorManagerMap.set(machineConfig, executorManager);
        this.log.debug(`initializing ${executor.name}`);
        const nniRootDir = executor.joinPath(executor.getTempPath(), 'nni');
        await executor.createFolder(executor.getRemoteExperimentRootDir(experimentStartupInfo_1.getExperimentId()));
        const remoteGpuScriptCollectorDir = executor.getRemoteScriptsPath(experimentStartupInfo_1.getExperimentId());
        await executor.createFolder(remoteGpuScriptCollectorDir, true);
        await executor.allowPermission(true, nniRootDir);
        const script = executor.generateGpuStatsScript(experimentStartupInfo_1.getExperimentId());
        executor.executeScript(script, false, true);
        const collectingCount = [];
        const disposable = this.timer.subscribe(async () => {
            if (collectingCount.length == 0) {
                collectingCount.push(true);
                const cmdresult = await executor.readLastLines(executor.joinPath(remoteGpuScriptCollectorDir, 'gpu_metrics'));
                if (cmdresult !== "") {
                    executorManager.rmMeta.gpuSummary = JSON.parse(cmdresult);
                    if (executorManager.rmMeta.gpuSummary.gpuCount === 0) {
                        this.log.warning(`No GPU found on remote machine ${machineConfig.host}`);
                        this.timer.unsubscribe(disposable);
                    }
                }
                if (this.stopping) {
                    this.timer.unsubscribe(disposable);
                    this.log.debug(`Stopped GPU collector on ${machineConfig.host}, since experiment is exiting.`);
                }
                collectingCount.pop();
            }
        });
    }
    async prepareTrialJob(trialJobId) {
        const deferred = new ts_deferred_1.Deferred();
        if (this.gpuScheduler === undefined) {
            throw new Error('gpuScheduler is not initialized');
        }
        const trialJobDetail = this.trialJobsMap.get(trialJobId);
        if (trialJobDetail === undefined) {
            throw new errors_1.NNIError(errors_1.NNIErrorNames.INVALID_JOB_DETAIL, `Invalid job detail information for trial job ${trialJobId}`);
        }
        if (trialJobDetail.status !== 'WAITING') {
            deferred.resolve(true);
            return deferred.promise;
        }
        const rmScheduleResult = this.gpuScheduler.scheduleMachine(this.config.trialGpuNumber, trialJobDetail);
        if (rmScheduleResult.resultType === gpuData_1.ScheduleResultType.REQUIRE_EXCEED_TOTAL) {
            const errorMessage = `Required GPU number ${this.config.trialGpuNumber} is too large, no machine can meet`;
            this.log.error(errorMessage);
            deferred.reject();
            throw new errors_1.NNIError(errors_1.NNIErrorNames.RESOURCE_NOT_AVAILABLE, errorMessage);
        }
        else if (rmScheduleResult.resultType === gpuData_1.ScheduleResultType.SUCCEED
            && rmScheduleResult.scheduleInfo !== undefined) {
            const rmScheduleInfo = rmScheduleResult.scheduleInfo;
            trialJobDetail.rmMeta = rmScheduleInfo.rmMeta;
            const copyExpCodeDirPromise = this.machineCopyExpCodeDirPromiseMap.get(rmScheduleInfo.rmMeta.config);
            if (copyExpCodeDirPromise !== undefined) {
                await copyExpCodeDirPromise;
            }
            this.allocateExecutorManagerForTrial(trialJobDetail);
            const executor = await this.getExecutor(trialJobDetail.id);
            trialJobDetail.workingDirectory = executor.joinPath(executor.getRemoteExperimentRootDir(experimentStartupInfo_1.getExperimentId()), 'trials', trialJobDetail.id);
            await this.launchTrialOnScheduledMachine(trialJobId, trialJobDetail.form, rmScheduleInfo);
            trialJobDetail.status = 'RUNNING';
            trialJobDetail.url = `file://${rmScheduleInfo.rmMeta.config.host}:${trialJobDetail.workingDirectory}`;
            trialJobDetail.startTime = Date.now();
            this.trialJobsMap.set(trialJobId, trialJobDetail);
            deferred.resolve(true);
        }
        else if (rmScheduleResult.resultType === gpuData_1.ScheduleResultType.TMP_NO_AVAILABLE_GPU) {
            this.log.info(`Right now no available GPU can be allocated for trial ${trialJobId}, will try to schedule later`);
            deferred.resolve(false);
        }
        else {
            deferred.reject(`Invalid schedule resutl type: ${rmScheduleResult.resultType}`);
        }
        return deferred.promise;
    }
    async launchTrialOnScheduledMachine(trialJobId, form, rmScheduleInfo) {
        const cudaVisibleDevice = rmScheduleInfo.cudaVisibleDevice;
        const executor = await this.getExecutor(trialJobId);
        const trialJobDetail = this.trialJobsMap.get(trialJobId);
        if (trialJobDetail === undefined) {
            throw new Error(`Can not get trial job detail for job: ${trialJobId}`);
        }
        const trialLocalTempFolder = path_1.default.join(this.expRootDir, 'trials', trialJobId);
        await executor.createFolder(executor.joinPath(trialJobDetail.workingDirectory, '.nni'));
        let cudaVisible;
        if (this.config.trialGpuNumber === undefined) {
            cudaVisible = "";
        }
        else {
            if (typeof cudaVisibleDevice === 'string' && cudaVisibleDevice.length > 0) {
                cudaVisible = `CUDA_VISIBLE_DEVICES=${cudaVisibleDevice}`;
            }
            else {
                cudaVisible = `CUDA_VISIBLE_DEVICES=" "`;
            }
        }
        const nniManagerIp = this.config.nniManagerIp ? this.config.nniManagerIp : await utils_1.getIPV4Address();
        if (this.remoteRestServerPort === undefined) {
            const restServer = component.get(remoteMachineJobRestServer_1.RemoteMachineJobRestServer);
            this.remoteRestServerPort = restServer.clusterRestServerPort;
        }
        const version = this.versionCheck ? await utils_1.getVersion() : '';
        const runScriptTrialContent = executor.generateStartScript(trialJobDetail.workingDirectory, trialJobId, experimentStartupInfo_1.getExperimentId(), trialJobDetail.form.sequenceId.toString(), false, this.config.trialCommand, nniManagerIp, this.remoteRestServerPort, version, this.logCollection, cudaVisible);
        await util_1.execMkdir(path_1.default.join(trialLocalTempFolder, '.nni'));
        await fs_1.default.promises.writeFile(path_1.default.join(trialLocalTempFolder, executor.getScriptName("install_nni")), containerJobData_1.CONTAINER_INSTALL_NNI_SHELL_FORMAT, { encoding: 'utf8' });
        await fs_1.default.promises.writeFile(path_1.default.join(trialLocalTempFolder, executor.getScriptName("run")), runScriptTrialContent, { encoding: 'utf8' });
        await this.writeParameterFile(trialJobId, form.hyperParameters);
        await executor.copyDirectoryToRemote(trialLocalTempFolder, trialJobDetail.workingDirectory);
        executor.executeScript(executor.joinPath(trialJobDetail.workingDirectory, executor.getScriptName("run")), true, true);
    }
    async updateTrialJobStatus(trialJob, executor) {
        const deferred = new ts_deferred_1.Deferred();
        const jobpidPath = this.getJobPidPath(executor, trialJob.id);
        const trialReturnCodeFilePath = executor.joinPath(executor.getRemoteExperimentRootDir(experimentStartupInfo_1.getExperimentId()), 'trials', trialJob.id, '.nni', 'code');
        try {
            const isAlive = await executor.isProcessAlive(jobpidPath);
            if (!isAlive) {
                const trialReturnCode = await executor.getRemoteFileContent(trialReturnCodeFilePath);
                this.log.debug(`trailjob ${trialJob.id} return code: ${trialReturnCode}`);
                const match = trialReturnCode.trim()
                    .match(/^-?(\d+)\s+(\d+)$/);
                if (match !== null) {
                    const { 1: code, 2: timestamp } = match;
                    if (parseInt(code, 10) === 0) {
                        trialJob.status = 'SUCCEEDED';
                    }
                    else {
                        if (trialJob.isEarlyStopped === undefined) {
                            trialJob.status = 'FAILED';
                        }
                        else {
                            trialJob.status = utils_1.getJobCancelStatus(trialJob.isEarlyStopped);
                        }
                    }
                    trialJob.endTime = parseInt(timestamp, 10);
                    this.releaseTrialResource(trialJob);
                }
                this.log.debug(`trailJob status update: ${trialJob.id}, ${trialJob.status}`);
            }
            deferred.resolve(trialJob);
        }
        catch (error) {
            this.log.debug(`(Ignorable mostly)Update job status exception, error is ${error.message}`);
            if (error instanceof errors_1.NNIError && error.name === errors_1.NNIErrorNames.NOT_FOUND) {
                deferred.resolve(trialJob);
            }
            else {
                trialJob.status = 'UNKNOWN';
                deferred.resolve(trialJob);
            }
        }
        return deferred.promise;
    }
    get MetricsEmitter() {
        return this.metricsEmitter;
    }
    getJobPidPath(executor, jobId) {
        const trialJobDetail = this.trialJobsMap.get(jobId);
        if (trialJobDetail === undefined) {
            throw new errors_1.NNIError(errors_1.NNIErrorNames.INVALID_JOB_DETAIL, `Invalid job detail information for trial job ${jobId}`);
        }
        return executor.joinPath(trialJobDetail.workingDirectory, '.nni', 'jobpid');
    }
    async writeParameterFile(trialJobId, hyperParameters) {
        const executor = await this.getExecutor(trialJobId);
        const trialWorkingFolder = executor.joinPath(executor.getRemoteExperimentRootDir(experimentStartupInfo_1.getExperimentId()), 'trials', trialJobId);
        const trialLocalTempFolder = path_1.default.join(this.expRootDir, 'trials', trialJobId);
        const fileName = utils_1.generateParamFileName(hyperParameters);
        const localFilepath = path_1.default.join(trialLocalTempFolder, fileName);
        await fs_1.default.promises.writeFile(localFilepath, hyperParameters.value, { encoding: 'utf8' });
        await executor.copyFileToRemote(localFilepath, executor.joinPath(trialWorkingFolder, fileName));
    }
    getTrialOutputLocalPath(_trialJobId) {
        throw new errors_1.MethodNotImplementedError();
    }
    fetchTrialOutput(_trialJobId, _subpath) {
        throw new errors_1.MethodNotImplementedError();
    }
};
RemoteMachineTrainingService = __decorate([
    component.Singleton,
    __metadata("design:paramtypes", [Object])
], RemoteMachineTrainingService);
exports.RemoteMachineTrainingService = RemoteMachineTrainingService;
