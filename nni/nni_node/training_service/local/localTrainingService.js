"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.LocalTrainingService = void 0;
const events_1 = require("events");
const fs_1 = __importDefault(require("fs"));
const path_1 = __importDefault(require("path"));
const tail_stream_1 = __importDefault(require("tail-stream"));
const tree_kill_1 = __importDefault(require("tree-kill"));
const errors_1 = require("common/errors");
const experimentStartupInfo_1 = require("common/experimentStartupInfo");
const log_1 = require("common/log");
const shellUtils_1 = require("common/shellUtils");
const utils_1 = require("common/utils");
const util_1 = require("../common/util");
const gpuScheduler_1 = require("./gpuScheduler");
function decodeCommand(data) {
    if (data.length < 8) {
        return [false, '', '', data];
    }
    const commandType = data.slice(0, 2).toString();
    const contentLength = parseInt(data.slice(2, 8).toString(), 10);
    if (data.length < contentLength + 8) {
        return [false, '', '', data];
    }
    const content = data.slice(8, contentLength + 8).toString();
    const remain = data.slice(contentLength + 8);
    return [true, commandType, content, remain];
}
class LocalTrialJobDetail {
    id;
    status;
    submitTime;
    startTime;
    endTime;
    tags;
    url;
    workingDirectory;
    form;
    pid;
    gpuIndices;
    constructor(id, status, submitTime, workingDirectory, form) {
        this.id = id;
        this.status = status;
        this.submitTime = submitTime;
        this.workingDirectory = workingDirectory;
        this.form = form;
        this.url = `file://localhost:${workingDirectory}`;
        this.gpuIndices = [];
    }
}
class LocalTrainingService {
    config;
    eventEmitter;
    jobMap;
    jobQueue;
    initialized;
    stopping;
    rootDir;
    experimentId;
    gpuScheduler;
    occupiedGpuIndexNumMap;
    log;
    jobStreamMap;
    constructor(config) {
        this.config = config;
        this.eventEmitter = new events_1.EventEmitter();
        this.jobMap = new Map();
        this.jobQueue = [];
        this.stopping = false;
        this.log = log_1.getLogger('LocalTrainingService');
        this.experimentId = experimentStartupInfo_1.getExperimentId();
        this.jobStreamMap = new Map();
        this.log.info('Construct local machine training service.');
        this.occupiedGpuIndexNumMap = new Map();
        if (this.config.trialGpuNumber !== undefined && this.config.trialGpuNumber > 0) {
            this.gpuScheduler = new gpuScheduler_1.GPUScheduler();
        }
        if (this.config.gpuIndices === []) {
            throw new Error('gpuIndices cannot be empty when specified.');
        }
        this.rootDir = utils_1.getExperimentRootDir();
        if (!fs_1.default.existsSync(this.rootDir)) {
            throw new Error('root dir not created');
        }
        this.initialized = true;
    }
    async run() {
        this.log.info('Run local machine training service.');
        const longRunningTasks = [this.runJobLoop()];
        if (this.gpuScheduler !== undefined) {
            longRunningTasks.push(this.gpuScheduler.run());
        }
        await Promise.all(longRunningTasks);
        this.log.info('Local machine training service exit.');
    }
    async listTrialJobs() {
        const jobs = [];
        for (const key of this.jobMap.keys()) {
            const trialJob = await this.getTrialJob(key);
            jobs.push(trialJob);
        }
        return jobs;
    }
    async getTrialJob(trialJobId) {
        const trialJob = this.jobMap.get(trialJobId);
        if (trialJob === undefined) {
            throw new errors_1.NNIError(errors_1.NNIErrorNames.NOT_FOUND, 'Trial job not found');
        }
        if (trialJob.status === 'RUNNING') {
            const alive = await utils_1.isAlive(trialJob.pid);
            if (!alive) {
                trialJob.endTime = Date.now();
                this.setTrialJobStatus(trialJob, 'FAILED');
                try {
                    const state = await fs_1.default.promises.readFile(path_1.default.join(trialJob.workingDirectory, '.nni', 'state'), 'utf8');
                    const match = state.trim()
                        .match(/^(\d+)\s+(\d+)/);
                    if (match !== null) {
                        const { 1: code, 2: timestamp } = match;
                        if (parseInt(code, 10) === 0) {
                            this.setTrialJobStatus(trialJob, 'SUCCEEDED');
                        }
                        trialJob.endTime = parseInt(timestamp, 10);
                    }
                }
                catch (error) {
                }
                this.log.debug(`trialJob status update: ${trialJobId}, ${trialJob.status}`);
            }
        }
        return trialJob;
    }
    async getTrialFile(trialJobId, fileName) {
        if (!['trial.log', 'stderr', 'model.onnx', 'stdout'].includes(fileName)) {
            throw new Error(`File unaccessible: ${fileName}`);
        }
        let encoding = null;
        if (!fileName.includes('.') || fileName.match(/.*\.(txt|log)/g)) {
            encoding = 'utf8';
        }
        const logPath = path_1.default.join(this.rootDir, 'trials', trialJobId, fileName);
        if (!fs_1.default.existsSync(logPath)) {
            throw new Error(`File not found: ${logPath}`);
        }
        return fs_1.default.promises.readFile(logPath, { encoding: encoding });
    }
    addTrialJobMetricListener(listener) {
        this.eventEmitter.on('metric', listener);
    }
    removeTrialJobMetricListener(listener) {
        this.eventEmitter.off('metric', listener);
    }
    submitTrialJob(form) {
        const trialJobId = utils_1.uniqueString(5);
        const trialJobDetail = new LocalTrialJobDetail(trialJobId, 'WAITING', Date.now(), path_1.default.join(this.rootDir, 'trials', trialJobId), form);
        this.jobQueue.push(trialJobId);
        this.jobMap.set(trialJobId, trialJobDetail);
        this.log.debug('submitTrialJob: return:', trialJobDetail);
        return Promise.resolve(trialJobDetail);
    }
    async updateTrialJob(trialJobId, form) {
        const trialJobDetail = this.jobMap.get(trialJobId);
        if (trialJobDetail === undefined) {
            throw new Error(`updateTrialJob failed: ${trialJobId} not found`);
        }
        await this.writeParameterFile(trialJobDetail.workingDirectory, form.hyperParameters);
        return trialJobDetail;
    }
    async cancelTrialJob(trialJobId, isEarlyStopped = false) {
        const trialJob = this.jobMap.get(trialJobId);
        if (trialJob === undefined) {
            throw new errors_1.NNIError(errors_1.NNIErrorNames.NOT_FOUND, 'Trial job not found');
        }
        if (trialJob.pid === undefined) {
            this.setTrialJobStatus(trialJob, 'USER_CANCELED');
            return Promise.resolve();
        }
        tree_kill_1.default(trialJob.pid, 'SIGTERM');
        this.setTrialJobStatus(trialJob, utils_1.getJobCancelStatus(isEarlyStopped));
        const startTime = Date.now();
        while (await utils_1.isAlive(trialJob.pid)) {
            if (Date.now() - startTime > 4999) {
                tree_kill_1.default(trialJob.pid, 'SIGKILL', (err) => {
                    if (err) {
                        this.log.error(`kill trial job error: ${err}`);
                    }
                });
                break;
            }
            await utils_1.delay(500);
        }
        return Promise.resolve();
    }
    async setClusterMetadata(_key, _value) { return; }
    async getClusterMetadata(_key) { return ''; }
    async cleanUp() {
        this.log.info('Stopping local machine training service...');
        this.stopping = true;
        for (const stream of this.jobStreamMap.values()) {
            stream.end(0);
            stream.emit('end');
        }
        if (this.gpuScheduler !== undefined) {
            await this.gpuScheduler.stop();
        }
        return Promise.resolve();
    }
    onTrialJobStatusChanged(trialJob, oldStatus) {
        if (['SUCCEEDED', 'FAILED', 'USER_CANCELED', 'SYS_CANCELED', 'EARLY_STOPPED'].includes(trialJob.status)) {
            if (this.jobStreamMap.has(trialJob.id)) {
                const stream = this.jobStreamMap.get(trialJob.id);
                if (stream === undefined) {
                    throw new Error(`Could not find stream in trial ${trialJob.id}`);
                }
                setTimeout(() => {
                    stream.end(0);
                    stream.emit('end');
                    this.jobStreamMap.delete(trialJob.id);
                }, 5000);
            }
        }
        if (trialJob.gpuIndices !== undefined && trialJob.gpuIndices.length > 0 && this.gpuScheduler !== undefined) {
            if (oldStatus === 'RUNNING' && trialJob.status !== 'RUNNING') {
                for (const index of trialJob.gpuIndices) {
                    const num = this.occupiedGpuIndexNumMap.get(index);
                    if (num === undefined) {
                        throw new Error(`gpu resource schedule error`);
                    }
                    else if (num === 1) {
                        this.occupiedGpuIndexNumMap.delete(index);
                    }
                    else {
                        this.occupiedGpuIndexNumMap.set(index, num - 1);
                    }
                }
            }
        }
    }
    getEnvironmentVariables(trialJobDetail, resource, gpuNum) {
        const envVariables = [
            { key: 'NNI_PLATFORM', value: 'local' },
            { key: 'NNI_EXP_ID', value: this.experimentId },
            { key: 'NNI_SYS_DIR', value: trialJobDetail.workingDirectory },
            { key: 'NNI_TRIAL_JOB_ID', value: trialJobDetail.id },
            { key: 'NNI_OUTPUT_DIR', value: trialJobDetail.workingDirectory },
            { key: 'NNI_TRIAL_SEQ_ID', value: trialJobDetail.form.sequenceId.toString() },
            { key: 'NNI_CODE_DIR', value: this.config.trialCodeDirectory }
        ];
        if (gpuNum !== undefined) {
            envVariables.push({
                key: 'CUDA_VISIBLE_DEVICES',
                value: this.gpuScheduler === undefined ? '-1' : resource.gpuIndices.join(',')
            });
        }
        return envVariables;
    }
    setExtraProperties(trialJobDetail, resource) {
        trialJobDetail.gpuIndices = resource.gpuIndices;
    }
    tryGetAvailableResource() {
        const resource = { gpuIndices: [] };
        if (this.gpuScheduler === undefined) {
            return [true, resource];
        }
        let selectedGPUIndices = [];
        const availableGpuIndices = this.gpuScheduler.getAvailableGPUIndices(this.config.useActiveGpu, this.occupiedGpuIndexNumMap);
        for (const index of availableGpuIndices) {
            const num = this.occupiedGpuIndexNumMap.get(index);
            if (num === undefined || num < this.config.maxTrialNumberPerGpu) {
                selectedGPUIndices.push(index);
            }
        }
        if (this.config.gpuIndices !== undefined) {
            this.checkSpecifiedGpuIndices();
            selectedGPUIndices = selectedGPUIndices.filter((index) => this.config.gpuIndices.includes(index));
        }
        if (selectedGPUIndices.length < this.config.trialGpuNumber) {
            return [false, resource];
        }
        selectedGPUIndices.splice(this.config.trialGpuNumber);
        Object.assign(resource, { gpuIndices: selectedGPUIndices });
        return [true, resource];
    }
    checkSpecifiedGpuIndices() {
        const gpuCount = this.gpuScheduler.getSystemGpuCount();
        if (this.config.gpuIndices !== undefined && gpuCount !== undefined) {
            for (const index of this.config.gpuIndices) {
                if (index >= gpuCount) {
                    throw new Error(`Specified GPU index not found: ${index}`);
                }
            }
        }
    }
    occupyResource(resource) {
        if (this.gpuScheduler !== undefined) {
            for (const index of resource.gpuIndices) {
                const num = this.occupiedGpuIndexNumMap.get(index);
                if (num === undefined) {
                    this.occupiedGpuIndexNumMap.set(index, 1);
                }
                else {
                    this.occupiedGpuIndexNumMap.set(index, num + 1);
                }
            }
        }
    }
    async runJobLoop() {
        while (!this.stopping) {
            while (!this.stopping && this.jobQueue.length !== 0) {
                const trialJobId = this.jobQueue[0];
                const trialJobDetail = this.jobMap.get(trialJobId);
                if (trialJobDetail !== undefined && trialJobDetail.status === 'WAITING') {
                    const [success, resource] = this.tryGetAvailableResource();
                    if (!success) {
                        break;
                    }
                    this.occupyResource(resource);
                    await this.runTrialJob(trialJobId, resource);
                }
                this.jobQueue.shift();
            }
            await utils_1.delay(5000);
        }
    }
    setTrialJobStatus(trialJob, newStatus) {
        if (trialJob.status !== newStatus) {
            const oldStatus = trialJob.status;
            trialJob.status = newStatus;
            this.onTrialJobStatusChanged(trialJob, oldStatus);
        }
    }
    getScript(workingDirectory) {
        const script = [];
        if (process.platform === 'win32') {
            script.push(`$PSDefaultParameterValues = @{'Out-File:Encoding' = 'utf8'}`);
            script.push(`cd $env:NNI_CODE_DIR`);
            script.push(`cmd.exe /c ${this.config.trialCommand} 1>${path_1.default.join(workingDirectory, 'stdout')} 2>${path_1.default.join(workingDirectory, 'stderr')}`, `$NOW_DATE = [int64](([datetime]::UtcNow)-(get-date "1/1/1970")).TotalSeconds`, `$NOW_DATE = "$NOW_DATE" + (Get-Date -Format fff).ToString()`, `Write $LASTEXITCODE " " $NOW_DATE  | Out-File "${path_1.default.join(workingDirectory, '.nni', 'state')}" -NoNewline -encoding utf8`);
        }
        else {
            script.push(`cd $NNI_CODE_DIR`);
            script.push(`eval ${this.config.trialCommand} 1>${path_1.default.join(workingDirectory, 'stdout')} 2>${path_1.default.join(workingDirectory, 'stderr')}`);
            if (process.platform === 'darwin') {
                script.push(`echo $? \`date +%s999\` >'${path_1.default.join(workingDirectory, '.nni', 'state')}'`);
            }
            else {
                script.push(`echo $? \`date +%s%3N\` >'${path_1.default.join(workingDirectory, '.nni', 'state')}'`);
            }
        }
        return script;
    }
    async runTrialJob(trialJobId, resource) {
        const trialJobDetail = this.jobMap.get(trialJobId);
        const variables = this.getEnvironmentVariables(trialJobDetail, resource, this.config.trialGpuNumber);
        const runScriptContent = [];
        if (process.platform !== 'win32') {
            runScriptContent.push('#!/bin/bash');
        }
        else {
            runScriptContent.push(`$env:PATH=${shellUtils_1.powershellString(process.env['path'])}`);
        }
        for (const variable of variables) {
            runScriptContent.push(util_1.setEnvironmentVariable(variable));
        }
        const scripts = this.getScript(trialJobDetail.workingDirectory);
        scripts.forEach((script) => {
            runScriptContent.push(script);
        });
        await util_1.execMkdir(trialJobDetail.workingDirectory);
        await util_1.execMkdir(path_1.default.join(trialJobDetail.workingDirectory, '.nni'));
        await util_1.execNewFile(path_1.default.join(trialJobDetail.workingDirectory, '.nni', 'metrics'));
        const scriptName = util_1.getScriptName('run');
        await fs_1.default.promises.writeFile(path_1.default.join(trialJobDetail.workingDirectory, scriptName), runScriptContent.join(utils_1.getNewLine()), { encoding: 'utf8', mode: 0o777 });
        await this.writeParameterFile(trialJobDetail.workingDirectory, trialJobDetail.form.hyperParameters);
        const trialJobProcess = util_1.runScript(path_1.default.join(trialJobDetail.workingDirectory, scriptName));
        this.setTrialJobStatus(trialJobDetail, 'RUNNING');
        trialJobDetail.startTime = Date.now();
        trialJobDetail.pid = trialJobProcess.pid;
        this.setExtraProperties(trialJobDetail, resource);
        let buffer = Buffer.alloc(0);
        const stream = tail_stream_1.default.createReadStream(path_1.default.join(trialJobDetail.workingDirectory, '.nni', 'metrics'));
        stream.on('data', (data) => {
            buffer = Buffer.concat([buffer, data]);
            while (buffer.length > 0) {
                const [success, , content, remain] = decodeCommand(buffer);
                if (!success) {
                    break;
                }
                this.eventEmitter.emit('metric', {
                    id: trialJobDetail.id,
                    data: content
                });
                this.log.debug(`Sending metrics, job id: ${trialJobDetail.id}, metrics: ${content}`);
                buffer = remain;
            }
        });
        this.jobStreamMap.set(trialJobDetail.id, stream);
    }
    async writeParameterFile(directory, hyperParameters) {
        const filepath = path_1.default.join(directory, utils_1.generateParamFileName(hyperParameters));
        await fs_1.default.promises.writeFile(filepath, hyperParameters.value, { encoding: 'utf8' });
    }
    async getTrialOutputLocalPath(trialJobId) {
        return Promise.resolve(path_1.default.join(this.rootDir, 'trials', trialJobId));
    }
    async fetchTrialOutput(trialJobId, subpath) {
        let trialLocalPath = await this.getTrialOutputLocalPath(trialJobId);
        if (subpath !== undefined) {
            trialLocalPath = path_1.default.join(trialLocalPath, subpath);
        }
        if (fs_1.default.existsSync(trialLocalPath)) {
            return Promise.resolve();
        }
        else {
            return Promise.reject(new Error('Trial local path not exist.'));
        }
    }
}
exports.LocalTrainingService = LocalTrainingService;
