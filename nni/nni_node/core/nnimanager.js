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
var __importStar = (this && this.__importStar) || function (mod) {
    if (mod && mod.__esModule) return mod;
    var result = {};
    if (mod != null) for (var k in mod) if (k !== "default" && Object.prototype.hasOwnProperty.call(mod, k)) __createBinding(result, mod, k);
    __setModuleDefault(result, mod);
    return result;
};
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.NNIManager = void 0;
const assert_1 = __importDefault(require("assert"));
const ts_deferred_1 = require("ts-deferred");
const component = __importStar(require("../common/component"));
const datastore_1 = require("../common/datastore");
const errors_1 = require("../common/errors");
const experimentStartupInfo_1 = require("../common/experimentStartupInfo");
const log_1 = require("../common/log");
const experimentConfig_1 = require("../common/experimentConfig");
const experimentManager_1 = require("../common/experimentManager");
const tensorboardManager_1 = require("../common/tensorboardManager");
const utils_1 = require("../common/utils");
const commands_1 = require("./commands");
const ipcInterface_1 = require("./ipcInterface");
const rest_server_1 = require("../rest_server");
class NNIManager {
    trainingService;
    dispatcher;
    experimentManager;
    currSubmittedTrialNum;
    trialConcurrencyChange;
    log;
    dataStore;
    experimentProfile;
    dispatcherPid;
    status;
    waitingTrials;
    trialJobs;
    trialDataForTuner;
    readonly;
    config;
    trialJobMetricListener;
    constructor() {
        this.currSubmittedTrialNum = 0;
        this.trialConcurrencyChange = 0;
        this.experimentManager = component.get(experimentManager_1.ExperimentManager);
        this.dispatcherPid = 0;
        this.waitingTrials = [];
        this.trialJobs = new Map();
        this.trialDataForTuner = '';
        this.readonly = false;
        this.log = log_1.getLogger('NNIManager');
        this.dataStore = component.get(datastore_1.DataStore);
        this.status = {
            status: 'INITIALIZED',
            errors: []
        };
        this.trialJobMetricListener = (metric) => {
            this.onTrialJobMetrics(metric).catch((err) => {
                this.criticalError(errors_1.NNIError.FromError(err, 'Job metrics error: '));
            });
        };
        const pipe = experimentStartupInfo_1.getDispatcherPipe();
        if (pipe !== null) {
            this.dispatcher = ipcInterface_1.createDispatcherPipeInterface(pipe);
        }
    }
    updateExperimentProfile(experimentProfile, updateType) {
        if (this.readonly) {
            return Promise.reject(new Error('Error: can not update experiment profile in readonly mode!'));
        }
        switch (updateType) {
            case 'TRIAL_CONCURRENCY':
                this.updateTrialConcurrency(experimentProfile.params.trialConcurrency);
                break;
            case 'MAX_EXEC_DURATION':
                this.experimentProfile.params.maxExperimentDuration = experimentProfile.params.maxExperimentDuration;
                break;
            case 'SEARCH_SPACE':
                this.updateSearchSpace(experimentProfile.params.searchSpace);
                break;
            case 'MAX_TRIAL_NUM':
                this.experimentProfile.params.maxTrialNumber = experimentProfile.params.maxTrialNumber;
                break;
            default:
                throw new Error('Error: unrecognized updateType');
        }
        return this.storeExperimentProfile();
    }
    importData(data) {
        if (this.readonly) {
            return Promise.reject(new Error('Error: can not import data in readonly mode!'));
        }
        if (this.dispatcher === undefined) {
            return Promise.reject(new Error('tuner has not been setup'));
        }
        this.dispatcher.sendCommand(commands_1.IMPORT_DATA, data);
        return this.dataStore.storeTrialJobEvent('IMPORT_DATA', '', data);
    }
    getImportedData() {
        return this.dataStore.getImportedData();
    }
    async exportData() {
        return this.dataStore.exportTrialHpConfigs();
    }
    addCustomizedTrialJob(hyperParams) {
        if (this.readonly) {
            return Promise.reject(new Error('Error: can not add customized trial job in readonly mode!'));
        }
        if (this.currSubmittedTrialNum >= this.maxTrialNum) {
            return Promise.reject(new Error('reach maxTrialNum'));
        }
        const packedParameter = {
            parameter_id: null,
            parameter_source: 'customized',
            parameters: JSON.parse(hyperParams)
        };
        const form = {
            sequenceId: this.experimentProfile.nextSequenceId++,
            hyperParameters: {
                value: JSON.stringify(packedParameter),
                index: 0
            }
        };
        this.waitingTrials.push(form);
        this.dataStore.storeTrialJobEvent('ADD_CUSTOMIZED', '', hyperParams);
        return Promise.resolve(form.sequenceId);
    }
    async cancelTrialJobByUser(trialJobId) {
        if (this.readonly) {
            return Promise.reject(new Error('Error: can not cancel trial job in readonly mode!'));
        }
        this.log.info(`User cancelTrialJob: ${trialJobId}`);
        await this.trainingService.cancelTrialJob(trialJobId);
        await this.dataStore.storeTrialJobEvent('USER_TO_CANCEL', trialJobId, '');
    }
    async startExperiment(config) {
        this.experimentProfile = {
            params: config,
            id: experimentStartupInfo_1.getExperimentId(),
            execDuration: 0,
            logDir: utils_1.getExperimentRootDir(),
            startTime: Date.now(),
            endTime: undefined,
            nextSequenceId: 0,
            revision: 0
        };
        this.config = config;
        this.log.info(`Starting experiment: ${this.experimentProfile.id}`);
        await this.storeExperimentProfile();
        if (this.trainingService === undefined) {
            this.log.info('Setup training service...');
            this.trainingService = await this.initTrainingService(config);
        }
        this.log.info('Setup tuner...');
        const dispatcherCommand = utils_1.getMsgDispatcherCommand(config);
        this.log.debug(`dispatcher command: ${dispatcherCommand}`);
        const checkpointDir = await this.createCheckpointDir();
        this.setupTuner(dispatcherCommand, undefined, 'start', checkpointDir);
        this.setStatus('RUNNING');
        await this.storeExperimentProfile();
        this.run().catch((err) => {
            this.criticalError(err);
        });
        return this.experimentProfile.id;
    }
    async resumeExperiment(readonly) {
        const experimentId = experimentStartupInfo_1.getExperimentId();
        this.log.info(`Resuming experiment: ${experimentId}`);
        this.experimentProfile = await this.dataStore.getExperimentProfile(experimentId);
        const config = this.experimentProfile.params;
        this.config = config;
        if (this.trainingService === undefined) {
            this.log.info('Setup training service...');
            this.trainingService = await this.initTrainingService(config);
        }
        this.readonly = readonly;
        if (readonly) {
            this.setStatus('VIEWED');
            return;
        }
        this.log.info('Setup tuner...');
        const dispatcherCommand = utils_1.getMsgDispatcherCommand(config);
        this.log.debug(`dispatcher command: ${dispatcherCommand}`);
        const checkpointDir = await this.createCheckpointDir();
        this.setupTuner(dispatcherCommand, undefined, 'resume', checkpointDir);
        const allTrialJobs = await this.dataStore.listTrialJobs();
        this.currSubmittedTrialNum = allTrialJobs.length;
        await Promise.all(allTrialJobs
            .filter((job) => job.status === 'WAITING' || job.status === 'RUNNING')
            .map((job) => this.dataStore.storeTrialJobEvent('FAILED', job.trialJobId)));
        const finishedTrialData = await this.exportData();
        const importedData = await this.dataStore.getImportedData();
        let trialData = JSON.parse(finishedTrialData);
        for (const oneImportedData of importedData) {
            trialData = trialData.concat(JSON.parse(oneImportedData));
        }
        this.trialDataForTuner = JSON.stringify(trialData);
        if (this.experimentProfile.execDuration < this.maxDuration &&
            this.currSubmittedTrialNum < this.maxTrialNum &&
            this.experimentProfile.endTime) {
            delete this.experimentProfile.endTime;
        }
        this.setStatus('RUNNING');
        this.run().catch((err) => {
            this.criticalError(err);
        });
    }
    getTrialJob(trialJobId) {
        return this.dataStore.getTrialJob(trialJobId);
    }
    async setClusterMetadata(key, value) {
        if (this.trainingService === undefined) {
            this.log.info('Setup training service...');
            switch (key) {
                case 'kubeflow_config': {
                    const kubeflowModule = await Promise.resolve().then(() => __importStar(require('../training_service/kubernetes/kubeflow/kubeflowTrainingService')));
                    this.trainingService = new kubeflowModule.KubeflowTrainingService();
                    break;
                }
                case 'frameworkcontroller_config': {
                    const fcModule = await Promise.resolve().then(() => __importStar(require('../training_service/kubernetes/frameworkcontroller/frameworkcontrollerTrainingService')));
                    this.trainingService = new fcModule.FrameworkControllerTrainingService();
                    break;
                }
                case 'adl_config': {
                    const adlModule = await Promise.resolve().then(() => __importStar(require('../training_service/kubernetes/adl/adlTrainingService')));
                    this.trainingService = new adlModule.AdlTrainingService();
                    break;
                }
                default:
                    throw new Error("Setup training service failed.");
            }
        }
        await this.trainingService.setClusterMetadata(key, value);
    }
    getClusterMetadata(key) {
        return this.trainingService.getClusterMetadata(key);
    }
    async getTrialJobStatistics() {
        return this.dataStore.getTrialJobStatistics();
    }
    async stopExperiment() {
        await this.stopExperimentTopHalf();
        await this.stopExperimentBottomHalf();
    }
    async stopExperimentTopHalf() {
        this.setStatus('STOPPING');
        this.log.info('Stopping experiment, cleaning up ...');
        if (this.dispatcher === undefined) {
            this.log.error('Tuner has not been setup');
            return;
        }
        this.trainingService.removeTrialJobMetricListener(this.trialJobMetricListener);
        if (this.dispatcherPid > 0) {
            this.dispatcher.sendCommand(commands_1.TERMINATE);
            for (let i = 0; i < 30; i++) {
                if (!await utils_1.isAlive(this.dispatcherPid)) {
                    break;
                }
                await utils_1.delay(1000);
            }
            await utils_1.killPid(this.dispatcherPid);
        }
        this.dispatcher = undefined;
    }
    async stopExperimentBottomHalf() {
        try {
            const trialJobList = await this.trainingService.listTrialJobs();
            for (const trialJob of trialJobList) {
                if (trialJob.status === 'RUNNING' ||
                    trialJob.status === 'WAITING') {
                    try {
                        this.log.info(`cancelTrialJob: ${trialJob.id}`);
                        await this.trainingService.cancelTrialJob(trialJob.id);
                    }
                    catch (error) {
                        this.log.debug(`ignorable error on canceling trial ${trialJob.id}. ${error}`);
                    }
                }
            }
            await this.trainingService.cleanUp();
        }
        catch (err) {
            this.log.error(`${err.stack}`);
        }
        if (this.experimentProfile.endTime === undefined) {
            this.setEndtime();
        }
        await this.storeExperimentProfile();
        this.setStatus('STOPPED');
        this.log.info('Experiment stopped.');
        let hasError = false;
        try {
            await this.experimentManager.stop();
            await component.get(tensorboardManager_1.TensorboardManager).stop();
            await this.dataStore.close();
            await component.get(rest_server_1.RestServer).shutdown();
        }
        catch (err) {
            hasError = true;
            this.log.error(`${err.stack}`);
        }
        finally {
            log_1.stopLogging();
            process.exit(hasError ? 1 : 0);
        }
    }
    async getMetricData(trialJobId, metricType) {
        return this.dataStore.getMetricData(trialJobId, metricType);
    }
    async getMetricDataByRange(minSeqId, maxSeqId) {
        const trialJobs = await this.dataStore.listTrialJobs();
        const targetTrials = trialJobs.filter(trial => (trial.sequenceId !== undefined && minSeqId <= trial.sequenceId && trial.sequenceId <= maxSeqId));
        const targetTrialIds = new Set(targetTrials.map(trial => trial.trialJobId));
        const allMetrics = await this.dataStore.getMetricData();
        return allMetrics.filter(metric => targetTrialIds.has(metric.trialJobId));
    }
    async getLatestMetricData() {
        const allMetrics = await this.dataStore.getMetricData();
        const finals = [];
        const latestIntermediates = new Map();
        for (const metric of allMetrics) {
            if (metric.type !== 'PERIODICAL') {
                finals.push(metric);
            }
            else {
                const old = latestIntermediates.get(metric.trialJobId);
                if (old === undefined || old.sequence <= metric.sequence) {
                    latestIntermediates.set(metric.trialJobId, metric);
                }
            }
        }
        return finals.concat(Array.from(latestIntermediates.values()));
    }
    async getTrialFile(trialJobId, fileName) {
        return this.trainingService.getTrialFile(trialJobId, fileName);
    }
    getExperimentProfile() {
        const deferred = new ts_deferred_1.Deferred();
        deferred.resolve(this.experimentProfile);
        return deferred.promise;
    }
    getStatus() {
        return this.status;
    }
    async listTrialJobs(status) {
        return this.dataStore.listTrialJobs(status);
    }
    get maxDuration() {
        const value = this.experimentProfile.params.maxExperimentDuration;
        return (value === undefined ? Infinity : experimentConfig_1.toSeconds(value));
    }
    get maxTrialNum() {
        const value = this.experimentProfile.params.maxTrialNumber;
        return (value === undefined ? Infinity : value);
    }
    get maxTrialDuration() {
        const value = this.experimentProfile.params.maxTrialDuration;
        return (value === undefined ? Infinity : experimentConfig_1.toSeconds(value));
    }
    async initTrainingService(config) {
        let platform;
        if (Array.isArray(config.trainingService)) {
            platform = 'hybrid';
        }
        else if (config.trainingService.platform) {
            platform = config.trainingService.platform;
        }
        else {
            platform = config.trainingServicePlatform;
        }
        if (!platform) {
            throw new Error('Cannot detect training service platform');
        }
        const reuseMode = Array.isArray(config.trainingService) || config.trainingService.reuseMode;
        if (reuseMode) {
            const module_ = await Promise.resolve().then(() => __importStar(require('../training_service/reusable/routerTrainingService')));
            return await module_.RouterTrainingService.construct(config);
        }
        else if (platform === 'local') {
            const module_ = await Promise.resolve().then(() => __importStar(require('../training_service/local/localTrainingService')));
            return new module_.LocalTrainingService(config.trainingService);
        }
        else if (platform === 'kubeflow') {
            const module_ = await Promise.resolve().then(() => __importStar(require('../training_service/kubernetes/kubeflow/kubeflowTrainingService')));
            return new module_.KubeflowTrainingService();
        }
        else if (platform === 'frameworkcontroller') {
            const module_ = await Promise.resolve().then(() => __importStar(require('../training_service/kubernetes/frameworkcontroller/frameworkcontrollerTrainingService')));
            return new module_.FrameworkControllerTrainingService();
        }
        else if (platform === 'adl') {
            const module_ = await Promise.resolve().then(() => __importStar(require('../training_service/kubernetes/adl/adlTrainingService')));
            return new module_.AdlTrainingService();
        }
        else {
            const module_ = await Promise.resolve().then(() => __importStar(require('../training_service/reusable/routerTrainingService')));
            return await module_.RouterTrainingService.construct(config);
        }
    }
    setupTuner(command, cwd, mode, dataDirectory) {
        if (this.dispatcher !== undefined) {
            return;
        }
        const stdio = ['ignore', process.stdout, process.stderr, 'pipe', 'pipe'];
        let newCwd;
        if (cwd === undefined || cwd === '') {
            newCwd = utils_1.getLogDir();
        }
        else {
            newCwd = cwd;
        }
        const includeIntermediateResultsEnv = !!(this.config.deprecated && this.config.deprecated.includeIntermediateResults);
        const nniEnv = {
            SDK_PROCESS: 'dispatcher',
            NNI_MODE: mode,
            NNI_CHECKPOINT_DIRECTORY: dataDirectory,
            NNI_LOG_DIRECTORY: utils_1.getLogDir(),
            NNI_LOG_LEVEL: utils_1.getLogLevel(),
            NNI_INCLUDE_INTERMEDIATE_RESULTS: includeIntermediateResultsEnv,
            CUDA_VISIBLE_DEVICES: experimentConfig_1.toCudaVisibleDevices(this.experimentProfile.params.tunerGpuIndices)
        };
        const newEnv = Object.assign({}, process.env, nniEnv);
        const tunerProc = utils_1.getTunerProc(command, stdio, newCwd, newEnv);
        this.dispatcherPid = tunerProc.pid;
        this.dispatcher = ipcInterface_1.createDispatcherInterface(tunerProc);
        return;
    }
    updateTrialConcurrency(trialConcurrency) {
        this.trialConcurrencyChange += (trialConcurrency - this.experimentProfile.params.trialConcurrency);
        this.experimentProfile.params.trialConcurrency = trialConcurrency;
        return;
    }
    updateSearchSpace(searchSpace) {
        if (this.dispatcher === undefined) {
            throw new Error('Error: tuner has not been setup');
        }
        this.log.info(`Updated search space ${searchSpace}`);
        this.dispatcher.sendCommand(commands_1.UPDATE_SEARCH_SPACE, JSON.stringify(searchSpace));
        this.experimentProfile.params.searchSpace = searchSpace;
        return;
    }
    async periodicallyUpdateExecDuration() {
        let count = 1;
        while (!['ERROR', 'STOPPING', 'STOPPED'].includes(this.status.status)) {
            await utils_1.delay(1000 * 1);
            if (['RUNNING', 'NO_MORE_TRIAL', 'TUNER_NO_MORE_TRIAL'].includes(this.status.status)) {
                this.experimentProfile.execDuration += 1;
                if (count % 10 === 0) {
                    await this.storeExperimentProfile();
                }
            }
            count += 1;
        }
    }
    async pingDispatcher() {
        if (this.dispatcher === undefined) {
            throw new Error('Error: tuner has not been setup');
        }
        while (!['ERROR', 'STOPPING', 'STOPPED'].includes(this.status.status)) {
            this.dispatcher.sendCommand(commands_1.PING);
            await utils_1.delay(1000 * 5);
        }
    }
    async stopTrialIfOverMaxDurationLimit() {
        if (this.maxTrialDuration === Infinity) {
            return;
        }
        for (const trialJobId of Array.from(this.trialJobs.keys())) {
            const trialJobDetail = this.trialJobs.get(trialJobId);
            if (undefined !== trialJobDetail &&
                trialJobDetail.status === 'RUNNING' &&
                trialJobDetail.startTime !== undefined) {
                const currentTrialDuration = (new Date().getTime() - trialJobDetail.startTime) / 1000;
                if (currentTrialDuration > this.maxTrialDuration) {
                    const isEarlyStopped = true;
                    await this.trainingService.cancelTrialJob(trialJobId, isEarlyStopped);
                    this.log.info(`Trial job ${trialJobDetail.id} has been canceled because it is over max trial duration.`);
                }
            }
        }
    }
    async requestTrialJobsStatus() {
        let finishedTrialJobNum = 0;
        if (this.dispatcher === undefined) {
            throw new Error('Error: tuner has not been setup');
        }
        for (const trialJobId of Array.from(this.trialJobs.keys())) {
            const trialJobDetail = await this.trainingService.getTrialJob(trialJobId);
            const oldTrialJobDetail = this.trialJobs.get(trialJobId);
            if (oldTrialJobDetail !== undefined && oldTrialJobDetail.status !== trialJobDetail.status) {
                this.log.info(`Trial job ${trialJobDetail.id} status changed from ${oldTrialJobDetail.status} to ${trialJobDetail.status}`);
                this.trialJobs.set(trialJobId, Object.assign({}, trialJobDetail));
                await this.dataStore.storeTrialJobEvent(trialJobDetail.status, trialJobDetail.id, undefined, trialJobDetail);
            }
            const newTrialJobDetail = this.trialJobs.get(trialJobId);
            if (newTrialJobDetail !== undefined) {
                newTrialJobDetail.message = trialJobDetail.message;
            }
            let hyperParams = undefined;
            switch (trialJobDetail.status) {
                case 'SUCCEEDED':
                case 'USER_CANCELED':
                case 'EARLY_STOPPED':
                    this.trialJobs.delete(trialJobId);
                    finishedTrialJobNum++;
                    hyperParams = trialJobDetail.form.hyperParameters.value;
                    this.dispatcher.sendCommand(commands_1.TRIAL_END, JSON.stringify({
                        trial_job_id: trialJobDetail.id,
                        event: trialJobDetail.status,
                        hyper_params: hyperParams
                    }));
                    break;
                case 'FAILED':
                case 'SYS_CANCELED':
                    this.trialJobs.delete(trialJobId);
                    finishedTrialJobNum++;
                    hyperParams = trialJobDetail.form.hyperParameters.value;
                    this.dispatcher.sendCommand(commands_1.TRIAL_END, JSON.stringify({
                        trial_job_id: trialJobDetail.id,
                        event: trialJobDetail.status,
                        hyper_params: hyperParams
                    }));
                    break;
                case 'WAITING':
                case 'RUNNING':
                case 'UNKNOWN':
                    break;
                default:
            }
        }
        return finishedTrialJobNum;
    }
    async manageTrials() {
        if (this.dispatcher === undefined) {
            throw new Error('Error: tuner has not been setup');
        }
        let allFinishedTrialJobNum = this.currSubmittedTrialNum;
        let waitSubmittedToFinish;
        while (!['ERROR', 'STOPPING', 'STOPPED'].includes(this.status.status)) {
            await this.stopTrialIfOverMaxDurationLimit();
            const finishedTrialJobNum = await this.requestTrialJobsStatus();
            allFinishedTrialJobNum += finishedTrialJobNum;
            const requestTrialNum = this.trialConcurrencyChange + finishedTrialJobNum;
            if (requestTrialNum >= 0) {
                this.trialConcurrencyChange = 0;
            }
            else {
                this.trialConcurrencyChange = requestTrialNum;
            }
            assert_1.default(this.status.status === 'RUNNING' ||
                this.status.status === 'DONE' ||
                this.status.status === 'NO_MORE_TRIAL' ||
                this.status.status === 'TUNER_NO_MORE_TRIAL', `Actual status: ${this.status.status}`);
            if (this.experimentProfile.execDuration > this.maxDuration ||
                this.currSubmittedTrialNum >= this.maxTrialNum) {
                if (this.status.status !== 'DONE') {
                    this.setStatus('NO_MORE_TRIAL');
                    waitSubmittedToFinish = this.currSubmittedTrialNum;
                    assert_1.default(allFinishedTrialJobNum <= waitSubmittedToFinish);
                    if (allFinishedTrialJobNum >= waitSubmittedToFinish) {
                        this.setStatus('DONE');
                        this.setEndtime();
                        await this.storeExperimentProfile();
                        this.log.info('Experiment done.');
                    }
                }
            }
            else {
                this.requestTrialJobs(requestTrialNum);
                if (this.status.status === 'DONE') {
                    delete this.experimentProfile.endTime;
                    await this.storeExperimentProfile();
                }
                if (this.status.status !== 'TUNER_NO_MORE_TRIAL') {
                    this.setStatus('RUNNING');
                }
                for (let i = this.trialJobs.size; i < this.experimentProfile.params.trialConcurrency; i++) {
                    if (this.waitingTrials.length === 0 ||
                        this.currSubmittedTrialNum >= this.maxTrialNum) {
                        break;
                    }
                    const form = this.waitingTrials.shift();
                    this.currSubmittedTrialNum++;
                    this.log.info('submitTrialJob: form:', form);
                    const trialJobDetail = await this.trainingService.submitTrialJob(form);
                    const Snapshot = Object.assign({}, trialJobDetail);
                    await this.storeExperimentProfile();
                    this.trialJobs.set(trialJobDetail.id, Snapshot);
                    const trialJobDetailSnapshot = this.trialJobs.get(trialJobDetail.id);
                    if (trialJobDetailSnapshot != undefined) {
                        await this.dataStore.storeTrialJobEvent(trialJobDetailSnapshot.status, trialJobDetailSnapshot.id, form.hyperParameters.value, trialJobDetailSnapshot);
                    }
                    else {
                        assert_1.default(false, `undefined trialJobDetail in trialJobs: ${trialJobDetail.id}`);
                    }
                }
            }
            await utils_1.delay(1000 * 5);
        }
    }
    storeExperimentProfile() {
        this.experimentProfile.revision += 1;
        return this.dataStore.storeExperimentProfile(this.experimentProfile);
    }
    async run() {
        assert_1.default(this.dispatcher !== undefined);
        this.addEventListeners();
        this.sendInitTunerCommands();
        await Promise.all([
            this.periodicallyUpdateExecDuration(),
            this.pingDispatcher().catch((err) => {
                throw errors_1.NNIError.FromError(err, 'Dispatcher error: ');
            }),
            this.trainingService.run().catch((err) => {
                throw errors_1.NNIError.FromError(err, 'Training service error: ');
            }),
            this.manageTrials().catch((err) => {
                throw errors_1.NNIError.FromError(err, 'Job management error: ');
            })
        ]);
    }
    addEventListeners() {
        this.log.info('Add event listeners');
        if (this.dispatcher === undefined) {
            throw new Error('Error: tuner or job maintainer have not been setup');
        }
        this.trainingService.addTrialJobMetricListener(this.trialJobMetricListener);
        this.dispatcher.onCommand((commandType, content) => {
            this.onTunerCommand(commandType, content).catch((err) => {
                this.criticalError(errors_1.NNIError.FromError(err, 'Tuner command event error: '));
            });
        });
        this.dispatcher.onError((error) => {
            this.log.error(`Dispatcher error: ${error.message}`);
            this.criticalError(new Error('Dispatcher stream error, tuner may have crashed.'));
        });
    }
    sendInitTunerCommands() {
        if (this.dispatcher === undefined) {
            throw new Error('Dispatcher error: tuner has not been setup');
        }
        this.log.debug(`Send tuner command: INITIALIZE: ${this.experimentProfile.params.searchSpace}`);
        this.dispatcher.sendCommand(commands_1.INITIALIZE, JSON.stringify(this.experimentProfile.params.searchSpace));
    }
    async onTrialJobMetrics(metric) {
        this.log.debug('NNIManager received trial job metrics:', metric);
        if (this.trialJobs.has(metric.id)) {
            await this.dataStore.storeMetricData(metric.id, metric.data);
            if (this.dispatcher === undefined) {
                throw new Error('Error: tuner has not been setup');
            }
            this.dispatcher.sendCommand(commands_1.REPORT_METRIC_DATA, metric.data);
        }
        else {
            this.log.warning('NNIManager received non-existent trial job metrics:', metric);
        }
    }
    requestTrialJobs(jobNum) {
        if (jobNum < 1) {
            return;
        }
        if (this.dispatcher === undefined) {
            throw new Error('Dispatcher error: tuner has not been setup');
        }
        if (this.config.deprecated && this.config.deprecated.multiThread) {
            for (let i = 0; i < jobNum; i++) {
                this.dispatcher.sendCommand(commands_1.REQUEST_TRIAL_JOBS, '1');
            }
        }
        else {
            this.dispatcher.sendCommand(commands_1.REQUEST_TRIAL_JOBS, String(jobNum));
        }
    }
    async onTunerCommand(commandType, content) {
        this.log.info(`NNIManager received command from dispatcher: ${commandType}, ${content}`);
        switch (commandType) {
            case commands_1.INITIALIZED: {
                if (this.trialDataForTuner.length > 0) {
                    if (this.dispatcher === undefined) {
                        throw new Error('Dispatcher error: tuner has not been setup');
                    }
                    this.dispatcher.sendCommand(commands_1.IMPORT_DATA, this.trialDataForTuner);
                }
                this.requestTrialJobs(this.experimentProfile.params.trialConcurrency);
                break;
            }
            case commands_1.NEW_TRIAL_JOB: {
                if (this.status.status === 'TUNER_NO_MORE_TRIAL') {
                    this.log.warning('It is not supposed to receive more trials after NO_MORE_TRIAL is set');
                    this.setStatus('RUNNING');
                }
                const trialRequestContent = JSON.parse(content);
                const noneConstraint = { type: 'None', gpus: [] };
                const form = {
                    sequenceId: this.experimentProfile.nextSequenceId++,
                    hyperParameters: {
                        value: content,
                        index: 0
                    },
                    placementConstraint: trialRequestContent.placement_constraint ? trialRequestContent.placement_constraint : noneConstraint
                };
                this.waitingTrials.push(form);
                break;
            }
            case commands_1.SEND_TRIAL_JOB_PARAMETER: {
                const tunerCommand = JSON.parse(content);
                assert_1.default(tunerCommand.parameter_index >= 0);
                assert_1.default(tunerCommand.trial_job_id !== undefined);
                const trialJobForm = {
                    sequenceId: -1,
                    hyperParameters: {
                        value: content,
                        index: tunerCommand.parameter_index
                    }
                };
                this.log.info('updateTrialJob: job id:', tunerCommand.trial_job_id, 'form:', trialJobForm);
                await this.trainingService.updateTrialJob(tunerCommand.trial_job_id, trialJobForm);
                if (tunerCommand['parameters'] !== null) {
                    await this.dataStore.storeTrialJobEvent('ADD_HYPERPARAMETER', tunerCommand.trial_job_id, content, undefined);
                }
                break;
            }
            case commands_1.NO_MORE_TRIAL_JOBS: {
                if (!['ERROR', 'STOPPING', 'STOPPED'].includes(this.status.status)) {
                    this.setStatus('TUNER_NO_MORE_TRIAL');
                }
                break;
            }
            case commands_1.KILL_TRIAL_JOB: {
                this.log.info('cancelTrialJob:', content);
                await this.trainingService.cancelTrialJob(JSON.parse(content), true);
                break;
            }
            default:
                throw new Error('Error: unsupported command type from tuner');
        }
    }
    criticalError(err) {
        this.logError(err);
        console.error(err);
    }
    logError(err) {
        if (err.stack !== undefined) {
            this.log.error(err.stack);
        }
        this.status.errors.push(err.message);
        this.setEndtime();
        this.setStatus('ERROR');
    }
    setStatus(status) {
        if (status !== this.status.status) {
            this.log.info(`Change NNIManager status from: ${this.status.status} to: ${status}`);
            this.status.status = status;
            this.experimentManager.setExperimentInfo(this.experimentProfile.id, 'status', this.status.status);
        }
    }
    setEndtime() {
        this.experimentProfile.endTime = Date.now();
        this.experimentManager.setExperimentInfo(this.experimentProfile.id, 'endTime', this.experimentProfile.endTime);
    }
    async createCheckpointDir() {
        const chkpDir = utils_1.getCheckpointDir();
        await utils_1.mkDirP(chkpDir);
        return chkpDir;
    }
    async getTrialOutputLocalPath(trialJobId) {
        return this.trainingService.getTrialOutputLocalPath(trialJobId);
    }
    async fetchTrialOutput(trialJobId, subpath) {
        return this.trainingService.fetchTrialOutput(trialJobId, subpath);
    }
}
exports.NNIManager = NNIManager;
