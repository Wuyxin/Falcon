"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.RouterTrainingService = void 0;
const log_1 = require("common/log");
const errors_1 = require("common/errors");
const utils_1 = require("common/utils");
const paiTrainingService_1 = require("../pai/paiTrainingService");
const remoteMachineTrainingService_1 = require("../remote_machine/remoteMachineTrainingService");
const kubeflowTrainingService_1 = require("../kubernetes/kubeflow/kubeflowTrainingService");
const frameworkcontrollerTrainingService_1 = require("../kubernetes/frameworkcontroller/frameworkcontrollerTrainingService");
const trialDispatcher_1 = require("./trialDispatcher");
class RouterTrainingService {
    log;
    internalTrainingService;
    static async construct(config) {
        const instance = new RouterTrainingService();
        instance.log = log_1.getLogger('RouterTrainingService');
        const platform = Array.isArray(config.trainingService) ? 'hybrid' : config.trainingService.platform;
        if (platform === 'remote' && config.trainingService.reuseMode === false) {
            instance.internalTrainingService = new remoteMachineTrainingService_1.RemoteMachineTrainingService(config.trainingService);
        }
        else if (platform === 'openpai' && config.trainingService.reuseMode === false) {
            instance.internalTrainingService = new paiTrainingService_1.PAITrainingService(config.trainingService);
        }
        else if (platform === 'kubeflow' && config.trainingService.reuseMode === false) {
            instance.internalTrainingService = new kubeflowTrainingService_1.KubeflowTrainingService();
        }
        else if (platform === 'frameworkcontroller' && config.trainingService.reuseMode === false) {
            instance.internalTrainingService = new frameworkcontrollerTrainingService_1.FrameworkControllerTrainingService();
        }
        else {
            instance.internalTrainingService = await trialDispatcher_1.TrialDispatcher.construct(config);
        }
        return instance;
    }
    constructor() { }
    async listTrialJobs() {
        if (this.internalTrainingService === undefined) {
            throw new Error("TrainingService is not assigned!");
        }
        return await this.internalTrainingService.listTrialJobs();
    }
    async getTrialJob(trialJobId) {
        if (this.internalTrainingService === undefined) {
            throw new Error("TrainingService is not assigned!");
        }
        return await this.internalTrainingService.getTrialJob(trialJobId);
    }
    async getTrialFile(_trialJobId, _fileName) {
        throw new errors_1.MethodNotImplementedError();
    }
    addTrialJobMetricListener(listener) {
        if (this.internalTrainingService === undefined) {
            throw new Error("TrainingService is not assigned!");
        }
        this.internalTrainingService.addTrialJobMetricListener(listener);
    }
    removeTrialJobMetricListener(listener) {
        if (this.internalTrainingService === undefined) {
            throw new Error("TrainingService is not assigned!");
        }
        this.internalTrainingService.removeTrialJobMetricListener(listener);
    }
    async submitTrialJob(form) {
        if (this.internalTrainingService === undefined) {
            throw new Error("TrainingService is not assigned!");
        }
        return await this.internalTrainingService.submitTrialJob(form);
    }
    async updateTrialJob(trialJobId, form) {
        if (this.internalTrainingService === undefined) {
            throw new Error("TrainingService is not assigned!");
        }
        return await this.internalTrainingService.updateTrialJob(trialJobId, form);
    }
    async cancelTrialJob(trialJobId, isEarlyStopped) {
        if (this.internalTrainingService === undefined) {
            throw new Error("TrainingService is not assigned!");
        }
        await this.internalTrainingService.cancelTrialJob(trialJobId, isEarlyStopped);
    }
    async setClusterMetadata(_key, _value) { return; }
    async getClusterMetadata(_key) { return ''; }
    async cleanUp() {
        if (this.internalTrainingService === undefined) {
            throw new Error("TrainingService is not assigned!");
        }
        await this.internalTrainingService.cleanUp();
    }
    async run() {
        while (this.internalTrainingService === undefined) {
            await utils_1.delay(100);
        }
        return await this.internalTrainingService.run();
    }
    async getTrialOutputLocalPath(trialJobId) {
        if (this.internalTrainingService === undefined) {
            throw new Error("TrainingService is not assigned!");
        }
        return this.internalTrainingService.getTrialOutputLocalPath(trialJobId);
    }
    async fetchTrialOutput(trialJobId, subpath) {
        if (this.internalTrainingService === undefined) {
            throw new Error("TrainingService is not assigned!");
        }
        return this.internalTrainingService.fetchTrialOutput(trialJobId, subpath);
    }
}
exports.RouterTrainingService = RouterTrainingService;
