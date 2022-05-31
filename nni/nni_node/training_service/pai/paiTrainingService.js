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
exports.PAITrainingService = void 0;
const fs_1 = __importDefault(require("fs"));
const path_1 = __importDefault(require("path"));
const request_1 = __importDefault(require("request"));
const component = __importStar(require("common/component"));
const events_1 = require("events");
const ts_deferred_1 = require("ts-deferred");
const experimentStartupInfo_1 = require("common/experimentStartupInfo");
const log_1 = require("common/log");
const errors_1 = require("common/errors");
const utils_1 = require("common/utils");
const experimentConfig_1 = require("common/experimentConfig");
const paiJobInfoCollector_1 = require("./paiJobInfoCollector");
const paiJobRestServer_1 = require("./paiJobRestServer");
const paiConfig_1 = require("./paiConfig");
const typescript_string_operations_1 = require("typescript-string-operations");
const utils_2 = require("common/utils");
const containerJobData_1 = require("../common/containerJobData");
const util_1 = require("../common/util");
const yaml = require('js-yaml');
let PAITrainingService = class PAITrainingService {
    log;
    metricsEmitter;
    trialJobsMap;
    expRootDir;
    jobQueue;
    stopping = false;
    paiToken;
    paiTokenUpdateTime;
    paiTokenUpdateInterval;
    experimentId;
    paiJobCollector;
    paiRestServerPort;
    nniManagerIpConfig;
    versionCheck = true;
    logCollection = 'none';
    paiJobRestServer;
    protocol;
    copyExpCodeDirPromise;
    paiJobConfig;
    nniVersion;
    config;
    constructor(config) {
        this.log = log_1.getLogger('PAITrainingService');
        this.metricsEmitter = new events_1.EventEmitter();
        this.trialJobsMap = new Map();
        this.jobQueue = [];
        this.expRootDir = path_1.default.join('/nni-experiments', experimentStartupInfo_1.getExperimentId());
        this.experimentId = experimentStartupInfo_1.getExperimentId();
        this.paiJobCollector = new paiJobInfoCollector_1.PAIJobInfoCollector(this.trialJobsMap);
        this.paiTokenUpdateInterval = 7200000;
        this.log.info('Construct paiBase training service.');
        this.config = config;
        this.versionCheck = !this.config.debug;
        this.paiJobRestServer = new paiJobRestServer_1.PAIJobRestServer(this);
        this.paiToken = this.config.token;
        this.protocol = this.config.host.toLowerCase().startsWith('https://') ? 'https' : 'http';
        this.copyExpCodeDirPromise = this.copyTrialCode();
    }
    async copyTrialCode() {
        await util_1.validateCodeDir(this.config.trialCodeDirectory);
        const nniManagerNFSExpCodeDir = path_1.default.join(this.config.localStorageMountPoint, this.experimentId, 'nni-code');
        await util_1.execMkdir(nniManagerNFSExpCodeDir);
        this.log.info(`Starting copy codeDir data from ${this.config.trialCodeDirectory} to ${nniManagerNFSExpCodeDir}`);
        await util_1.execCopydir(this.config.trialCodeDirectory, nniManagerNFSExpCodeDir);
    }
    async run() {
        this.log.info('Run PAI training service.');
        if (this.paiJobRestServer === undefined) {
            throw new Error('paiJobRestServer not initialized!');
        }
        await this.paiJobRestServer.start();
        this.paiJobRestServer.setEnableVersionCheck = this.versionCheck;
        this.log.info(`PAI Training service rest server listening on: ${this.paiJobRestServer.endPoint}`);
        await Promise.all([
            this.statusCheckingLoop(),
            this.submitJobLoop()
        ]);
        this.log.info('PAI training service exit.');
    }
    async submitJobLoop() {
        while (!this.stopping) {
            while (!this.stopping && this.jobQueue.length > 0) {
                const trialJobId = this.jobQueue[0];
                if (await this.submitTrialJobToPAI(trialJobId)) {
                    this.jobQueue.shift();
                }
                else {
                    break;
                }
            }
            await utils_1.delay(3000);
        }
    }
    async listTrialJobs() {
        const jobs = [];
        for (const key of this.trialJobsMap.keys()) {
            jobs.push(await this.getTrialJob(key));
        }
        return jobs;
    }
    async getTrialFile(_trialJobId, _fileName) {
        throw new errors_1.MethodNotImplementedError();
    }
    async getTrialJob(trialJobId) {
        const paiTrialJob = this.trialJobsMap.get(trialJobId);
        if (paiTrialJob === undefined) {
            throw new Error(`trial job ${trialJobId} not found`);
        }
        return paiTrialJob;
    }
    addTrialJobMetricListener(listener) {
        this.metricsEmitter.on('metric', listener);
    }
    removeTrialJobMetricListener(listener) {
        this.metricsEmitter.off('metric', listener);
    }
    cancelTrialJob(trialJobId, isEarlyStopped = false) {
        const trialJobDetail = this.trialJobsMap.get(trialJobId);
        if (trialJobDetail === undefined) {
            return Promise.reject(new Error(`cancelTrialJob: trial job id ${trialJobId} not found`));
        }
        if (trialJobDetail.status === 'UNKNOWN') {
            trialJobDetail.status = 'USER_CANCELED';
            return Promise.resolve();
        }
        const stopJobRequest = {
            uri: `${this.config.host}/rest-server/api/v2/jobs/${this.config.username}~${trialJobDetail.paiJobName}/executionType`,
            method: 'PUT',
            json: true,
            body: { value: 'STOP' },
            headers: {
                'Content-Type': 'application/json',
                Authorization: `Bearer ${this.paiToken}`
            }
        };
        trialJobDetail.isEarlyStopped = isEarlyStopped;
        const deferred = new ts_deferred_1.Deferred();
        request_1.default(stopJobRequest, (error, response, _body) => {
            if ((error !== undefined && error !== null) || response.statusCode >= 400) {
                this.log.error(`PAI Training service: stop trial ${trialJobId} to PAI Cluster failed!`);
                deferred.reject((error !== undefined && error !== null) ? error.message :
                    `Stop trial failed, http code: ${response.statusCode}`);
            }
            else {
                deferred.resolve();
            }
        });
        return deferred.promise;
    }
    async cleanUp() {
        this.log.info('Stopping PAI training service...');
        this.stopping = true;
        if (this.paiJobRestServer === undefined) {
            throw new Error('paiJobRestServer not initialized!');
        }
        try {
            await this.paiJobRestServer.stop();
            this.log.info('PAI Training service rest server stopped successfully.');
        }
        catch (error) {
            this.log.error(`PAI Training service rest server stopped failed, error: ${error.message}`);
        }
    }
    get MetricsEmitter() {
        return this.metricsEmitter;
    }
    formatPAIHost(host) {
        if (host.startsWith('http://')) {
            this.protocol = 'http';
            return host.replace('http://', '');
        }
        else if (host.startsWith('https://')) {
            this.protocol = 'https';
            return host.replace('https://', '');
        }
        else {
            return host;
        }
    }
    async statusCheckingLoop() {
        while (!this.stopping) {
            await this.paiJobCollector.retrieveTrialStatus(this.protocol, this.paiToken, this.config);
            if (this.paiJobRestServer === undefined) {
                throw new Error('paiBaseJobRestServer not implemented!');
            }
            if (this.paiJobRestServer.getErrorMessage !== undefined) {
                throw new Error(this.paiJobRestServer.getErrorMessage);
            }
            await utils_1.delay(3000);
        }
    }
    async setClusterMetadata(_key, _value) { return; }
    async getClusterMetadata(_key) { return ''; }
    async updateTrialJob(trialJobId, form) {
        const trialJobDetail = this.trialJobsMap.get(trialJobId);
        if (trialJobDetail === undefined) {
            throw new Error(`updateTrialJob failed: ${trialJobId} not found`);
        }
        await this.writeParameterFile(trialJobDetail.logPath, form.hyperParameters);
        return trialJobDetail;
    }
    async submitTrialJob(form) {
        this.log.info('submitTrialJob: form:', form);
        const trialJobId = utils_2.uniqueString(5);
        const trialWorkingFolder = path_1.default.join(this.expRootDir, 'trials', trialJobId);
        const paiJobName = `nni_exp_${this.experimentId}_trial_${trialJobId}`;
        const logPath = path_1.default.join(this.config.localStorageMountPoint, this.experimentId, trialJobId);
        const paiJobDetailUrl = `${this.config.host}/job-detail.html?username=${this.config.username}&jobName=${paiJobName}`;
        const trialJobDetail = new paiConfig_1.PAITrialJobDetail(trialJobId, 'WAITING', paiJobName, Date.now(), trialWorkingFolder, form, logPath, paiJobDetailUrl);
        this.trialJobsMap.set(trialJobId, trialJobDetail);
        this.jobQueue.push(trialJobId);
        return trialJobDetail;
    }
    async generateNNITrialCommand(trialJobDetail, command) {
        const containerNFSExpCodeDir = `${this.config.containerStorageMountPoint}/${this.experimentId}/nni-code`;
        const containerWorkingDir = `${this.config.containerStorageMountPoint}/${this.experimentId}/${trialJobDetail.id}`;
        const nniPaiTrialCommand = typescript_string_operations_1.String.Format(paiConfig_1.PAI_TRIAL_COMMAND_FORMAT, `${containerWorkingDir}`, `${containerWorkingDir}/nnioutput`, trialJobDetail.id, this.experimentId, trialJobDetail.form.sequenceId, false, containerNFSExpCodeDir, command, this.config.nniManagerIp || await utils_2.getIPV4Address(), this.paiRestServerPort, this.nniVersion, this.logCollection)
            .replace(/\r\n|\n|\r/gm, '');
        return nniPaiTrialCommand;
    }
    async generateJobConfigInYamlFormat(trialJobDetail) {
        const jobName = `nni_exp_${this.experimentId}_trial_${trialJobDetail.id}`;
        let nniJobConfig = undefined;
        if (this.config.openpaiConfig !== undefined) {
            nniJobConfig = JSON.parse(JSON.stringify(this.config.openpaiConfig));
            nniJobConfig.name = jobName;
            for (const taskRoleIndex in nniJobConfig.taskRoles) {
                const commands = nniJobConfig.taskRoles[taskRoleIndex].commands;
                const nniTrialCommand = await this.generateNNITrialCommand(trialJobDetail, commands.join(" && ").replace(/(["'$`\\])/g, '\\$1'));
                nniJobConfig.taskRoles[taskRoleIndex].commands = [nniTrialCommand];
            }
        }
        else {
            nniJobConfig = {
                protocolVersion: 2,
                name: jobName,
                type: 'job',
                jobRetryCount: 0,
                prerequisites: [
                    {
                        type: 'dockerimage',
                        uri: this.config.dockerImage,
                        name: 'docker_image_0'
                    }
                ],
                taskRoles: {
                    taskrole: {
                        instances: 1,
                        completion: {
                            minFailedInstances: 1,
                            minSucceededInstances: -1
                        },
                        taskRetryCount: 0,
                        dockerImage: 'docker_image_0',
                        resourcePerInstance: {
                            gpu: this.config.trialGpuNumber,
                            cpu: this.config.trialCpuNumber,
                            memoryMB: experimentConfig_1.toMegaBytes(this.config.trialMemorySize)
                        },
                        commands: [
                            await this.generateNNITrialCommand(trialJobDetail, this.config.trialCommand)
                        ]
                    }
                },
                extras: {
                    'storages': [
                        {
                            name: this.config.storageConfigName
                        }
                    ],
                    submitFrom: 'submit-job-v2'
                }
            };
            if (this.config.virtualCluster) {
                nniJobConfig.defaults = {
                    virtualCluster: this.config.virtualCluster
                };
            }
        }
        return yaml.safeDump(nniJobConfig);
    }
    async submitTrialJobToPAI(trialJobId) {
        const deferred = new ts_deferred_1.Deferred();
        const trialJobDetail = this.trialJobsMap.get(trialJobId);
        if (trialJobDetail === undefined) {
            throw new Error(`Failed to find PAITrialJobDetail for job ${trialJobId}`);
        }
        if (this.paiJobRestServer === undefined) {
            throw new Error('paiJobRestServer is not initialized');
        }
        if (this.copyExpCodeDirPromise !== undefined) {
            await this.copyExpCodeDirPromise;
            this.log.info(`Copy codeDir data finished.`);
            this.copyExpCodeDirPromise = undefined;
        }
        this.paiRestServerPort = this.paiJobRestServer.clusterRestServerPort;
        await util_1.execMkdir(trialJobDetail.logPath);
        await fs_1.default.promises.writeFile(path_1.default.join(trialJobDetail.logPath, 'install_nni.sh'), containerJobData_1.CONTAINER_INSTALL_NNI_SHELL_FORMAT, { encoding: 'utf8' });
        if (trialJobDetail.form !== undefined) {
            await this.writeParameterFile(trialJobDetail.logPath, trialJobDetail.form.hyperParameters);
        }
        const paiJobConfig = await this.generateJobConfigInYamlFormat(trialJobDetail);
        this.log.debug(paiJobConfig);
        const submitJobRequest = {
            uri: `${this.config.host}/rest-server/api/v2/jobs`,
            method: 'POST',
            body: paiJobConfig,
            followAllRedirects: true,
            headers: {
                'Content-Type': 'text/yaml',
                Authorization: `Bearer ${this.paiToken}`
            }
        };
        request_1.default(submitJobRequest, (error, response, body) => {
            if ((error !== undefined && error !== null) || response.statusCode >= 400) {
                const errorMessage = (error !== undefined && error !== null) ? error.message :
                    `Submit trial ${trialJobId} failed, http code:${response.statusCode}, http body: ${body}`;
                this.log.error(errorMessage);
                trialJobDetail.status = 'FAILED';
                deferred.reject(errorMessage);
            }
            else {
                trialJobDetail.submitTime = Date.now();
            }
            deferred.resolve(true);
        });
        return deferred.promise;
    }
    async writeParameterFile(directory, hyperParameters) {
        const filepath = path_1.default.join(directory, utils_2.generateParamFileName(hyperParameters));
        await fs_1.default.promises.writeFile(filepath, hyperParameters.value, { encoding: 'utf8' });
    }
    getTrialOutputLocalPath(_trialJobId) {
        throw new errors_1.MethodNotImplementedError();
    }
    fetchTrialOutput(_trialJobId, _subpath) {
        throw new errors_1.MethodNotImplementedError();
    }
};
PAITrainingService = __decorate([
    component.Singleton,
    __metadata("design:paramtypes", [Object])
], PAITrainingService);
exports.PAITrainingService = PAITrainingService;
