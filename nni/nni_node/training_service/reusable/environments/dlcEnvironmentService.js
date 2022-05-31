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
exports.DlcEnvironmentService = void 0;
const fs_1 = __importDefault(require("fs"));
const path_1 = __importDefault(require("path"));
const component = __importStar(require("common/component"));
const log_1 = require("common/log");
const experimentStartupInfo_1 = require("common/experimentStartupInfo");
const dlcClient_1 = require("../dlc/dlcClient");
const dlcConfig_1 = require("../dlc/dlcConfig");
const environment_1 = require("../environment");
const fileCommandChannel_1 = require("../channels/fileCommandChannel");
const mountedStorageService_1 = require("../storages/mountedStorageService");
const typescript_ioc_1 = require("typescript-ioc");
const storageService_1 = require("../storageService");
let DlcEnvironmentService = class DlcEnvironmentService extends environment_1.EnvironmentService {
    log = log_1.getLogger('dlcEnvironmentService');
    experimentId;
    config;
    constructor(config, info) {
        super();
        this.experimentId = info.experimentId;
        this.config = config;
        component.Container.bind(storageService_1.StorageService).to(mountedStorageService_1.MountedStorageService).scope(typescript_ioc_1.Scope.Singleton);
        const storageService = component.get(storageService_1.StorageService);
        const remoteRoot = storageService.joinPath(this.config.localStorageMountPoint, 'nni-experiments', this.experimentId);
        const localRoot = storageService.joinPath(this.config.localStorageMountPoint, 'nni-experiments');
        storageService.initialize(localRoot, remoteRoot);
    }
    get hasStorageService() {
        return true;
    }
    initCommandChannel(eventEmitter) {
        this.commandChannel = new fileCommandChannel_1.FileCommandChannel(eventEmitter);
    }
    createEnvironmentInformation(envId, envName) {
        return new dlcConfig_1.DlcEnvironmentInformation(envId, envName);
    }
    get getName() {
        return 'dlc';
    }
    async refreshEnvironmentsStatus(environments) {
        environments.forEach(async (environment) => {
            const dlcClient = environment.dlcClient;
            if (!dlcClient) {
                return Promise.reject('DLC client not initialized!');
            }
            const newStatus = await dlcClient.updateStatus(environment.status);
            switch (newStatus.toUpperCase()) {
                case 'CREATING':
                case 'CREATED':
                case 'WAITING':
                case 'QUEUED':
                    environment.setStatus('WAITING');
                    break;
                case 'RUNNING':
                    environment.setStatus('RUNNING');
                    break;
                case 'COMPLETED':
                case 'SUCCEEDED':
                    environment.setStatus('SUCCEEDED');
                    break;
                case 'FAILED':
                    environment.setStatus('FAILED');
                    return Promise.reject(`DLC: job ${environment.envId} is failed!`);
                case 'STOPPED':
                case 'STOPPING':
                    environment.setStatus('USER_CANCELED');
                    break;
                default:
                    environment.setStatus('UNKNOWN');
            }
        });
    }
    async startEnvironment(environment) {
        const dlcEnvironment = environment;
        const environmentRoot = path_1.default.join(this.config.containerStorageMountPoint, `/nni-experiments/${this.experimentId}`);
        const localRoot = path_1.default.join(this.config.localStorageMountPoint, `/nni-experiments/${this.experimentId}`);
        dlcEnvironment.workingFolder = `${localRoot}/envs/${environment.id}`;
        dlcEnvironment.runnerWorkingFolder = `${environmentRoot}/envs/${environment.id}`;
        if (!fs_1.default.existsSync(`${dlcEnvironment.workingFolder}/commands`)) {
            await fs_1.default.promises.mkdir(`${dlcEnvironment.workingFolder}/commands`, { recursive: true });
        }
        environment.command = `cd ${environmentRoot} && ${environment.command} 1>${environment.runnerWorkingFolder}/trialrunner_stdout 2>${environment.runnerWorkingFolder}/trialrunner_stderr`;
        const dlcClient = new dlcClient_1.DlcClient(this.config.type, this.config.image, this.config.jobType, this.config.podCount, this.experimentId, environment.id, this.config.ecsSpec, this.config.region, this.config.nasDataSourceId, this.config.accessKeyId, this.config.accessKeySecret, environment.command, dlcEnvironment.workingFolder, this.config.ossDataSourceId);
        dlcEnvironment.id = await dlcClient.submit();
        this.log.debug('dlc: before getTrackingUrl');
        dlcEnvironment.trackingUrl = await dlcClient.getTrackingUrl();
        this.log.debug(`dlc trackingUrl: ${dlcEnvironment.trackingUrl}`);
        dlcEnvironment.dlcClient = dlcClient;
    }
    async stopEnvironment(environment) {
        const dlcEnvironment = environment;
        const dlcClient = dlcEnvironment.dlcClient;
        if (!dlcClient) {
            throw new Error('DLC client not initialized!');
        }
        dlcClient.stop();
    }
};
DlcEnvironmentService = __decorate([
    component.Singleton,
    __metadata("design:paramtypes", [Object, experimentStartupInfo_1.ExperimentStartupInfo])
], DlcEnvironmentService);
exports.DlcEnvironmentService = DlcEnvironmentService;
