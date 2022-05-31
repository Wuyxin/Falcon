"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.RunnerSettings = exports.NodeInformation = exports.EnvironmentService = exports.EnvironmentInformation = exports.TrialGpuSummary = void 0;
const log_1 = require("common/log");
const webCommandChannel_1 = require("./channels/webCommandChannel");
class TrialGpuSummary {
    gpuCount;
    timestamp;
    gpuInfos;
    assignedGpuIndexMap = new Map();
    constructor(gpuCount, timestamp, gpuInfos) {
        this.gpuCount = gpuCount;
        this.timestamp = timestamp;
        this.gpuInfos = gpuInfos;
    }
}
exports.TrialGpuSummary = TrialGpuSummary;
class EnvironmentInformation {
    defaultNodeId = "default";
    log;
    isNoGpuWarned = false;
    isAlive = true;
    isRunnerReady = false;
    status = "UNKNOWN";
    runningTrialCount = 0;
    assignedTrialCount = 0;
    latestTrialReleasedTime = -1;
    id;
    envId;
    name;
    trackingUrl = "";
    workingFolder = "";
    runnerWorkingFolder = "";
    command = "";
    nodeCount = 1;
    nodes;
    gpuSummaries = new Map();
    usableGpus;
    maxTrialNumberPerGpu;
    useActiveGpu;
    environmentService;
    useSharedStorage;
    constructor(id, name, envId) {
        this.log = log_1.getLogger('EnvironmentInformation');
        this.id = id;
        this.name = name;
        this.envId = envId ? envId : name;
        this.nodes = new Map();
    }
    setStatus(status) {
        if (this.status !== status) {
            this.log.info(`EnvironmentInformation: ${this.envId} change status from ${this.status} to ${status}.`);
            this.status = status;
        }
    }
    setGpuSummary(nodeId, newGpuSummary) {
        if (nodeId === null || nodeId === undefined) {
            nodeId = this.defaultNodeId;
        }
        const originalGpuSummary = this.gpuSummaries.get(nodeId);
        if (undefined === originalGpuSummary) {
            newGpuSummary.assignedGpuIndexMap = new Map();
            this.gpuSummaries.set(nodeId, newGpuSummary);
        }
        else {
            originalGpuSummary.gpuCount = newGpuSummary.gpuCount;
            originalGpuSummary.timestamp = newGpuSummary.timestamp;
            originalGpuSummary.gpuInfos = newGpuSummary.gpuInfos;
        }
    }
    get defaultGpuSummary() {
        const gpuSummary = this.gpuSummaries.get(this.defaultNodeId);
        if (gpuSummary === undefined) {
            if (false === this.isNoGpuWarned) {
                this.log.warning(`EnvironmentInformation: ${this.envId} no default gpu found. current gpu info`, this.gpuSummaries);
                this.isNoGpuWarned = true;
            }
        }
        else {
            this.isNoGpuWarned = false;
        }
        return gpuSummary;
    }
}
exports.EnvironmentInformation = EnvironmentInformation;
class EnvironmentService {
    async init() {
        return;
    }
    commandChannel;
    get prefetchedEnvironmentCount() {
        return 0;
    }
    initCommandChannel(eventEmitter) {
        this.commandChannel = webCommandChannel_1.WebCommandChannel.getInstance(eventEmitter);
    }
    get getCommandChannel() {
        if (this.commandChannel === undefined) {
            throw new Error("Command channel not initialized!");
        }
        return this.commandChannel;
    }
    get environmentMaintenceLoopInterval() {
        return 5000;
    }
    get hasMoreEnvironments() {
        return true;
    }
    createEnvironmentInformation(envId, envName) {
        return new EnvironmentInformation(envId, envName);
    }
}
exports.EnvironmentService = EnvironmentService;
class NodeInformation {
    id;
    status = "UNKNOWN";
    endTime;
    constructor(id) {
        this.id = id;
    }
}
exports.NodeInformation = NodeInformation;
class RunnerSettings {
    experimentId = "";
    platform = "";
    nniManagerIP = "";
    nniManagerPort = 8081;
    nniManagerVersion = "";
    logCollection = "none";
    command = "";
    enableGpuCollector = true;
    commandChannel = "file";
}
exports.RunnerSettings = RunnerSettings;
