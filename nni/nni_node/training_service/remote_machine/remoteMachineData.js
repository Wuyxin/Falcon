"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.ExecutorManager = exports.RemoteMachineTrialJobDetail = exports.RemoteCommandResult = exports.RemoteMachineMeta = void 0;
const shellExecutor_1 = require("./shellExecutor");
class RemoteMachineMeta {
    config;
    gpuSummary;
    occupiedGpuIndexMap;
    constructor(config) {
        this.config = config;
        this.occupiedGpuIndexMap = new Map();
    }
}
exports.RemoteMachineMeta = RemoteMachineMeta;
class RemoteCommandResult {
    stdout;
    stderr;
    exitCode;
    constructor(stdout, stderr, exitCode) {
        this.stdout = stdout;
        this.stderr = stderr;
        this.exitCode = exitCode;
    }
}
exports.RemoteCommandResult = RemoteCommandResult;
class RemoteMachineTrialJobDetail {
    id;
    status;
    submitTime;
    startTime;
    endTime;
    tags;
    url;
    workingDirectory;
    form;
    rmMeta;
    isEarlyStopped;
    gpuIndices;
    constructor(id, status, submitTime, workingDirectory, form) {
        this.id = id;
        this.status = status;
        this.submitTime = submitTime;
        this.workingDirectory = workingDirectory;
        this.form = form;
        this.tags = [];
        this.gpuIndices = [];
    }
}
exports.RemoteMachineTrialJobDetail = RemoteMachineTrialJobDetail;
class ExecutorManager {
    rmMeta;
    executorMap = new Map();
    executors = [];
    constructor(config) {
        this.rmMeta = new RemoteMachineMeta(config);
    }
    async getExecutor(id) {
        let isFound = false;
        let executor;
        if (this.executorMap.has(id)) {
            executor = this.executorMap.get(id);
            if (executor === undefined) {
                throw new Error("executor shouldn't be undefined before return!");
            }
            return executor;
        }
        for (const candidateExecutor of this.executors) {
            if (candidateExecutor.addUsage()) {
                isFound = true;
                executor = candidateExecutor;
                break;
            }
        }
        if (!isFound) {
            executor = await this.createShellExecutor();
        }
        if (executor === undefined) {
            throw new Error("executor shouldn't be undefined before set!");
        }
        this.executorMap.set(id, executor);
        return executor;
    }
    releaseAllExecutor() {
        this.executorMap.clear();
        for (const executor of this.executors) {
            executor.close();
        }
        this.executors = [];
    }
    releaseExecutor(id) {
        const executor = this.executorMap.get(id);
        if (executor === undefined) {
            throw new Error(`executor for ${id} is not found`);
        }
        executor.releaseUsage();
        this.executorMap.delete(id);
    }
    async createShellExecutor() {
        const executor = new shellExecutor_1.ShellExecutor();
        await executor.initialize(this.rmMeta);
        if (!executor.addUsage()) {
            throw new Error("failed to add usage on new created Executor! It's a wired bug!");
        }
        this.executors.push(executor);
        return executor;
    }
}
exports.ExecutorManager = ExecutorManager;
