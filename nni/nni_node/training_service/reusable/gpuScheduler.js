"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.GpuScheduler = exports.GpuSchedulerSetting = void 0;
const assert_1 = __importDefault(require("assert"));
const log_1 = require("common/log");
const utils_1 = require("common/utils");
const gpuData_1 = require("../common/gpuData");
class GpuSchedulerSetting {
    useActiveGpu = false;
    maxTrialNumberPerGpu = 1;
}
exports.GpuSchedulerSetting = GpuSchedulerSetting;
class GpuScheduler {
    log = log_1.getLogger('GpuScheduler');
    policyName = 'recently-idle';
    defaultSetting;
    roundRobinIndex = 0;
    constructor(gpuSchedulerSetting = undefined) {
        if (undefined === gpuSchedulerSetting) {
            gpuSchedulerSetting = new GpuSchedulerSetting();
        }
        this.defaultSetting = gpuSchedulerSetting;
    }
    setSettings(gpuSchedulerSetting) {
        this.defaultSetting = gpuSchedulerSetting;
    }
    scheduleMachine(environments, constraint, defaultRequiredGPUNum, trialDetail) {
        if (constraint.type == 'None' || constraint.type == 'GPUNumber') {
            let requiredGPUNum = 0;
            if (constraint.type == 'None') {
                if (defaultRequiredGPUNum === undefined) {
                    requiredGPUNum = 0;
                }
                else {
                    requiredGPUNum = defaultRequiredGPUNum;
                }
            }
            else if (constraint.type == 'GPUNumber') {
                const gpus = constraint.gpus;
                if (gpus.length != 1) {
                    throw new Error("Placement constraint of GPUNumber must have exactly one number.");
                }
                requiredGPUNum = gpus[0];
            }
            assert_1.default(requiredGPUNum >= 0);
            const eligibleEnvironments = environments.filter((environment) => environment.defaultGpuSummary === undefined || requiredGPUNum === 0 ||
                (requiredGPUNum !== undefined && environment.defaultGpuSummary.gpuCount >= requiredGPUNum));
            if (eligibleEnvironments.length === 0) {
                return ({
                    resultType: gpuData_1.ScheduleResultType.REQUIRE_EXCEED_TOTAL,
                    gpuIndices: undefined,
                    environment: undefined,
                });
            }
            if (requiredGPUNum > 0) {
                const result = this.scheduleGPUHost(environments, requiredGPUNum, trialDetail);
                if (result !== undefined) {
                    return result;
                }
            }
            else {
                const allocatedRm = this.selectMachine(environments, environments);
                return this.allocateHost(requiredGPUNum, allocatedRm, [], trialDetail);
            }
            return {
                resultType: gpuData_1.ScheduleResultType.TMP_NO_AVAILABLE_GPU,
                gpuIndices: undefined,
                environment: undefined,
            };
        }
        else {
            assert_1.default(constraint.type === 'Device');
            if (constraint.gpus.length == 0) {
                throw new Error("Device constraint is used but no device is specified.");
            }
            const gpus = constraint.gpus;
            const selectedHost = gpus[0][0];
            const differentHosts = gpus.filter((gpuTuple) => gpuTuple[0] != selectedHost);
            if (differentHosts.length >= 1) {
                throw new Error("Device constraint does not support using multiple hosts");
            }
            if (environments.length == 0) {
                return {
                    resultType: gpuData_1.ScheduleResultType.TMP_NO_AVAILABLE_GPU,
                    gpuIndices: undefined,
                    environment: undefined,
                };
            }
            for (const environment of environments) {
                if (!('rmMachineMeta' in environment)) {
                    throw new Error(`Environment Device placement constraint only supports remote training service for now.`);
                }
            }
            const eligibleEnvironments = environments.filter((environment) => environment.rmMachineMeta != undefined &&
                environment.rmMachineMeta?.host == selectedHost);
            if (eligibleEnvironments.length === 0) {
                throw new Error(`The the required host (host: ${selectedHost}) is not found.`);
            }
            const selectedEnvironment = eligibleEnvironments[0];
            const availableResources = this.gpuResourceDetection([selectedEnvironment]);
            const selectedGPUs = [];
            if (selectedEnvironment.defaultGpuSummary === undefined) {
                return {
                    resultType: gpuData_1.ScheduleResultType.TMP_NO_AVAILABLE_GPU,
                    gpuIndices: undefined,
                    environment: undefined,
                };
            }
            for (const gpuTuple of gpus) {
                const gpuIdx = gpuTuple[1];
                if (gpuIdx >= selectedEnvironment.defaultGpuSummary.gpuCount) {
                    throw new Error(`The gpuIdx of placement constraint ${gpuIdx} exceeds gpuCount of the host ${selectedHost}`);
                }
                if (availableResources.has(selectedEnvironment)) {
                    for (const gpuInfo of availableResources.get(selectedEnvironment)) {
                        if (gpuInfo.index === gpuIdx) {
                            selectedGPUs.push(gpuInfo);
                        }
                    }
                }
            }
            if (selectedGPUs.length === constraint.gpus.length) {
                for (const gpuInfo of selectedGPUs) {
                    let num = selectedEnvironment.defaultGpuSummary?.assignedGpuIndexMap.get(gpuInfo.index);
                    if (num === undefined) {
                        num = 0;
                    }
                    selectedEnvironment.defaultGpuSummary?.assignedGpuIndexMap.set(gpuInfo.index, num + 1);
                }
                return {
                    resultType: gpuData_1.ScheduleResultType.SUCCEED,
                    environment: selectedEnvironment,
                    gpuIndices: selectedGPUs,
                };
            }
            else {
                return {
                    resultType: gpuData_1.ScheduleResultType.TMP_NO_AVAILABLE_GPU,
                    gpuIndices: undefined,
                    environment: undefined,
                };
            }
        }
    }
    removeGpuReservation(trial) {
        if (trial.environment !== undefined &&
            trial.environment.defaultGpuSummary !== undefined &&
            trial.assignedGpus !== undefined &&
            trial.assignedGpus.length > 0) {
            for (const gpuInfo of trial.assignedGpus) {
                const defaultGpuSummary = trial.environment.defaultGpuSummary;
                const num = defaultGpuSummary.assignedGpuIndexMap.get(gpuInfo.index);
                if (num !== undefined) {
                    if (num === 1) {
                        defaultGpuSummary.assignedGpuIndexMap.delete(gpuInfo.index);
                    }
                    else {
                        defaultGpuSummary.assignedGpuIndexMap.set(gpuInfo.index, num - 1);
                    }
                }
            }
        }
    }
    scheduleGPUHost(environments, requiredGPUNumber, trial) {
        const totalResourceMap = this.gpuResourceDetection(environments);
        const qualifiedEnvironments = [];
        totalResourceMap.forEach((gpuInfos, environment) => {
            if (gpuInfos !== undefined && gpuInfos.length >= requiredGPUNumber) {
                qualifiedEnvironments.push(environment);
            }
        });
        if (qualifiedEnvironments.length > 0) {
            const allocatedEnvironment = this.selectMachine(qualifiedEnvironments, environments);
            const gpuInfos = totalResourceMap.get(allocatedEnvironment);
            if (gpuInfos !== undefined) {
                return this.allocateHost(requiredGPUNumber, allocatedEnvironment, gpuInfos, trial);
            }
            else {
                assert_1.default(false, 'gpuInfos is undefined');
            }
        }
        return undefined;
    }
    gpuResourceDetection(environments) {
        const totalResourceMap = new Map();
        environments.forEach((environment) => {
            if (environment.defaultGpuSummary !== undefined) {
                const defaultGpuSummary = environment.defaultGpuSummary;
                const availableGPUs = [];
                const designatedGpuIndices = new Set(environment.usableGpus);
                if (designatedGpuIndices.size > 0) {
                    for (const gpuIndex of designatedGpuIndices) {
                        if (gpuIndex >= environment.defaultGpuSummary.gpuCount) {
                            throw new Error(`Specified GPU index not found: ${gpuIndex}`);
                        }
                    }
                }
                if (undefined !== defaultGpuSummary.gpuInfos) {
                    defaultGpuSummary.gpuInfos.forEach((gpuInfo) => {
                        if (designatedGpuIndices.size === 0 || designatedGpuIndices.has(gpuInfo.index)) {
                            if (defaultGpuSummary.assignedGpuIndexMap !== undefined) {
                                const num = defaultGpuSummary.assignedGpuIndexMap.get(gpuInfo.index);
                                const maxTrialNumberPerGpu = environment.maxTrialNumberPerGpu ? environment.maxTrialNumberPerGpu : this.defaultSetting.maxTrialNumberPerGpu;
                                const useActiveGpu = environment.useActiveGpu ? environment.useActiveGpu : this.defaultSetting.useActiveGpu;
                                if ((num === undefined && (!useActiveGpu && gpuInfo.activeProcessNum === 0 || useActiveGpu)) ||
                                    (num !== undefined && num < maxTrialNumberPerGpu)) {
                                    availableGPUs.push(gpuInfo);
                                }
                            }
                            else {
                                throw new Error(`occupiedGpuIndexMap is undefined!`);
                            }
                        }
                    });
                }
                totalResourceMap.set(environment, availableGPUs);
            }
        });
        return totalResourceMap;
    }
    selectMachine(qualifiedEnvironments, allEnvironments) {
        assert_1.default(qualifiedEnvironments !== undefined && qualifiedEnvironments.length > 0);
        if (this.policyName === 'random') {
            return utils_1.randomSelect(qualifiedEnvironments);
        }
        else if (this.policyName === 'round-robin') {
            return this.roundRobinSelect(qualifiedEnvironments, allEnvironments);
        }
        else if (this.policyName === 'recently-idle') {
            return this.recentlyIdleSelect(qualifiedEnvironments, allEnvironments);
        }
        else {
            throw new Error(`Unsupported schedule policy: ${this.policyName}`);
        }
    }
    recentlyIdleSelect(qualifiedEnvironments, allEnvironments) {
        const now = Date.now();
        let selectedEnvironment = undefined;
        let minTimeInterval = Number.MAX_SAFE_INTEGER;
        for (const environment of qualifiedEnvironments) {
            if (environment.latestTrialReleasedTime > 0 && (now - environment.latestTrialReleasedTime) < minTimeInterval) {
                selectedEnvironment = environment;
                minTimeInterval = now - environment.latestTrialReleasedTime;
            }
        }
        if (selectedEnvironment === undefined) {
            return this.roundRobinSelect(qualifiedEnvironments, allEnvironments);
        }
        selectedEnvironment.latestTrialReleasedTime = -1;
        return selectedEnvironment;
    }
    roundRobinSelect(qualifiedEnvironments, allEnvironments) {
        while (!qualifiedEnvironments.includes(allEnvironments[this.roundRobinIndex % allEnvironments.length])) {
            this.roundRobinIndex++;
        }
        return allEnvironments[this.roundRobinIndex++ % allEnvironments.length];
    }
    selectGPUsForTrial(gpuInfos, requiredGPUNum) {
        return gpuInfos.slice(0, requiredGPUNum);
    }
    allocateHost(requiredGPUNum, environment, gpuInfos, trialDetails) {
        assert_1.default(gpuInfos.length >= requiredGPUNum);
        const allocatedGPUs = this.selectGPUsForTrial(gpuInfos, requiredGPUNum);
        const defaultGpuSummary = environment.defaultGpuSummary;
        if (undefined === defaultGpuSummary) {
            throw new Error(`Environment ${environment.id} defaultGpuSummary shouldn't be undefined!`);
        }
        allocatedGPUs.forEach((gpuInfo) => {
            let num = defaultGpuSummary.assignedGpuIndexMap.get(gpuInfo.index);
            if (num === undefined) {
                num = 0;
            }
            defaultGpuSummary.assignedGpuIndexMap.set(gpuInfo.index, num + 1);
        });
        trialDetails.assignedGpus = allocatedGPUs;
        return {
            resultType: gpuData_1.ScheduleResultType.SUCCEED,
            environment: environment,
            gpuIndices: allocatedGPUs,
        };
    }
}
exports.GpuScheduler = GpuScheduler;
