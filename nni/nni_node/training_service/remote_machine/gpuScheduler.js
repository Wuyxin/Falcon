"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.GPUScheduler = void 0;
const assert_1 = __importDefault(require("assert"));
const log_1 = require("common/log");
const utils_1 = require("common/utils");
const gpuData_1 = require("../common/gpuData");
class GPUScheduler {
    machineExecutorMap;
    log = log_1.getLogger('GPUScheduler');
    policyName = 'round-robin';
    roundRobinIndex = 0;
    configuredRMs = [];
    constructor(machineExecutorMap) {
        assert_1.default(machineExecutorMap.size > 0);
        this.machineExecutorMap = machineExecutorMap;
        this.configuredRMs = Array.from(machineExecutorMap.values(), manager => manager.rmMeta);
    }
    scheduleMachine(requiredGPUNum, trialJobDetail) {
        if (requiredGPUNum === undefined) {
            requiredGPUNum = 0;
        }
        assert_1.default(requiredGPUNum >= 0);
        const allRMs = Array.from(this.machineExecutorMap.values(), manager => manager.rmMeta);
        assert_1.default(allRMs.length > 0);
        const eligibleRM = allRMs.filter((rmMeta) => rmMeta.gpuSummary === undefined || requiredGPUNum === 0 || (requiredGPUNum !== undefined && rmMeta.gpuSummary.gpuCount >= requiredGPUNum));
        if (eligibleRM.length === 0) {
            return ({
                resultType: gpuData_1.ScheduleResultType.REQUIRE_EXCEED_TOTAL,
                scheduleInfo: undefined
            });
        }
        if (requiredGPUNum > 0) {
            const result = this.scheduleGPUHost(requiredGPUNum, trialJobDetail);
            if (result !== undefined) {
                return result;
            }
        }
        else {
            const allocatedRm = this.selectMachine(allRMs);
            return this.allocateHost(requiredGPUNum, allocatedRm, [], trialJobDetail);
        }
        this.log.warning(`Scheduler: trialJob id ${trialJobDetail.id}, no machine can be scheduled, return TMP_NO_AVAILABLE_GPU `);
        return {
            resultType: gpuData_1.ScheduleResultType.TMP_NO_AVAILABLE_GPU,
            scheduleInfo: undefined
        };
    }
    removeGpuReservation(trialJobId, trialJobMap) {
        const trialJobDetail = trialJobMap.get(trialJobId);
        if (trialJobDetail === undefined) {
            throw new Error(`could not get trialJobDetail by id ${trialJobId}`);
        }
        if (trialJobDetail.rmMeta !== undefined &&
            trialJobDetail.rmMeta.occupiedGpuIndexMap !== undefined &&
            trialJobDetail.gpuIndices !== undefined &&
            trialJobDetail.gpuIndices.length > 0) {
            for (const gpuInfo of trialJobDetail.gpuIndices) {
                const num = trialJobDetail.rmMeta.occupiedGpuIndexMap.get(gpuInfo.index);
                if (num !== undefined) {
                    if (num === 1) {
                        trialJobDetail.rmMeta.occupiedGpuIndexMap.delete(gpuInfo.index);
                    }
                    else {
                        trialJobDetail.rmMeta.occupiedGpuIndexMap.set(gpuInfo.index, num - 1);
                    }
                }
            }
        }
        trialJobDetail.gpuIndices = [];
        trialJobMap.set(trialJobId, trialJobDetail);
    }
    scheduleGPUHost(requiredGPUNum, trialJobDetail) {
        const totalResourceMap = this.gpuResourceDetection();
        const qualifiedRMs = [];
        totalResourceMap.forEach((gpuInfos, rmMeta) => {
            if (gpuInfos !== undefined && gpuInfos.length >= requiredGPUNum) {
                qualifiedRMs.push(rmMeta);
            }
        });
        if (qualifiedRMs.length > 0) {
            const allocatedRm = this.selectMachine(qualifiedRMs);
            const gpuInfos = totalResourceMap.get(allocatedRm);
            if (gpuInfos !== undefined) {
                return this.allocateHost(requiredGPUNum, allocatedRm, gpuInfos, trialJobDetail);
            }
            else {
                assert_1.default(false, 'gpuInfos is undefined');
            }
        }
        return undefined;
    }
    gpuResourceDetection() {
        const totalResourceMap = new Map();
        this.machineExecutorMap.forEach((executorManager, machineConfig) => {
            const rmMeta = executorManager.rmMeta;
            if (rmMeta.gpuSummary !== undefined) {
                const availableGPUs = [];
                const designatedGpuIndices = machineConfig.gpuIndices;
                if (designatedGpuIndices !== undefined) {
                    for (const gpuIndex of designatedGpuIndices) {
                        if (gpuIndex >= rmMeta.gpuSummary.gpuCount) {
                            throw new Error(`Specified GPU index not found: ${gpuIndex}`);
                        }
                    }
                }
                this.log.debug(`designated gpu indices: ${designatedGpuIndices}`);
                rmMeta.gpuSummary.gpuInfos.forEach((gpuInfo) => {
                    if (designatedGpuIndices === undefined || designatedGpuIndices.includes(gpuInfo.index)) {
                        if (rmMeta.occupiedGpuIndexMap !== undefined) {
                            const num = rmMeta.occupiedGpuIndexMap.get(gpuInfo.index);
                            if ((num === undefined && (!machineConfig.useActiveGpu && gpuInfo.activeProcessNum === 0 || machineConfig.useActiveGpu)) ||
                                (num !== undefined && num < machineConfig.maxTrialNumberPerGpu)) {
                                availableGPUs.push(gpuInfo);
                            }
                        }
                        else {
                            throw new Error(`occupiedGpuIndexMap initialize error!`);
                        }
                    }
                });
                totalResourceMap.set(rmMeta, availableGPUs);
            }
        });
        return totalResourceMap;
    }
    selectMachine(rmMetas) {
        assert_1.default(rmMetas !== undefined && rmMetas.length > 0);
        if (this.policyName === 'random') {
            return utils_1.randomSelect(rmMetas);
        }
        else if (this.policyName === 'round-robin') {
            return this.roundRobinSelect(rmMetas);
        }
        else {
            throw new Error(`Unsupported schedule policy: ${this.policyName}`);
        }
    }
    roundRobinSelect(rmMetas) {
        while (!rmMetas.includes(this.configuredRMs[this.roundRobinIndex % this.configuredRMs.length])) {
            this.roundRobinIndex++;
        }
        return this.configuredRMs[this.roundRobinIndex++ % this.configuredRMs.length];
    }
    selectGPUsForTrial(gpuInfos, requiredGPUNum) {
        return gpuInfos.slice(0, requiredGPUNum);
    }
    allocateHost(requiredGPUNum, rmMeta, gpuInfos, trialJobDetail) {
        assert_1.default(gpuInfos.length >= requiredGPUNum);
        const allocatedGPUs = this.selectGPUsForTrial(gpuInfos, requiredGPUNum);
        allocatedGPUs.forEach((gpuInfo) => {
            if (rmMeta.occupiedGpuIndexMap !== undefined) {
                let num = rmMeta.occupiedGpuIndexMap.get(gpuInfo.index);
                if (num === undefined) {
                    num = 0;
                }
                rmMeta.occupiedGpuIndexMap.set(gpuInfo.index, num + 1);
            }
            else {
                throw new Error(`Machine ${rmMeta.config.host} occupiedGpuIndexMap initialize error!`);
            }
        });
        trialJobDetail.gpuIndices = allocatedGPUs;
        trialJobDetail.rmMeta = rmMeta;
        return {
            resultType: gpuData_1.ScheduleResultType.SUCCEED,
            scheduleInfo: {
                rmMeta: rmMeta,
                cudaVisibleDevice: allocatedGPUs
                    .map((gpuInfo) => {
                    return gpuInfo.index;
                })
                    .join(',')
            }
        };
    }
}
exports.GPUScheduler = GPUScheduler;
