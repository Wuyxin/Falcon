"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.GPUScheduler = void 0;
const fs_1 = __importDefault(require("fs"));
const os_1 = __importDefault(require("os"));
const path_1 = __importDefault(require("path"));
const log_1 = require("common/log");
const utils_1 = require("common/utils");
const util_1 = require("../common/util");
class GPUScheduler {
    gpuSummary;
    stopping;
    log;
    gpuMetricCollectorScriptFolder;
    constructor() {
        this.stopping = false;
        this.log = log_1.getLogger('GPUScheduler');
        this.gpuMetricCollectorScriptFolder = `${os_1.default.tmpdir()}/${os_1.default.userInfo().username}/nni/script`;
    }
    async run() {
        await this.runGpuMetricsCollectorScript();
        while (!this.stopping) {
            try {
                await this.updateGPUSummary();
            }
            catch (error) {
                this.log.error('Read GPU summary failed with error: ', error);
            }
            if (this.gpuSummary !== undefined && this.gpuSummary.gpuCount === 0) {
                throw new Error('GPU not available. Please check your CUDA configuration');
            }
            await utils_1.delay(5000);
        }
    }
    getAvailableGPUIndices(useActiveGpu, occupiedGpuIndexNumMap) {
        if (this.gpuSummary !== undefined) {
            if (process.platform === 'win32' || useActiveGpu) {
                return this.gpuSummary.gpuInfos.map((info) => info.index);
            }
            else {
                return this.gpuSummary.gpuInfos.filter((info) => occupiedGpuIndexNumMap.get(info.index) === undefined && info.activeProcessNum === 0 ||
                    occupiedGpuIndexNumMap.get(info.index) !== undefined)
                    .map((info) => info.index);
            }
        }
        return [];
    }
    getSystemGpuCount() {
        if (this.gpuSummary !== undefined) {
            return this.gpuSummary.gpuCount;
        }
        return undefined;
    }
    async stop() {
        this.stopping = true;
        try {
            const pid = await fs_1.default.promises.readFile(path_1.default.join(this.gpuMetricCollectorScriptFolder, 'pid'), 'utf8');
            await util_1.execKill(pid);
            await util_1.execRemove(this.gpuMetricCollectorScriptFolder);
        }
        catch (error) {
            this.log.error(`GPU scheduler error: ${error}`);
        }
    }
    async runGpuMetricsCollectorScript() {
        await util_1.execMkdir(this.gpuMetricCollectorScriptFolder, true);
        util_1.runGpuMetricsCollector(this.gpuMetricCollectorScriptFolder);
    }
    async updateGPUSummary() {
        const gpuMetricPath = path_1.default.join(this.gpuMetricCollectorScriptFolder, 'gpu_metrics');
        if (fs_1.default.existsSync(gpuMetricPath)) {
            const cmdresult = await util_1.execTail(gpuMetricPath);
            if (cmdresult !== undefined && cmdresult.stdout !== undefined) {
                this.gpuSummary = JSON.parse(cmdresult.stdout);
            }
            else {
                this.log.error('Could not get gpu metrics information!');
            }
        }
        else {
            this.log.warning('gpu_metrics file does not exist!');
        }
    }
}
exports.GPUScheduler = GPUScheduler;
