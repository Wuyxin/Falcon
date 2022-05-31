"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.GPU_INFO_COLLECTOR_FORMAT_WINDOWS = exports.parseGpuIndices = exports.GPUSummary = exports.GPUInfo = exports.ScheduleResultType = void 0;
var ScheduleResultType;
(function (ScheduleResultType) {
    ScheduleResultType[ScheduleResultType["SUCCEED"] = 0] = "SUCCEED";
    ScheduleResultType[ScheduleResultType["TMP_NO_AVAILABLE_GPU"] = 1] = "TMP_NO_AVAILABLE_GPU";
    ScheduleResultType[ScheduleResultType["REQUIRE_EXCEED_TOTAL"] = 2] = "REQUIRE_EXCEED_TOTAL";
})(ScheduleResultType = exports.ScheduleResultType || (exports.ScheduleResultType = {}));
class GPUInfo {
    activeProcessNum;
    gpuMemUtil;
    gpuUtil;
    index;
    gpuMemTotal;
    gpuMemFree;
    gpuMemUsed;
    gpuType;
    constructor(activeProcessNum, gpuMemUtil, gpuUtil, index, gpuMemTotal, gpuMemFree, gpuMemUsed, gpuType) {
        this.activeProcessNum = activeProcessNum;
        this.gpuMemUtil = gpuMemUtil;
        this.gpuUtil = gpuUtil;
        this.index = index;
        this.gpuMemTotal = gpuMemTotal;
        this.gpuMemFree = gpuMemFree;
        this.gpuMemUsed = gpuMemUsed;
        this.gpuType = gpuType;
    }
}
exports.GPUInfo = GPUInfo;
class GPUSummary {
    gpuCount;
    timestamp;
    gpuInfos;
    constructor(gpuCount, timestamp, gpuInfos) {
        this.gpuCount = gpuCount;
        this.timestamp = timestamp;
        this.gpuInfos = gpuInfos;
    }
}
exports.GPUSummary = GPUSummary;
function parseGpuIndices(gpuIndices) {
    if (gpuIndices === undefined) {
        return undefined;
    }
    const indices = gpuIndices.split(',')
        .map((x) => parseInt(x, 10));
    if (indices.length > 0) {
        return new Set(indices);
    }
    else {
        throw new Error('gpuIndices can not be empty if specified.');
    }
}
exports.parseGpuIndices = parseGpuIndices;
exports.GPU_INFO_COLLECTOR_FORMAT_WINDOWS = `
$env:METRIC_OUTPUT_DIR="{0}"
$app = Start-Process "python" -ArgumentList "-m nni.tools.gpu_tool.gpu_metrics_collector" -passthru -NoNewWindow \
-redirectStandardOutput {0}\\stdout -redirectStandardError {0}\\stderr
Write $app.ID | Out-File {1} -NoNewline -encoding utf8
`;
