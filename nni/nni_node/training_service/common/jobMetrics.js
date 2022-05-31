"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.JobMetrics = void 0;
class JobMetrics {
    jobId;
    metrics;
    jobStatus;
    endTimestamp;
    constructor(jobId, metrics, jobStatus, endTimestamp) {
        this.jobId = jobId;
        this.metrics = metrics;
        this.jobStatus = jobStatus;
        this.endTimestamp = endTimestamp;
    }
}
exports.JobMetrics = JobMetrics;
