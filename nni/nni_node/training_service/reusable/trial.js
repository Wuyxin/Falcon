"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.TrialDetail = void 0;
class TrialDetail {
    id;
    status;
    submitTime;
    startTime;
    endTime;
    tags;
    url;
    workingDirectory;
    form;
    isEarlyStopped;
    environment;
    message;
    settings = {};
    nodes;
    assignedGpus;
    TRIAL_METADATA_DIR = ".nni";
    constructor(id, status, submitTime, workingDirectory, form) {
        this.id = id;
        this.status = status;
        this.submitTime = submitTime;
        this.workingDirectory = workingDirectory;
        this.form = form;
        this.tags = [];
        this.nodes = new Map();
    }
}
exports.TrialDetail = TrialDetail;
