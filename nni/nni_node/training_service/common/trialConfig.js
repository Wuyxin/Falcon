"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.TrialConfig = void 0;
class TrialConfig {
    command;
    codeDir;
    gpuNum;
    reuseEnvironment = true;
    constructor(command, codeDir, gpuNum) {
        this.command = command;
        this.codeDir = codeDir;
        this.gpuNum = gpuNum;
    }
}
exports.TrialConfig = TrialConfig;
