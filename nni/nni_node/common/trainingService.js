"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.NNIManagerIpConfig = exports.TrainingServiceError = exports.TrainingService = void 0;
class TrainingServiceError extends Error {
    errCode;
    constructor(errorCode, errorMessage) {
        super(errorMessage);
        this.errCode = errorCode;
    }
    get errorCode() {
        return this.errCode;
    }
}
exports.TrainingServiceError = TrainingServiceError;
class TrainingService {
}
exports.TrainingService = TrainingService;
class NNIManagerIpConfig {
    nniManagerIp;
    constructor(nniManagerIp) {
        this.nniManagerIp = nniManagerIp;
    }
}
exports.NNIManagerIpConfig = NNIManagerIpConfig;
