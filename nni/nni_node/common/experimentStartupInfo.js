"use strict";
var __importDefault = (this && this.__importDefault) || function (mod) {
    return (mod && mod.__esModule) ? mod : { "default": mod };
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.getDispatcherPipe = exports.isReadonly = exports.getPlatform = exports.isNewExperiment = exports.getBasePort = exports.getExperimentId = exports.setExperimentStartupInfo = exports.getExperimentStartupInfo = exports.ExperimentStartupInfo = void 0;
const strict_1 = __importDefault(require("assert/strict"));
const path_1 = __importDefault(require("path"));
let singleton = null;
class ExperimentStartupInfo {
    experimentId;
    newExperiment;
    basePort;
    logDir = '';
    logLevel;
    readonly;
    dispatcherPipe;
    platform;
    urlprefix;
    constructor(args) {
        this.experimentId = args.experimentId;
        this.newExperiment = (args.action === 'create');
        this.basePort = args.port;
        this.logDir = path_1.default.join(args.experimentsDirectory, args.experimentId);
        this.logLevel = args.logLevel;
        this.readonly = (args.action === 'view');
        this.dispatcherPipe = args.dispatcherPipe ?? null;
        this.platform = args.mode;
        this.urlprefix = args.urlPrefix;
    }
    static getInstance() {
        strict_1.default.notEqual(singleton, null);
        return singleton;
    }
}
exports.ExperimentStartupInfo = ExperimentStartupInfo;
function getExperimentStartupInfo() {
    return ExperimentStartupInfo.getInstance();
}
exports.getExperimentStartupInfo = getExperimentStartupInfo;
function setExperimentStartupInfo(args) {
    singleton = new ExperimentStartupInfo(args);
}
exports.setExperimentStartupInfo = setExperimentStartupInfo;
function getExperimentId() {
    return getExperimentStartupInfo().experimentId;
}
exports.getExperimentId = getExperimentId;
function getBasePort() {
    return getExperimentStartupInfo().basePort;
}
exports.getBasePort = getBasePort;
function isNewExperiment() {
    return getExperimentStartupInfo().newExperiment;
}
exports.isNewExperiment = isNewExperiment;
function getPlatform() {
    return getExperimentStartupInfo().platform;
}
exports.getPlatform = getPlatform;
function isReadonly() {
    return getExperimentStartupInfo().readonly;
}
exports.isReadonly = isReadonly;
function getDispatcherPipe() {
    return getExperimentStartupInfo().dispatcherPipe;
}
exports.getDispatcherPipe = getDispatcherPipe;
